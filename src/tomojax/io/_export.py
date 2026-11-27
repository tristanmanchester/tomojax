"""Plan and publish an export of one registered array."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict, dataclass
import errno
import fcntl
import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile

import msgspec
import numpy as np

from tomojax import progress
from tomojax._data import array_record
from tomojax.core.input_files import check_output, publishing as protect_inputs
from tomojax.errors import InvalidParameter
from tomojax.io import _export_formats as formats
from tomojax.io._export_data import Blocks, Crop, ExportSource, Selection, Values
from tomojax.io._export_formats import Document, Format, format_for, slice_path
from tomojax.io._export_resume import Journal, continuing, partial_path, prior_state


def existing_parent(path: Path) -> Path:
    return next(parent for parent in path.absolute().parents if parent.is_dir())


@contextmanager
def publishing(paths: Sequence[Path]) -> Iterator[None]:
    """Lock each destination directory for the duration of publication.

    Parameters
    ----------
    paths : sequence of Path
        Destination paths with existing parents.

    Yields
    ------
    None
        The caller can install files until the context exits.
    """
    with ExitStack() as stack:
        for parent in sorted({p.parent.resolve() for p in paths}):
            fd = os.open(parent, os.O_RDONLY)
            stack.callback(os.close, fd)
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield


def inspect_export(path: Path) -> Document | None:
    """Read the record at an interchange destination.

    Parameters
    ----------
    path : Path
        Primary file or owned directory.

    Returns
    -------
    dict or None
        Export record if the destination contains a recognized export.
    """
    fmt = format_for(path)
    if fmt.layout == "directory" and _previous(path).is_dir():
        with publishing([path]):
            recover_directory(fmt, path)
    if fmt.layout == "group":
        paths = fmt.paths(path)
        if not any(p.exists() for p in paths):
            return None
        with publishing(paths):
            if not all(p.is_file() for p in paths) or not (record := fmt.read_record(path)):
                raise ValueError(
                    f"incomplete {fmt.name} export at {path}: missing record or payload"
                )
            return record
    if not path.exists() or path.is_dir() != (fmt.layout == "directory"):
        return None
    return fmt.read_record(path)


def _previous(path: Path) -> Path:
    return path.with_name(f".{path.name}.previous")


def _installation_paths(fmt: Format, path: Path) -> tuple[Path, ...]:
    paths = fmt.paths(path)
    return (*paths, _previous(path)) if fmt.layout == "directory" else paths


def _check_export_paths(fmt: Format, path: Path) -> None:
    for destination in _installation_paths(fmt, path):
        check_output(destination)


def recover_directory(fmt: Format, path: Path) -> None:
    """Restore a directory interrupted between renames while holding the publication lock."""
    aside = _previous(path)
    if not aside.exists():
        return
    if aside.is_symlink() or not aside.is_dir() or not fmt.read_record(aside, original=path):
        raise ValueError(f"unrecognized export backup: {aside}")
    if not path.exists():
        with protect_inputs(aside, path):
            aside.rename(path)
    elif fmt.read_record(path):
        with protect_inputs(aside):
            shutil.rmtree(aside)
    else:
        raise ValueError(f"incomplete export at {path}; previous export at {aside}")


def _published_member(fmt: Format, path: Path, staged: Path) -> bool:
    """An interrupted installation can already have linked this exact prepared file."""
    if not fmt.resumable or not path.exists() or path.is_symlink():
        return False
    if fmt.layout == "directory":
        before, after = fmt.record_path(staged), fmt.record_path(path)
        return before.is_file() and after.is_file() and before.samefile(after)
    return staged.is_file() and path.samefile(staged)


def install_export(
    staged: Mapping[Path, Path],
    fmt: Format,
    path: Path,
    *,
    force: bool,
    undo: ExitStack,
    cleanup: ExitStack,
) -> None:
    """Install staged files and register rollback actions.

    Parameters
    ----------
    staged : mapping of Path to Path
        Final destinations and their completed temporary paths.
    fmt : Format
        Destination layout.
    path : Path
        Primary destination.
    force : bool
        Replace an existing output.
    undo : ExitStack
        Rollback actions to discard on commit.
    cleanup : ExitStack
        Work to perform on either commit or rollback.
    """
    # Cleanup outlives rollback and removes the previous directory after commit.
    _ = cleanup.enter_context(protect_inputs(*_installation_paths(fmt, path)))
    paths = fmt.paths(path)
    if fmt.layout == "directory":
        recover_directory(fmt, path)
    for p in paths:
        if not staged[p].exists():
            raise FileNotFoundError(f"missing staged export {p}")
        if os.path.lexists(p) and not force and not _published_member(fmt, p, staged[p]):
            raise FileExistsError(f"output already exists: {p}")
    if fmt.layout == "directory":
        if not force and _published_member(fmt, path, staged[path]):
            return
        aside = _previous(path)
        if path.exists():
            if path.is_symlink() or not path.is_dir() or not fmt.read_record(path):
                raise ValueError(
                    f"refusing to replace a directory without its TomoJAX export record: {path}"
                )
            path.rename(aside)
            undo.callback(aside.rename, path)
            cleanup.callback(shutil.rmtree, aside, True)
        staged[path].rename(path)
        undo.callback(path.rename, staged[path])
        return
    # Backups of individual files are hard links. A single-file replacement never
    # vacates its public name. A group first withdraws the old commit record.
    for p in reversed(paths):
        if os.path.lexists(p) and force:
            backup = staged[p].parent / (p.name + ".previous")
            os.link(p, backup, follow_symlinks=False)
            cleanup.callback(backup.unlink, missing_ok=True)
            undo.callback(backup.replace, p)
            if fmt.layout == "group" and p == paths[-1]:
                p.unlink()
    for p in paths:
        if force:
            new = not os.path.lexists(p)
            staged[p].replace(p)
            if new:
                undo.callback(p.unlink, missing_ok=True)
        elif not _published_member(fmt, p, staged[p]):
            os.link(staged[p], p)
            undo.callback(p.unlink)


@dataclass
class ExportPlan:
    """A validated choice of destination, pixels and encoding.

    Parameters
    ----------
    path : Path
        Primary destination.
    format : Format
        Interchange writer and file layout.
    selection : Selection
        Registered source, bounds and written axis order.
    values : Values
        Stored type and scaling.
    details : dict
        Dry-run report.
    record : dict
        Metadata embedded in or written beside the output.
    resume_request : dict or None, optional
        The registered source and options that determine restartable bytes.
    """

    path: Path
    format: Format
    selection: Selection
    values: Values
    details: Document
    record: Document
    resume_request: Document | None = None

    def pending_paths(self) -> tuple[Path, ...]:
        """Public destinations still to be installed, including all unrelated existing outputs.

        Returns
        -------
        tuple of Path
            Destination names for exclusive creation checks.
        """
        paths = self.format.paths(self.path)
        if self.resume_request is None:
            return paths
        state = prior_state(self.path, self.resume_request)
        if state is None or not state.finished:
            return paths
        primary = partial_path(self.path) / "files" / self.path.name
        return tuple(
            path
            for path, staged in zip(paths, self.format.paths(primary), strict=True)
            if not _published_member(self.format, path, staged)
        )

    @contextmanager
    def staged(self, staged: Path) -> Iterator[Document]:
        """Prepare a receipt and hold the export lock until the caller installs its files.

        Parameters
        ----------
        staged : Path
            Private primary destination with the public basename.

        Yields
        ------
        dict
            Receipt for this generation.
        """
        if self.resume_request is None:
            yield self._write(staged)
            return
        with continuing(self.path, self.resume_request, asdict(self.values)) as journal:
            folder = journal.root / "files"
            with protect_inputs(folder):
                folder.mkdir(exist_ok=True)
            primary = folder / self.path.name
            receipt = self._write(primary, journal)
            expected = self.resume_request["source"]
            identity = self.selection.source.identity
            if isinstance(expected, list) and (identity is None or tuple(expected) != identity()):
                raise InvalidParameter(
                    "source", "the source file changed during export; rerun the export"
                )
            if not journal.state.finished:
                journal.dirty.update(self.format.paths(primary))
                journal.dirty.add(self.format.record_path(primary))
                journal.state.finished = True
                journal.save(force=True)
            if staged.parent == journal.root / "install":
                with protect_inputs(staged.parent):
                    if staged.parent.exists():
                        shutil.rmtree(staged.parent)
                    staged.parent.mkdir()
            # Keep the synced generation until the last public rename and manifest
            # have committed. Private hard links cost no additional payload storage.
            for source, target in zip(
                self.format.paths(primary), self.format.paths(staged), strict=True
            ):
                if source.is_dir():
                    shutil.copytree(source, target, copy_function=os.link)
                else:
                    os.link(source, target)
            yield receipt

    def _write(self, staged: Path, journal: Journal | None = None) -> Document:
        """Write all files into a staging area.

        Parameters
        ----------
        staged : Path
            Primary staging path with the final basename.

        Returns
        -------
        dict
            Receipt including byte and clipping counts.
        """
        blocks = Blocks(self.selection, self.values, journal=journal)
        if journal:
            blocks.below, blocks.above = journal.state.below, journal.state.above
        with progress.task("export", self.path.name, self.selection.stored_shape[0], "slices"):
            if journal is None or not journal.state.finished:
                self.format.writer(staged, blocks, self.record)
            progress.check_canceled()
        size = sum(
            p.stat().st_size
            for root in self.format.paths(staged)
            for p in (root.rglob("*") if root.is_dir() else (root,))
            if p.is_file()
        )
        return {
            **self.details,
            "bytes_written": size,
            "clipped": {"below": blocks.below, "above": blocks.above},
        }

    def publish(self, *, force: bool) -> Document:
        """Write and install the complete export.

        Parameters
        ----------
        force : bool
            Replace any previous export.

        Returns
        -------
        dict
            Receipt including byte and clipping counts.
        """
        path = self.path
        _check_export_paths(self.format, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with (
            nullcontext(partial_path(path) / "install")
            if self.resume_request is not None
            else tempfile.TemporaryDirectory(prefix=f".{path.name}.partial-", dir=path.parent)
        ) as tmp:
            staged = Path(tmp) / path.name
            try:
                with self.staged(staged) as receipt:
                    paths = self.format.paths(path)
                    installs = dict(zip(paths, self.format.paths(staged), strict=True))
                    with publishing(paths), ExitStack() as cleanup, ExitStack() as undo:
                        install_export(
                            installs, self.format, path, force=force, undo=undo, cleanup=cleanup
                        )
                        undo.pop_all()
                    return receipt
            except OSError as exc:
                if exc.errno != errno.ENOSPC:
                    raise
                written = sum(p.stat().st_size for p in Path(tmp).rglob("*") if p.is_file())
                raise OSError(
                    errno.ENOSPC,
                    f"export {path} stopped after {written} bytes; "
                    f"{shutil.disk_usage(path.parent).free} bytes free",
                ) from exc


def plan_export(
    path: Path,
    source: ExportSource,
    *,
    dtype: str | None = None,
    value_range: Sequence[float] | None = None,
    crop: Crop | None = None,
    binning: int = 1,
    axis: str | None = None,
    compress: bool | None = None,
) -> ExportPlan:
    """Validate and describe an export of a registered source.

    Parameters
    ----------
    path : Path
        Primary destination.
    source : ExportSource
        Three-dimensional pixels, registration and record.
    dtype : str, optional
        Stored scalar type.
    value_range : sequence of float, optional
        Physical values mapped to the integer endpoints.
    crop : sequence, optional
        Half-open intervals in source axis order.
    binning : int, optional
        Spatial mean-bin factor.
    axis : str, optional
        Axis to write first.
    compress : bool, optional
        Apply lossless compression with the format codec.

    Returns
    -------
    ExportPlan
        Resolved selection, value mapping, files and storage requirements.
    """
    selection = Selection.select(source, crop, binning, axis)
    path, fmt, settings = formats.destination(path, selection)
    _check_export_paths(fmt, path)
    if fmt.spatial_only and any(a.frame != "volume" for a in source.axes):
        choices = ", ".join(f.name for f in formats.FORMATS if not f.spatial_only)
        raise InvalidParameter("path", f"{fmt.name} needs a spatial volume; choose from {choices}")
    options = {
        "dtype": dtype,
        "range": value_range,
        "crop": crop,
        "binning": None if binning == 1 else binning,
        "axis": axis if selection.order != (0, 1, 2) else None,
    }
    for name, value in options.items():
        if value is not None and name not in fmt.options:
            choices = ", ".join(f.name for f in formats.FORMATS if name in f.options)
            raise InvalidParameter(
                name, f"{fmt.name} does not support {name}; choose from {choices}"
            )
    metadata = (
        source.record(selection.bounds, binning) if callable(source.record) else source.record
    )
    resume_request = None
    previous = None
    if fmt.resumable:
        identity = source.identity() if source.identity else None
        registration = {
            "array": array_record(
                source.shape, source.dtype.str, source.axes, source.unit, source.domain
            ),
            "record": {k: v for k, v in metadata.items() if k != "provenance"},
        }
        resume_request = {
            "registration": hashlib.sha256(
                msgspec.json.encode(registration, order="sorted")
            ).hexdigest(),
            "source": list(identity) if identity is not None else None,
            "member": source.name,
            "shape": list(source.shape),
            "input_dtype": source.dtype.str,
            "domain": source.domain,
            "format": fmt.name,
            "dtype": dtype,
            "bounds": [list(pair) for pair in selection.bounds],
            "binning": binning,
            "order": list(selection.order),
            "range": list(value_range) if value_range is not None else None,
            "compress": compress,
        }
        previous = prior_state(path, resume_request)
    values = (
        Values(**previous.values)
        if previous is not None
        else Values.resolve(selection, dtype, value_range)
    )
    array = {
        "name": source.name,
        **array_record(selection.shape, values.dtype, selection.axes, values.unit, source.domain),
    }
    nbytes = math.prod(selection.shape) * np.dtype(values.dtype).itemsize
    paths = fmt.paths(path)
    slices = fmt.layout == "directory" and fmt.ordered_pages
    files = {
        "count": selection.shape[0] + 1 if slices else len(paths),
        "first": str(slice_path(path, 0) if slices else paths[0]),
        "last": str(slice_path(path, selection.shape[0] - 1) if slices else paths[-1]),
        "record": str(fmt.record_path(path)),
    }
    selection_info = {
        "source_shape": list(source.shape),
        "axes": [a.name for a in selection.axes],
        "crop": [list(pair) for pair in selection.bounds],
        "binning": binning,
    }
    encoding = {
        **settings,
        "format": fmt.name,
        "selection": selection_info,
        "values": {**asdict(values), "range": list(values.range) if values.range else None},
        "compress": (
            "compress" in fmt.options and (fmt.compressed_by_default or source.domain == "mask")
        )
        if compress is None
        else bool(compress and "compress" in fmt.options),
        "bigtiff": fmt.layout == "file" and fmt.ordered_pages and nbytes >= (4 << 30) - (64 << 20),
        "files": files,
    }
    record = {
        **metadata,
        "format": 2,
        "kind": "export",
        "arrays": {source.name: array},
        "export": {
            **encoding,
            "request": {
                "dtype": dtype,
                "range": value_range,
                "axis": axis,
                "compress": compress,
                "crop": selection_info["crop"],
                "binning": binning,
            },
        },
    }
    details = {
        **encoding,
        **array,
        "paths": [str(p) for p in paths],
        "destination_exists": any(os.path.lexists(p) for p in paths),
        "payload_bytes": nbytes,
        "free_bytes": shutil.disk_usage(existing_parent(path)).free,
        "scratch_bytes": nbytes if fmt.ordered_pages and selection.order[0] != 0 else 0,
        "memory_budget": max(
            256 << 20,
            2 * math.prod(source.shape[1:]) * source.dtype.itemsize * selection.factors[0],
        ),
        "carries": fmt.carries,
        "resumes": resume_request is not None and resume_request["source"] is not None,
    }
    return ExportPlan(path, fmt, selection, values, details, record, resume_request)
