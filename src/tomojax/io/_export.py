"""Plan and publish an export of one registered array."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
import errno
import fcntl
import math
import os
from pathlib import Path
import shutil
import tempfile

import msgspec
import numpy as np

from tomojax import progress
from tomojax.core.input_files import check_output, publishing as protect_inputs
from tomojax.errors import InvalidParameter
from tomojax.io._export_data import Blocks, Crop, ExportSource, Selection, Values
from tomojax.io._export_formats import FORMATS, Document, Format, format_for, slice_path


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
        if os.path.lexists(p) and not force:
            raise FileExistsError(f"output already exists: {p}")
    if fmt.layout == "directory":
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
        if os.path.lexists(p):
            backup = staged[p].parent / (p.name + ".previous")
            os.link(p, backup, follow_symlinks=False)
            undo.callback(backup.replace, p)
            if fmt.layout == "group" and p == paths[-1]:
                p.unlink()
    for p in paths:
        if force:
            new = not os.path.lexists(p)
            staged[p].replace(p)
            if new:
                undo.callback(p.unlink, missing_ok=True)
        else:
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
    """

    path: Path
    format: Format
    selection: Selection
    values: Values
    details: Document
    record: Document

    def write(self, staged: Path) -> Document:
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
        blocks = Blocks(self.selection, self.values)
        with progress.task("export", self.path.name, self.selection.stored_shape[0], "slices"):
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
        with tempfile.TemporaryDirectory(prefix=f".{path.name}.partial-", dir=path.parent) as tmp:
            staged = Path(tmp) / path.name
            try:
                receipt = self.write(staged)
            except OSError as exc:
                if exc.errno != errno.ENOSPC:
                    raise
                written = sum(p.stat().st_size for p in Path(tmp).rglob("*") if p.is_file())
                raise OSError(
                    errno.ENOSPC,
                    f"export {path} stopped after {written} bytes; "
                    f"{shutil.disk_usage(tmp).free} bytes free",
                ) from exc
            paths = self.format.paths(path)
            installs = dict(zip(paths, self.format.paths(staged), strict=True))
            with publishing(paths), ExitStack() as cleanup, ExitStack() as undo:
                install_export(installs, self.format, path, force=force, undo=undo, cleanup=cleanup)
                undo.pop_all()
            return receipt


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
    if not path.name or path.name == "..":
        raise InvalidParameter("path", "expected an output file or directory name")
    fmt = format_for(path)
    _check_export_paths(fmt, path)
    selection = Selection.select(source, crop, binning, axis)
    if fmt.spatial_only and any(a.frame != "volume" for a in source.axes):
        choices = ", ".join(f.name for f in FORMATS if not f.spatial_only)
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
            choices = ", ".join(f.name for f in FORMATS if name in f.options)
            raise InvalidParameter(
                name, f"{fmt.name} does not support {name}; choose from {choices}"
            )
    values = Values.resolve(selection, dtype, value_range)
    axes = [msgspec.to_builtins(a) for a in selection.axes]
    array = {
        "name": source.name,
        "shape": list(selection.shape),
        "dtype": values.dtype,
        "axes": axes,
        "unit": values.unit,
        "domain": source.domain,
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
        "format": fmt.name,
        "selection": selection_info,
        "values": {**asdict(values), "range": list(values.range) if values.range else None},
        "compress": ("compress" in fmt.options and source.domain == "mask")
        if compress is None
        else bool(compress and "compress" in fmt.options),
        "bigtiff": fmt.layout == "file" and fmt.ordered_pages and nbytes >= (4 << 30) - (64 << 20),
        "files": files,
    }
    record = {
        **source.record,
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
        "resumes": False,
    }
    return ExportPlan(path, fmt, selection, values, details, record)
