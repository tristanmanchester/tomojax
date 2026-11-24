"""File input tracking and atomic output replacement."""

from __future__ import annotations

from contextlib import contextmanager
import errno
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterator


# A worker executes one Python job at a time. Its code can load data on any thread.
_input_recorders: dict[object, set[str]] = {}


@contextmanager
def record_inputs(paths: set[str]) -> Iterator[None]:
    """Collect loader inputs across all threads for the duration of a worker call."""
    token = object()
    _input_recorders[token] = paths
    try:
        yield
    finally:
        del _input_recorders[token]


def read_path[T: (str, Path)](path: T) -> T:
    """Record the resolved input at the point of use, before the caller can change directory."""
    for paths in tuple(_input_recorders.values()):
        paths.add(str(Path(path).resolve()))
    return path


def virtual_mappings(data: Any) -> Iterator[Any]:
    """Expand HDF5's %b block number in virtual file and dataset names."""
    import h5py

    for source in data.virtual_sources():
        names = source.file_name, source.dset_name
        if not any("%" in name for name in names):
            yield source
            continue
        blocks = 1
        space = source.vspace
        if any("%b" in name.replace("%%", "") for name in names):
            start, stride, count, block = space.get_regular_hyperslab()
            axis = count.index(cast("Any", h5py).h5s.UNLIMITED)
            blocks = len(range(start[axis], data.shape[axis], stride[axis]))
            # Each source supplies one block along the unlimited dimension.
            count = list(count)
            count[axis] = 1
            space = space.copy()
            space.select_hyperslab(start, tuple(count), stride, block)
        for index in range(blocks):
            filename, dataset = (
                re.sub(r"%%|%b", lambda m, number=str(index): "%" if m[0] == "%%" else number, name)
                for name in names
            )
            yield source._replace(file_name=filename, dset_name=dataset, vspace=space)


def virtual_source_paths(data: Any) -> dict[Path, list[Any]]:
    """Resolve VDS names using HDF5's access prefix, file directory and cwd."""
    sources: dict[Path, list[Any]] = {}
    here = Path(data.file.filename)
    prefix = os.fsdecode(data.id.get_access_plist().get_virtual_prefix())
    parents = [Path(prefix)] if prefix else []
    parents.extend((here.parent, Path.cwd()))
    for source in virtual_mappings(data):
        name = Path(source.file_name)
        if name.is_absolute() and not name.exists():
            name = Path(name.name)
        candidates = (parent / name for parent in parents)
        path = (
            here
            if source.file_name == "."
            else next((file for file in candidates if file.is_file()), here.parent / name)
        )
        sources.setdefault(path, []).append(source)
    return sources


def backing_datasets(data: Any, *, require_pixels: bool) -> Iterator[Any]:
    """Visit mapped datasets, optionally rejecting missing detector pixels before a scan."""
    import h5py

    for path, mappings in virtual_source_paths(data).items():
        try:
            with h5py.File(read_path(path), "r") as backing:
                for source in mappings:
                    pixels = backing.get(source.dset_name)
                    if not isinstance(pixels, h5py.Dataset):
                        if require_pixels:
                            raise KeyError(source.dset_name)
                        continue  # missing sources contribute the VDS fill value
                    if require_pixels:
                        validate_mapping(path, source, pixels)
                    yield pixels
        except (OSError, KeyError) as exc:
            if require_pixels:
                raise ValueError(f"{data.name}: missing virtual detector pixels in {path}") from exc


def validate_mapping(path: Path, source: Any, pixels: Any) -> None:
    """Check that the mapped source selection fits the physical detector data."""
    import h5py

    if source.src_space.get_select_type() == cast("Any", h5py).h5s.SEL_ALL:
        missing = pixels.size < source.vspace.get_select_npoints()
    else:
        bounds = source.src_space.get_select_bounds()[1]
        missing = pixels.ndim != len(bounds) or any(
            n <= end for n, end in zip(pixels.shape, bounds, strict=True)
        )
    if missing:
        raise ValueError(f"{path}:{source.dset_name}: missing virtual detector pixels")


def read_dataset[T](
    dataset: T, *, require_pixels: bool = False, seen: set[tuple[str, str]] | None = None
) -> T:
    """Record resolved datasets and their VDS inputs without materialising their pixels."""
    if not (_input_recorders or require_pixels):
        return dataset
    data = cast("Any", dataset)  # h5py stubs omit VDS mappings.
    filename = read_path(str(data.file.filename))
    if data.is_virtual:
        seen = set() if seen is None else seen
        identity = (str(Path(filename).resolve()), data.name)
        if identity not in seen:
            seen.add(identity)
            for source in backing_datasets(data, require_pixels=require_pixels):
                read_dataset(source, require_pixels=require_pixels, seen=seen)
    return dataset


@contextmanager
def replaced_when_complete(path: str | os.PathLike[str], parts: str) -> Iterator[Path]:
    """A temporary file beside ``path`` that replaces it only once the write is complete.

    The temporary file is in the same directory, so on the same filesystem; on
    success it reaches the disk and moves onto ``path`` in one step, so a file
    already at ``path`` survives until the new one is whole. On any failure the
    temporary file is removed and nothing at ``path`` changes. A full disk raises
    ``OSError`` saying how much was written, the room left and ``parts``, what was
    being written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{target.name}.partial-", dir=target.parent)
    os.close(handle)
    mask = os.umask(0)
    os.umask(mask)
    temporary = Path(name)
    temporary.chmod(0o666 & ~mask)  # the mode an ordinary new file gets, not mkstemp's 0600
    try:
        yield temporary
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        temporary.replace(target)  # one step: the old file or the new, never neither
    except BaseException as err:
        written = temporary.stat().st_size if temporary.exists() else 0
        temporary.unlink(missing_ok=True)
        if _disk_full(err):
            free = shutil.disk_usage(target.parent).free
            raise OSError(
                errno.ENOSPC,
                f"saving {target} stopped after {written / 2**30:.1f} GiB written: "
                f"{target.parent} has {free / 2**30:.1f} GiB free. It was writing {parts}; "
                "free some space or save elsewhere",
            ) from err
        raise


def _disk_full(err: BaseException) -> bool:
    """Whether ``err``, or what caused it, is a full disk (HDF5 reports errno 28 in text)."""
    current: BaseException | None = err
    while current is not None:
        if isinstance(current, OSError) and current.errno == errno.ENOSPC:
            return True
        text = str(current)
        if "No space left on device" in text or "errno = 28" in text:
            return True
        current = current.__cause__ or current.__context__
    return False


__all__ = ["replaced_when_complete"]
