"""File input tracking and atomic output replacement."""

from __future__ import annotations

from contextlib import contextmanager, suppress
import errno
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

    import h5py


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


def source_file(here: Path, filename: str, variable: str, prefix: str = "") -> Path:
    """Follow HDF5's external-link and virtual-file search order."""
    import h5py

    name = Path(filename)
    candidates = []
    if name.is_absolute():
        candidates.append(name)
        name = Path(name.name)
    parents = [Path(p) for p in os.environ.get(variable, "").split(os.pathsep) if p]
    if prefix:
        parents.append(Path(prefix))
    parents.extend((here.parent, Path.cwd(), here.resolve().parent))
    candidates.extend(parent / name for parent in parents)
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            with suppress(OSError), h5py.File(candidate, "r"):
                return candidate
    return here.parent / name


def virtual_source_paths(data: Any) -> dict[Path, list[Any]]:
    """Group VDS mappings by the physical file selected by HDF5."""
    sources: dict[Path, list[Any]] = {}
    here = Path(data.file.filename)
    prefix = os.fsdecode(data.id.get_access_plist().get_virtual_prefix())
    for source in virtual_mappings(data):
        path = (
            here
            if source.file_name == "."
            else source_file(here, source.file_name, "HDF5_VDS_PREFIX", prefix)
        )
        sources.setdefault(path, []).append(source)
    return sources


@contextmanager
def read_hdf5(path: str | Path, **options: Any) -> Iterator[h5py.File]:
    """Open a read-only file and record the worker's complete storage dependencies."""
    import h5py

    with h5py.File(path, "r", **options) as file:
        record_hdf5_inputs(file)
        yield file


def record_hdf5_inputs(file: Any) -> None:
    """Walk links and dataset storage metadata without reading pixel arrays."""
    import h5py

    if not _input_recorders:
        return
    pending = hdf5_sources(file)
    root = read_path(Path(file.filename))
    seen = {(root.resolve(), root.parent.resolve())}
    while pending:
        here = read_path(pending.pop())
        identity = (here.resolve(), here.parent.resolve())
        if identity in seen or not here.is_file():
            continue
        seen.add(identity)
        try:
            backing = h5py.File(here, "r")
        except OSError:
            continue  # A dangling link has no further dependencies.
        with backing:
            pending.update(hdf5_sources(backing))


def hdf5_sources(file: Any) -> set[Path]:
    """Record raw storage and return the HDF5 files linked from this container."""
    import h5py

    sources: set[Path] = set()
    here = Path(file.filename)

    def visit(name: str, link: Any) -> None:
        if isinstance(link, cast("Any", h5py).ExternalLink):
            sources.add(source_file(here, link.filename, "HDF5_EXT_PREFIX"))
        elif isinstance(link, cast("Any", h5py).HardLink):
            data = file[name]
            if isinstance(data, h5py.Dataset):
                layout = cast("Any", data)  # h5py stubs omit external and virtual layouts.
                if layout.is_virtual:
                    sources.update(virtual_source_paths(layout))
                if layout.external:
                    # The opened dataset's access plist includes HDF5_EXTFILE_PREFIX
                    # with ${ORIGIN} expanded by HDF5 itself.
                    prefix = os.fsdecode(layout.id.get_access_plist().get_efile_prefix())
                    for filename, _, _ in layout.external:
                        read_path(Path(prefix) / filename)

    file.visititems_links(visit)
    return sources


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


def validate_virtual_pixels(data: Any, seen: set[tuple[str, str]] | None = None) -> None:
    """Check every physical source of a virtual detector before accepting the stack."""
    import h5py

    if not data.is_virtual:
        return
    seen = set() if seen is None else seen
    identity = (str(Path(data.file.filename).resolve()), data.name)
    if identity in seen:
        return
    seen.add(identity)
    for path, mappings in virtual_source_paths(data).items():
        try:
            with h5py.File(path, "r") as backing:
                for source in mappings:
                    pixels = backing.get(source.dset_name)
                    if not isinstance(pixels, h5py.Dataset):
                        raise KeyError(source.dset_name)
                    validate_mapping(path, source, pixels)
                    validate_virtual_pixels(pixels, seen)
        except (OSError, KeyError) as exc:
            raise ValueError(f"{data.name}: missing virtual detector pixels in {path}") from exc


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
