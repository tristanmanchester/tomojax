"""Host volumes larger than the device, and larger than memory, accessed a z slab at a time.

Volumes are C-ordered ``(nz, ny, nx)`` FP32 arrays, so a slab of whole
z planes is one contiguous block; work volumes can be anonymous files beside
a memory-mapped output. Pages of mapped volumes are released after each slab,
so a volume larger than memory never stays resident.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import math
import mmap
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import TYPE_CHECKING, Any
import weakref

import numpy as np

from tomojax import progress

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import DTypeLike

_DESCRIPTORS: dict[int, int] = {}  # open files of the work volumes alive, by id


_OWN_CGROUP, _CGROUP_MOUNTS = Path("/proc/self/cgroup"), Path("/proc/self/mountinfo")


def _cgroup_room_bytes() -> int | None:
    """Conservative no-reclaim room under visible cgroup v2 limits.

    ``memory.high`` is a throttle boundary, not a hard allocation limit. Current
    usage includes reclaimable page cache, so this can underestimate capacity.
    Hidden ancestors cannot be measured from inside a cgroup namespace.
    """
    try:
        paths = [
            line[3:] for line in _OWN_CGROUP.read_text().splitlines() if line.startswith("0::")
        ]
        mounts = _CGROUP_MOUNTS.read_text().splitlines()
    except OSError:
        return None
    if len(paths) != 1 or not paths[0].startswith("/") or ".." in Path(paths[0]).parts:
        return None
    own, rooms = Path(paths[0]), []
    for entry in mounts:
        header, separator, filesystem = entry.partition(" - ")
        fields = header.split()
        if not separator or not filesystem.startswith("cgroup2 ") or len(fields) < 5:
            continue
        root, mount = [
            Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), text)) for text in fields[3:5]
        ]
        if not own.is_relative_to(root) or not mount.is_absolute():
            continue
        # Every level from the process's group up to the mount has its own limits.
        relative = own.relative_to(root)
        group = mount / relative
        for level in (group, *group.parents[: len(relative.parts)]):
            for limit in ("memory.high", "memory.max"):
                try:
                    text = (level / limit).read_text().strip()
                    if text != "max":
                        used = int((level / "memory.current").read_text())
                        rooms.append(max(0, int(text) - used))
                except (OSError, ValueError):
                    pass
    return min(rooms, default=None)


def _system_available_bytes() -> int | None:
    """The machine's available memory, including Linux's reclaimable page cache."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    try:
        names = getattr(os, "sysconf_names", {})
        if "SC_AVPHYS_PAGES" not in names or "SC_PAGE_SIZE" not in names:
            return None
        avail_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if avail_pages <= 0 or page_size <= 0:
            return None
        return avail_pages * page_size
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def host_available_memory_bytes() -> int | None:
    """Return available host bytes, bounded by any Linux cgroup memory limit.

    Returns
    -------
    int or None
        Available host memory, including reclaimable page cache, or None when
        neither the operating system nor a cgroup reports capacity.
    """
    known = [n for n in (_system_available_bytes(), _cgroup_room_bytes()) if n is not None]
    return min(known, default=None)


def mapped_storage(array: object) -> np.memmap | None:
    """Find mapped ownership through NumPy and buffer-protocol views, without reading pixels."""
    current: object = array
    while isinstance(current, np.ndarray | memoryview):
        if isinstance(current, np.memmap):
            return current
        current = current.obj if isinstance(current, memoryview) else current.base
    return None


def scratch_directory(array: object) -> Path | None:
    """The directory of a file-backed array, where work volumes beside it belong; else None."""
    name = getattr(array, "filename", None)
    return None if name is None else Path(name).parent


# A consumer retaining multiple intermediate stacks may set a smaller resident limit.
resident_limit = ContextVar[int | None]("host_array_resident_limit", default=None)


class HostOutputs:
    """File-backed arrays owned by a producer until it returns them."""

    def __init__(self) -> None:
        self.files: list[weakref.ReferenceType[np.memmap]] = []

    def own(self, array: object) -> None:
        """Register a file-backed output made by this producer."""
        if isinstance(array, np.memmap) and id(array) in _DESCRIPTORS:
            self.files.append(weakref.ref(array))

    def allocate(self, shape: tuple[int, ...]) -> np.ndarray:
        """Allocate and register an output before filling it."""
        progress.check_canceled()
        array = host_array(shape)
        self.own(array)
        return array


@contextmanager
def discard_failed_outputs() -> Iterator[HostOutputs]:
    """Release the producer's temporary file mappings on error, even with a live traceback."""
    outputs = HostOutputs()
    try:
        yield outputs
    except BaseException:
        for ref in outputs.files:
            close_work_file(ref())
        raise


def host_array(
    shape: tuple[int, ...], *, dtype: DTypeLike = np.float32, copies: float = 1
) -> np.ndarray:
    """A zeroed C-ordered array: in memory, or a temporary file when it is large.

    The array goes to an unlinked file in the standard temporary directory
    (``TMPDIR``) when ``copies`` arrays of ``shape`` would take more than half of
    the available host memory; otherwise it is an ordinary array. Every large host
    array the library makes (a corrected stack, a solver's output and work
    volumes, the alignment's stacks) follows this one rule.
    """
    available = host_available_memory_bytes()
    need = copies * math.prod(shape) * np.dtype(dtype).itemsize
    limit = resident_limit.get()
    large = (available is not None and need > 0.5 * available) or (
        limit is not None and need > limit
    )
    return host_volume(shape, Path(tempfile.gettempdir()) if large else None, dtype=dtype)


def host_volume(
    shape: tuple[int, ...], directory: Path | None = None, *, dtype: DTypeLike = np.float32
) -> np.ndarray:
    """A zeroed C-ordered array (FP32 by default): in memory, or a temporary file in ``directory``.

    The temporary file has no name and is removed when the array is released.
    A file is made only when ``directory`` has room for all of it: a mapped page
    written past a full disk would end the process with a bus error.

    Raises
    ------
    MemoryError
        If ``directory`` has less free space than the array needs.
    """
    if directory is None:
        # Anonymous pages, zero and populated as they are written: no NumPy arena and no
        # eager huge-page advice (712cab8e: host FBP 16.1 s against 22.1 s on I12).
        size = max(1, math.prod(shape) * np.dtype(dtype).itemsize)
        storage = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        return np.ndarray(shape, dtype, buffer=storage, order="C")
    _require_room(shape, directory, dtype)
    with tempfile.NamedTemporaryFile(dir=directory) as handle:
        array = np.memmap(handle.name, dtype, "w+", shape=shape, order="C")
        descriptor = os.open(handle.name, os.O_RDWR)
    # The descriptor keeps the unlinked file open for slab reads and writes.
    _DESCRIPTORS[id(array)] = descriptor
    weakref.finalize(array, _close_descriptor, id(array))
    return array


def _close_descriptor(key: int) -> None:
    descriptor = _DESCRIPTORS.pop(key, None)
    if descriptor is not None:
        os.close(descriptor)


def close_work_file(array: object) -> None:
    """Close a discarded private mapped volume even if an exception retains its frame."""
    if isinstance(array, np.memmap) and id(array) in _DESCRIPTORS:
        _close_descriptor(id(array))
        mapping = array.base
        assert isinstance(mapping, mmap.mmap)
        mapping.close()


def _require_room(shape: tuple[int, ...], directory: Path, dtype: DTypeLike = np.float32) -> None:
    """Raise a plain error when ``directory`` cannot hold an array of ``shape`` and ``dtype``."""
    need = math.prod(shape) * np.dtype(dtype).itemsize
    free = shutil.disk_usage(directory).free
    if need <= free:
        return
    available = host_available_memory_bytes()
    memory = "an unknown amount" if available is None else f"{available / 2**30:.1f} GiB"
    raise MemoryError(
        f"a {' x '.join(map(str, shape))} {np.dtype(dtype).name} array needs "
        f"{need / 2**30:.1f} GiB; "
        f"host memory has {memory} available, so it is held in a temporary file in "
        f"{directory}, which has {free / 2**30:.1f} GiB free. Set TMPDIR to a directory "
        "on a disk with room (a RAM disk such as /tmp on some systems is memory, not "
        "disk), or make the array smaller: binning=, fewer views "
        "(tj.load_frames(path).selected(...)), or a smaller grid."
    )


def _owned_descriptor(array: np.ndarray) -> int | None:
    """Find owned file storage through mapped and ordinary NumPy views."""
    current: object = array
    while isinstance(current, np.ndarray):
        descriptor = _DESCRIPTORS.get(id(current))
        if descriptor is not None:
            return descriptor
        current = current.base
    return None


def file_identity(array: np.memmap) -> tuple[int, int] | None:
    """Device/inode identity of a named or privately owned unlinked mapped file."""
    descriptor, name = _owned_descriptor(array), array.filename
    if descriptor is not None:
        status = os.fstat(descriptor)
    elif name is not None:
        status = Path(name).stat()
    else:
        return None
    return status.st_dev, status.st_ino


def slab_file(array: np.ndarray) -> tuple[int, int] | None:
    """A new descriptor for a C-ordered memmap's file, and the offset of its first byte.

    None for arrays that are not memory-mapped files. The caller closes the descriptor.
    """
    mapping = getattr(array, "_mmap", None)
    kept, name = _owned_descriptor(array), getattr(array, "filename", None)
    if mapping is None or not array.flags.c_contiguous:
        return None
    if kept is not None:
        descriptor = os.dup(kept)
    elif name is not None:
        descriptor = os.open(name, os.O_RDWR)
    else:
        return None
    offset = int(getattr(array, "offset", 0))
    start = offset - offset % mmap.ALLOCATIONGRANULARITY
    return descriptor, start + array.ctypes.data - np.frombuffer(mapping, np.uint8).ctypes.data


def read_at(descriptor: int, out: np.ndarray, position: int, *, drop: bool = False) -> None:
    """Fill the contiguous ``out`` from the file at byte ``position``.

    With ``drop`` the bytes read leave the page cache.
    """
    view, done = memoryview(out).cast("B"), 0
    while done < len(view):
        count = os.preadv(descriptor, [view[done:]], position + done)
        if count == 0:
            raise EOFError(f"volume file ends before byte {position + len(view)}")
        done += count
    if drop:
        os.posix_fadvise(descriptor, position, len(view), os.POSIX_FADV_DONTNEED)


def write_at(descriptor: int, block: np.ndarray, position: int, *, drop: bool = False) -> None:
    """Write the contiguous ``block`` to the file at byte ``position``.

    With ``drop`` the bytes reach the disk and leave the page cache before
    this returns, so writes to a slow disk cannot fill memory.
    """
    view, done = memoryview(block).cast("B"), 0
    while done < len(view):
        done += os.pwrite(descriptor, view[done:], position + done)
    if drop:
        os.fdatasync(descriptor)
        os.posix_fadvise(descriptor, position, len(view), os.POSIX_FADV_DONTNEED)


def release(array: np.ndarray, start: int = 0, stop: int | None = None) -> None:
    """Unmap the pages of bytes ``[start, stop)`` of a memmap's mapping from this process.

    The page cache keeps their contents; this only stops them counting as this
    process's resident memory. Other arrays are left alone.
    """
    mapping = getattr(array, "_mmap", None)
    # Dropping private dirty pages of a copy-on-write input would undo its edits.
    if mapping is None or getattr(array, "mode", None) == "c":
        return
    first = start - start % mmap.PAGESIZE
    mapping.madvise(mmap.MADV_DONTNEED, first, min(len(mapping), stop or len(mapping)) - first)


def release_planes(array: np.ndarray, z0: int, z1: int) -> None:
    """:func:`release` the z planes ``[z0, z1)`` of a C-ordered memmap."""
    owner = mapped_storage(array)
    if owner is None or not array.flags.c_contiguous:
        return
    mapping = getattr(owner, "_mmap", None)
    if mapping is None:
        return
    planes = array[z0:z1, :, :]
    first = planes.ctypes.data - np.frombuffer(mapping, np.uint8).ctypes.data
    release(owner, first, first + planes.nbytes)


def fill(target: Any, initial: Any, *, depth: int = 32) -> Any:
    """Copy ``initial`` (a volume, a constant, or zero for None) into ``target`` by slabs.

    Memory-mapped pages of both are released after each slab, so a volume
    larger than memory can be initialized. Shared storage is left as it is.
    Returns ``target``.
    """
    if isinstance(initial, np.ndarray) and np.shares_memory(initial, target):
        return target
    for z0 in range(0, target.shape[0], depth):
        z1 = min(target.shape[0], z0 + depth)
        if initial is None or np.ndim(initial) == 0:
            target[z0:z1, :, :] = initial or 0
        else:
            target[z0:z1, :, :] = initial[z0:z1, :, :]
            release_planes(initial, z0, z1)
        release_planes(target, z0, z1)
    return target


def check_rows(data: np.ndarray, context: str, *, nonnegative: bool = False) -> None:
    """Raise unless every measurement is finite in FP32 (and nonnegative), a few views at a time.

    A memory-mapped stack's pages are released as they are checked.
    """
    from tomojax.errors import InvalidParameter

    for first in range(0, data.shape[0], 16):
        progress.check_canceled()
        block = np.asarray(data[first : first + 16], np.float32)
        if not np.isfinite(block).all() or (nonnegative and block.min() < 0):
            kind = "finite and nonnegative" if nonnegative else "finite in FP32"
            raise InvalidParameter("source", f"{context}: projections must be {kind}")
        release(data)
