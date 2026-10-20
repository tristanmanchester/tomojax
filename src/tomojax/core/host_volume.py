"""Host volumes larger than the device, and larger than memory, accessed a z slab at a time.

Volumes are Fortran-ordered ``(nx, ny, nz)`` FP32 arrays, so a slab of whole
z planes is one contiguous block; work volumes can be anonymous files beside
a memory-mapped output. Pages of mapped volumes are released after each slab,
so a volume larger than memory never stays resident.
"""

from __future__ import annotations

import mmap
import os
from pathlib import Path
import tempfile
from typing import Any
import weakref

import numpy as np

_DESCRIPTORS: dict[int, int] = {}  # open files of the work volumes alive, by id


def scratch_directory(array: np.ndarray | None) -> Path | None:
    """The directory of a file-backed array, where work volumes beside it belong."""
    name = getattr(array, "filename", None)
    return None if name is None else Path(name).parent


def host_volume(shape: tuple[int, int, int], directory: Path | None = None) -> np.ndarray:
    """A zeroed Fortran-ordered FP32 work volume: in memory, or a temporary file in ``directory``.

    The temporary file has no name and is removed when the array is released.
    """
    if directory is None:
        return np.zeros(shape, np.float32, order="F")
    with tempfile.NamedTemporaryFile(dir=directory) as handle:
        array = np.memmap(handle.name, np.float32, "w+", shape=shape, order="F")
        descriptor = os.open(handle.name, os.O_RDWR)
    # The descriptor keeps the unlinked file open for slab reads and writes.
    _DESCRIPTORS[id(array)] = descriptor
    weakref.finalize(array, lambda key: os.close(_DESCRIPTORS.pop(key)), id(array))
    return array


def slab_file(array: np.ndarray) -> tuple[int, int] | None:
    """A new descriptor for a Fortran-ordered memmap's file, and the offset of its first byte.

    None for arrays that are not memory-mapped files. The caller closes the descriptor.
    """
    mapping = getattr(array, "_mmap", None)
    kept, name = _DESCRIPTORS.get(id(array)), getattr(array, "filename", None)
    if mapping is None or not array.flags.f_contiguous:
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
    if mapping is None:
        return
    first = start - start % mmap.PAGESIZE
    mapping.madvise(mmap.MADV_DONTNEED, first, min(len(mapping), stop or len(mapping)) - first)


def release_planes(array: np.ndarray, z0: int, z1: int) -> None:
    """:func:`release` the z planes ``[z0, z1)`` of a Fortran-ordered memmap."""
    mapping = getattr(array, "_mmap", None)
    if mapping is None or not array.flags.f_contiguous:
        return
    planes = array[:, :, z0:z1]
    first = planes.ctypes.data - np.frombuffer(mapping, np.uint8).ctypes.data
    release(array, first, first + planes.nbytes)


def fill(target: Any, initial: Any, *, depth: int = 32) -> Any:
    """Copy ``initial`` (a volume, a constant, or zero for None) into ``target`` by slabs.

    Memory-mapped pages of both are released after each slab, so a volume
    larger than memory can be initialized. Shared storage is left as it is.
    Returns ``target``.
    """
    if isinstance(initial, np.ndarray) and np.shares_memory(initial, target):
        return target
    for z0 in range(0, target.shape[2], depth):
        z1 = min(target.shape[2], z0 + depth)
        if initial is None or np.ndim(initial) == 0:
            target[:, :, z0:z1] = initial or 0
        else:
            target[:, :, z0:z1] = initial[:, :, z0:z1]
            release_planes(initial, z0, z1)
        release_planes(target, z0, z1)
    return target


def check_rows(data: np.ndarray, context: str, *, nonnegative: bool = False) -> None:
    """Raise unless every measurement is finite in FP32 (and nonnegative), a few views at a time.

    A memory-mapped stack's pages are released as they are checked.
    """
    for first in range(0, data.shape[0], 16):
        block = np.asarray(data[first : first + 16], np.float32)
        if not np.isfinite(block).all() or (nonnegative and block.min() < 0):
            kind = "finite and nonnegative" if nonnegative else "finite in FP32"
            raise ValueError(f"{context}: projections must be {kind}")
        release(data)
