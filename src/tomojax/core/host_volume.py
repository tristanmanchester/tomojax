"""Host volumes larger than the device, and larger than memory, accessed a z slab at a time.

Volumes are Fortran-ordered ``(nx, ny, nz)`` FP32 arrays, so a slab of whole
z planes is one contiguous block; work volumes can be anonymous files beside
a memory-mapped output. Pages of mapped volumes are released after each slab,
so a volume larger than memory never stays resident.
"""

from __future__ import annotations

import mmap
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


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
    # The mapping keeps the unlinked file alive after its descriptor closes.
    with tempfile.TemporaryFile(dir=directory) as handle:
        return np.memmap(handle, np.float32, "w+", shape=shape, order="F")


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
