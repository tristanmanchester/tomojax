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
from typing import Any, Protocol
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


class SlabPayload(Protocol):
    """A stored block's immutable bytes, wherever they are held."""

    @property
    def nbytes(self) -> int:
        """Bytes stored."""
        ...

    @property
    def resident_nbytes(self) -> int:
        """Bytes held in host memory."""
        ...

    def read(self) -> np.ndarray:
        """The bytes as a ``uint8`` array, which the caller does not modify."""
        ...


class PayloadStorage(Protocol):
    """Where a :class:`SlabStore` puts the encoded bytes of the blocks it writes."""

    def put(self, data: np.ndarray) -> SlabPayload:
        """An immutable payload holding a copy of the contiguous ``uint8`` ``data``."""
        ...


type _Block = tuple[SlabPayload, bool]  # payload and whether it is compressed


class SlabSource(Protocol):
    """A fixed FP32 ``(nx, ny, nz)`` volume read a slab of z planes at a time."""

    @property
    def shape(self) -> tuple[int, int, int]:
        """The volume's shape."""
        ...

    def read(self, z0: int, z1: int) -> np.ndarray:
        """Planes ``[z0, z1)`` as a new array."""
        ...


class SlabStore:
    """An owned FP32 ``(nx, ny, nz)`` volume held as lossless blocks of ``depth`` z planes.

    Each block keeps the XOR of its FP32 words with ``reference`` (a fixed
    source of the same shape, such as another store or a volume computed plane
    by plane from a coarser one; +0 without one), compressed with LZ4 after a
    byte shuffle when that is smaller, so every word, NaN payloads and -0
    included, reads back exactly. Blocks equal to the reference are not stored.
    Stored blocks are immutable: :meth:`copy` shares them and writes replace
    them, so a copy is a cheap snapshot. Reads return new Fortran-ordered arrays
    and writes copy their input. ``store[:, :, z0:z1]`` reads and assigns whole
    planes; nothing converts the whole volume implicitly, and :meth:`materialize`
    does so on request. ``storage`` holds the encoded bytes of written blocks
    (private anonymous memory by default).
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        *,
        depth: int,
        reference: SlabSource | None = None,
        storage: PayloadStorage | None = None,
    ) -> None:
        if reference is not None and reference.shape != tuple(shape):
            raise ValueError(f"reference shape {reference.shape} differs from {tuple(shape)}")
        self.shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        self.depth, self.reference = depth, reference
        self.storage: PayloadStorage = storage or MEMORY_STORAGE
        self._blocks: list[_Block | None] = [None] * -(-shape[2] // depth)

    dtype = np.dtype(np.float32)

    @property
    def nbytes(self) -> int:
        """Bytes of the volume as FP32."""
        return 4 * self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def encoded_nbytes(self) -> int:
        """Bytes of the stored blocks, wherever they are."""
        return sum(b[0].nbytes for b in self._blocks if b is not None)

    @property
    def resident_nbytes(self) -> int:
        """Bytes of the stored blocks held in memory (not spilled)."""
        return sum(b[0].resident_nbytes for b in self._blocks if b is not None)

    def spill(self, directory: Path) -> None:
        """Move the blocks held in memory to an unnamed file in ``directory``.

        Each block is written once and read back on demand, without staying
        in the page cache; later writes are held in memory again. Copies made
        before keep their memory blocks, and copies made after share the file,
        which closes when no store refers to it.
        """
        target = None
        for k, stored in enumerate(self._blocks):
            if stored is not None and stored[0].resident_nbytes:
                target = target or _SpillFile(directory)
                self._blocks[k] = (target.put(stored[0].read()), stored[1])

    def __array__(self, *_: Any, **__: Any) -> np.ndarray:
        raise TypeError("SlabStore is not converted implicitly; call materialize()")

    def copy(self) -> SlabStore:
        """A snapshot sharing the stored blocks; later writes to either leave the other."""
        result = SlabStore(
            self.shape, depth=self.depth, reference=self.reference, storage=self.storage
        )
        result._blocks = self._blocks.copy()
        return result

    def read(self, z0: int, z1: int) -> np.ndarray:
        """Planes ``[z0, z1)`` as a new Fortran-ordered array."""
        out = np.empty((*self.shape[:2], z1 - z0), np.float32, order="F")
        for k in range(z0 // self.depth, -(-z1 // self.depth)):
            b0, b1 = k * self.depth, min(self.shape[2], (k + 1) * self.depth)
            lo, hi = max(z0, b0), min(z1, b1)
            out[:, :, lo - z0 : hi - z0] = self._block(k)[:, :, lo - b0 : hi - b0]
        return out

    def write(self, z0: int, z1: int, values: Any) -> None:
        """Store ``values`` as planes ``[z0, z1)``; ``values`` is copied, not kept."""
        values = np.asarray(values, np.float32)
        for k in range(z0 // self.depth, -(-z1 // self.depth)):
            b0, b1 = k * self.depth, min(self.shape[2], (k + 1) * self.depth)
            lo, hi = max(z0, b0), min(z1, b1)
            if lo == b0 and hi == b1:
                block = np.asfortranarray(values[:, :, lo - z0 : hi - z0])
            else:
                block = self._block(k)
                block[:, :, lo - b0 : hi - b0] = values[:, :, lo - z0 : hi - z0]
            self._blocks[k] = self._encode(block, b0, b1)

    def materialize(self, out: np.ndarray | None = None, *, consume: bool = False) -> np.ndarray:
        """The whole volume, in ``out`` or a new Fortran-ordered array.

        With ``consume`` each block is released once copied, so the peak stays
        near the dense volume; the store is then empty and must not be used.
        Copies keep their blocks.
        """
        if out is None:
            out = np.empty(self.shape, np.float32, order="F")
        for k, z0 in enumerate(range(0, self.shape[2], self.depth)):
            z1 = min(self.shape[2], z0 + self.depth)
            out[:, :, z0:z1] = self.read(z0, z1)
            if consume:
                self._blocks[k] = None
        if consume:
            self._blocks, self.reference = [], None
        return out

    def __getitem__(self, key: tuple[slice, slice, slice | int]) -> np.ndarray:
        z0, z1 = self._planes(key)
        planes = self.read(z0, z1)
        return planes[:, :, 0] if isinstance(key[2], int | np.integer) else planes

    def __setitem__(self, key: tuple[slice, slice, slice | int], values: Any) -> None:
        z0, z1 = self._planes(key)
        values = np.asarray(values, np.float32)
        self.write(z0, z1, values[:, :, None] if values.ndim == 2 else values)

    def _planes(self, key: Any) -> tuple[int, int]:
        whole = slice(None)
        if not (isinstance(key, tuple) and len(key) == 3 and key[0] == whole == key[1]):
            raise IndexError("SlabStore is indexed by whole planes: store[:, :, z0:z1]")
        if isinstance(key[2], int | np.integer):
            z = range(self.shape[2])[key[2]]
            return z, z + 1
        z0, z1, step = key[2].indices(self.shape[2])
        if step != 1:
            raise IndexError("SlabStore reads contiguous planes")
        return z0, max(z0, z1)

    def _block(self, k: int) -> np.ndarray:
        b0, b1 = k * self.depth, min(self.shape[2], (k + 1) * self.depth)
        shape = (*self.shape[:2], b1 - b0)
        stored = self._blocks[k]
        payload = None if stored is None else stored[0].read()
        if stored is None or payload is None:
            block = np.zeros(shape, np.float32, order="F")
        elif stored[1]:
            import imagecodecs

            block = np.empty(shape, np.float32, order="F")
            flat = block.reshape(-1, order="F").view(np.uint8)
            imagecodecs.blosc2_decode(payload, out=memoryview(flat), numthreads=4)
        else:
            block = np.array(payload.view(np.float32).reshape(shape, order="F"), order="F")
        if self.reference is not None:
            words = block.reshape(-1, order="F").view(np.uint32)
            words ^= self.reference.read(b0, b1).reshape(-1, order="F").view(np.uint32)
        return block

    def _encode(self, block: np.ndarray, b0: int, b1: int) -> _Block | None:
        words = np.array(block.reshape(-1, order="F").view(np.uint32))
        if self.reference is not None:
            words ^= self.reference.read(b0, b1).reshape(-1, order="F").view(np.uint32)
        if not words.any():
            return None
        import imagecodecs

        packed = imagecodecs.blosc2_encode(
            words, level=9, compressor="lz4", typesize=4, numthreads=4,
            shuffle=imagecodecs.BLOSC2.FILTER.SHUFFLE, splitmode=imagecodecs.BLOSC2.SPLIT.NEVER,
        )  # fmt: skip
        compressed = len(packed) < words.nbytes
        data = np.frombuffer(packed, np.uint8) if compressed else words.view(np.uint8)
        return self.storage.put(data), compressed


class _Memory:
    """A copy of a block's bytes in its own private anonymous mapping, freed with it."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = _owned(data)
        self.nbytes = self.resident_nbytes = self.data.nbytes

    def read(self) -> np.ndarray:
        return self.data


class _MemoryStorage:
    def put(self, data: np.ndarray) -> SlabPayload:
        return _Memory(data)


MEMORY_STORAGE: PayloadStorage = _MemoryStorage()
"""The default storage: each payload in its own private anonymous mapping, read without a copy."""


class _SpillFile:
    """An unnamed file of spilled blocks, closed with its last block."""

    def __init__(self, directory: Path) -> None:
        with tempfile.NamedTemporaryFile(dir=directory) as handle:
            self.descriptor = os.open(handle.name, os.O_RDWR)
        self.size = 0
        weakref.finalize(self, os.close, self.descriptor)

    def put(self, data: np.ndarray) -> SlabPayload:
        write_at(self.descriptor, data, self.size, drop=True)
        self.size += data.nbytes
        return _Spilled(self, self.size - data.nbytes, data.nbytes)


class _Spilled:
    """A block's bytes at ``position`` in a spill file."""

    resident_nbytes = 0

    def __init__(self, file: _SpillFile, position: int, nbytes: int) -> None:
        self.file, self.position, self.nbytes = file, position, nbytes

    def read(self) -> np.ndarray:
        data = np.empty(self.nbytes, np.uint8)
        read_at(self.file.descriptor, data, self.position, drop=True)
        return data


def _owned(data: Any) -> np.ndarray:
    """A read-only copy of ``data``'s bytes in its own private anonymous mapping, freed with it."""
    source = np.frombuffer(data, np.uint8)
    flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS  # not shared memory
    target = np.frombuffer(mmap.mmap(-1, max(1, source.size), flags=flags), np.uint8)
    target = target[: source.size]
    target[:] = source
    target.flags.writeable = False
    return target
