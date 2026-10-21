"""Bounded device storage for immutable encoded slabs."""

from __future__ import annotations

from typing import TYPE_CHECKING
import weakref

import numpy as np

from .host_volume import MEMORY_STORAGE

if TYPE_CHECKING:
    import jax

    from tomojax._typed_arrays import Device

    from .host_volume import PayloadStorage, SlabPayload


class DeviceStorage:
    """Share a byte-limited device tier across slab stores, with host fallback.

    Pass the same owner as ``storage=`` to multiple :class:`SlabStore` volumes
    to share one device budget. Encoded blocks that do not fit stay in host
    memory; ``reserve`` can add capacity after other work arrays are released.

    Copies finish before returning, so callers may immediately reuse inputs.
    Payload handles keep their arena alive and return its extent when their
    final reference disappears. Releasing space does not migrate existing
    fallback payloads. The owner and its handles are used by one host thread.
    """

    def __init__(
        self, device: Device, nbytes: int, *, fallback: PayloadStorage = MEMORY_STORAGE
    ) -> None:
        self.device, self.fallback = device, fallback
        self._arenas: list[_Arena] = []
        self.reserve(nbytes)

    @property
    def budget(self) -> int:
        """Bytes reserved across all arenas, whether occupied or free."""
        return sum(arena.nbytes for arena in self._arenas)

    @property
    def used(self) -> int:
        """Bytes held by live device payloads, counting shared handles once."""
        return sum(arena.used for arena in self._arenas)

    def reserve(self, nbytes: int) -> None:
        """Add an arena without relocating or invalidating existing payloads."""
        if nbytes:
            self._arenas.append(_Arena(self.device, nbytes))

    def put(self, encoded: np.ndarray) -> SlabPayload:
        """Copy contiguous bytes into an available extent, or into the fallback."""
        source = np.frombuffer(encoded, np.uint8)
        for arena in self._arenas:
            payload = arena.put(source)
            if payload is not None:
                return payload
        return self.fallback.put(encoded)


class _Arena:
    def __init__(self, device: Device, nbytes: int) -> None:
        self.device, self.nbytes, self.used = device, nbytes, 0
        self.free = [(0, nbytes)]
        self.buffer: np.ndarray | jax.Array
        if device.platform == "cpu":
            self.buffer = np.empty(nbytes, np.uint8)
        else:
            from .host_transfer import allocate_device

            self.buffer = allocate_device((nbytes,), dtype=np.dtype(np.uint8), device=device)

    def transfer(self, values: np.ndarray, offset: int, *, upload: bool) -> None:
        if isinstance(self.buffer, np.ndarray):
            target = self.buffer[offset : offset + values.nbytes]
            if upload:
                np.copyto(target, values)
            else:
                np.copyto(values, target)
            return
        import cupy as cp

        with cp.cuda.Device(self.device.local_hardware_id):
            arena = cp.from_dlpack(self.buffer)
            stream = cp.cuda.get_current_stream()
            host, device = values.ctypes.data, arena.data.ptr + offset
            kind = (
                cp.cuda.runtime.memcpyHostToDevice if upload else cp.cuda.runtime.memcpyDeviceToHost
            )
            cp.cuda.runtime.memcpyAsync(
                device if upload else host, host if upload else device,
                values.nbytes, kind, stream.ptr,
            )  # fmt: skip
            stream.synchronize()

    def put(self, values: np.ndarray) -> _DevicePayload | None:
        for index, (offset, length) in enumerate(self.free):
            if values.nbytes > length:
                continue
            self.transfer(values, offset, upload=True)
            if values.nbytes == length:
                self.free.pop(index)
            else:
                self.free[index] = (offset + values.nbytes, length - values.nbytes)
            self.used += values.nbytes
            return _DevicePayload(self, offset, values.nbytes)
        return None

    def release(self, offset: int, nbytes: int) -> None:
        self.used -= nbytes
        self.free.append((offset, nbytes))
        merged: list[tuple[int, int]] = []
        for first, size in sorted(self.free):
            if merged and merged[-1][0] + merged[-1][1] == first:
                start, previous = merged.pop()
                merged.append((start, previous + size))
            else:
                merged.append((first, size))
        self.free = merged


class _DevicePayload:
    def __init__(self, arena: _Arena, offset: int, nbytes: int) -> None:
        self._arena, self._offset, self.nbytes = arena, offset, nbytes
        weakref.finalize(self, arena.release, offset, nbytes)

    @property
    def resident_nbytes(self) -> int:
        return self.nbytes if self._arena.device.platform == "cpu" else 0

    def read(self) -> np.ndarray:
        out = np.empty(self.nbytes, np.uint8)
        self._arena.transfer(out, self._offset, upload=False)
        return out
