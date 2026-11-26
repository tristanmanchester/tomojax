"""Per-call reuse of exact-sized host transfer buffers with shared-view ownership."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Buffer, Callable
import math
from threading import RLock
from typing import TYPE_CHECKING
from weakref import ReferenceType, finalize, ref

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


class PinnedPool:
    """Recycle storage only after its last NumPy or buffer-protocol view dies."""

    def __init__(self, allocate: Callable[[int], Buffer]) -> None:
        self.allocate = allocate
        self.free: dict[int, list[Buffer]] = defaultdict(list)
        self.lock = RLock()

    def empty(self, shape: tuple[int, ...], dtype: DTypeLike = np.float32) -> np.ndarray:
        """Allocate storage without rounding its byte count upward."""
        dtype = np.dtype(dtype)
        size = math.prod(shape) * dtype.itemsize
        with self.lock:
            free = self.free[size]
            storage = free.pop() if free else None
        # Allocation can trigger GC/finalizers. Do not hold the lock around it.
        if storage is None:
            storage = self.allocate(size)
        root = np.ndarray((size // dtype.itemsize,), dtype=dtype, buffer=storage)
        # Downstream reshapes, slices and memoryviews retain this ultimate
        # ndarray owner. A finalizer on the returned reshape would be too early.
        finalize(root, PinnedPool._release, ref(self), size, storage).atexit = False
        return root.reshape(shape)

    @staticmethod
    def _release(pool_ref: ReferenceType[PinnedPool], size: int, storage: Buffer) -> None:
        pool = pool_ref()
        # An escaped output owns its buffer, not the plan's other free buffers.
        if pool is not None:
            with pool.lock:
                pool.free[size].append(storage)
