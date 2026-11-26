"""Calling-thread progress for groups of corrections and bounded frame reads."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager, nullcontext, suppress
import math
from typing import TYPE_CHECKING, Protocol, TypeGuard

import numpy as np

from tomojax import progress
from tomojax.core.devices import announce

from ._steps import step_name

# Raw views read per request: enough frames for a positional reader to read them at once.
_READ_BYTES = 256 * 2**20
# Host-memory work (copies, calibration sums) runs in bands on this many threads,
# with copies of at least _COPY_BYTES a band.
HOST_THREADS = 8
_COPY_BYTES = 4 * 2**20

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from types import TracebackType

    import jax
    from numpy.typing import ArrayLike

    from tomojax._data import FrameStack

    from ._fields import CalibrationFrames
    from ._steps import Step


class _Shaped(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...


def _shaped(value: object) -> TypeGuard[_Shaped]:
    # hasattr, unlike a runtime Protocol check, sees attributes a proxy delegates.
    return hasattr(value, "shape")


class HostCopies(AbstractContextManager["HostCopies"]):
    """Join every submitted download before its producer releases its buffers.

    Downloads land in their host targets in row bands on several threads: filling
    a new host stack is bound by first-touch page faults, which one core takes in turn.
    """

    def __init__(self) -> None:
        self.pending: dict[int, jax.Array] = {}
        self.pool = ThreadPoolExecutor(HOST_THREADS, thread_name_prefix="tomojax-copy")

    def start(self, batch: jax.Array) -> None:
        self.pending[id(batch)] = batch
        batch.copy_to_host_async()

    def into(self, target: np.ndarray, batch: jax.Array) -> None:
        """Copy the submitted ``batch`` into ``target``, a host array of its shape."""
        host: np.ndarray = np.asarray(batch)
        del self.pending[id(batch)]
        axis = int(target.shape[1] > target.shape[0])
        parts = min(HOST_THREADS, host.nbytes // _COPY_BYTES)

        def copy(band: slice) -> None:
            key = (slice(None),) * axis + (band,)
            target[key] = host[key]

        for done in [self.pool.submit(copy, b) for b in bands(target.shape[axis], parts)]:
            done.result()

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.pool.shutdown()
        for batch in self.pending.values():
            # A failed device operation must not mask a handler's exception.
            with suppress(Exception) if error is not None else nullcontext():
                _ = np.asarray(batch)
        self.pending.clear()


def bands(extent: int, parts: int) -> list[slice]:
    """Up to ``parts`` nonempty, nearly equal slices that together cover ``range(extent)``."""
    parts = max(1, min(parts, extent))
    return [slice(extent * i // parts, extent * (i + 1) // parts) for i in range(parts)]


@contextmanager
def correction_group(parts: Sequence[str | Step], batches: range, unit: str) -> Generator[None]:
    """Count one write or statistic per batch in a fused correction group."""
    label = ", ".join(p if isinstance(p, str) else step_name(p) for p in parts)
    with progress.task("correction", label, len(batches), unit):
        announce()
        yield


def view_batches(
    counts: CalibrationFrames, views: int, batch: int, indices: np.ndarray | None = None
) -> Generator[tuple[int, int, np.ndarray]]:
    """``(start, stop, frames)`` for each batch of views, the next read while one is used.

    Views are read in blocks of whole batches of at least ``_READ_BYTES``, so a
    source that reads many frames at once (positional HDF5 reads) has them to read.
    With ``indices``, batch ``start:stop`` holds frames ``indices[start:stop]``.
    """
    view_bytes = math.prod(counts.shape[1:]) * np.dtype(counts.dtype).itemsize
    block = max(1, _READ_BYTES // max(1, view_bytes * batch)) * batch
    starts = list(range(0, views, block))

    def read(start: int) -> np.ndarray:
        which = slice(start, min(start + block, views))
        frames = np.asarray(counts[which if indices is None else indices[which]])
        # Device arrays use native byte order; leave ordinary batches as views.
        return frames.astype(frames.dtype.newbyteorder("="), copy=False)

    with ThreadPoolExecutor(max_workers=1) as reader:
        pending = reader.submit(read, starts[0]) if starts else None
        for k, start in enumerate(starts):
            progress.check_canceled()
            assert pending is not None
            frames = pending.result()
            if k + 1 < len(starts):
                pending = reader.submit(read, starts[k + 1])
            for first in range(0, len(frames), batch):
                yield (
                    start + first,
                    start + min(first + batch, len(frames)),
                    frames[first : first + batch],
                )


def copy_projections(
    projections: ArrayLike | FrameStack | jax.Array, steps: Sequence[Step]
) -> np.ndarray:
    """Own the line-integral input before applying corrections in place."""
    from tomojax.core.host_volume import fill, host_array

    if not _shaped(projections) or any(step.whole_rows or step.selects_views for step in steps):
        return np.array(projections, np.float32, copy=True)
    out = host_array(projections.shape)
    depth = max(1, (32 << 20) // (4 * out.shape[1] * out.shape[2]))
    return fill(out, projections, depth=depth)
