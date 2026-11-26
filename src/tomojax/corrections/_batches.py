"""Calling-thread progress for groups of corrections and bounded frame reads."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager, nullcontext, suppress
from typing import TYPE_CHECKING, Any, Protocol, TypeGuard

import numpy as np

from tomojax import progress
from tomojax.core.devices import announce

from ._steps import step_name

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from types import TracebackType

    import jax
    from numpy.typing import ArrayLike

    from tomojax._data import FrameStack

    from ._steps import Step


class _Shaped(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...


def _shaped(value: object) -> TypeGuard[_Shaped]:
    # hasattr, unlike a runtime Protocol check, sees attributes a proxy delegates.
    return hasattr(value, "shape")


class HostCopies(AbstractContextManager["HostCopies"]):
    """Join every submitted download before its producer releases its buffers."""

    def __init__(self) -> None:
        self.pending: dict[int, jax.Array] = {}

    def start(self, batch: jax.Array) -> None:
        self.pending[id(batch)] = batch
        batch.copy_to_host_async()

    def result(self, batch: jax.Array) -> np.ndarray:
        host = np.asarray(batch)
        del self.pending[id(batch)]
        return host

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        for batch in self.pending.values():
            # A failed device operation must not mask a handler's exception.
            with suppress(Exception) if error is not None else nullcontext():
                np.asarray(batch)
        self.pending.clear()


@contextmanager
def correction_group(parts: Sequence[str | Step], batches: range, unit: str) -> Generator[None]:
    """Count one write or statistic per batch in a fused correction group."""
    label = ", ".join(p if isinstance(p, str) else step_name(p) for p in parts)
    with progress.task("correction", label, len(batches), unit):
        announce()
        yield


def view_batches(counts: Any, views: int, batch: int) -> Generator[tuple[int, int, np.ndarray]]:
    """``(start, stop, frames)`` for each batch of views, the next read while one is used."""
    starts = list(range(0, views, batch))

    def read(start: int) -> np.ndarray:
        frames = np.asarray(counts[start : min(start + batch, views)])
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
            yield start, min(start + batch, views), frames


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
