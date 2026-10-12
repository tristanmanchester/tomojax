"""One-reader bounded acquisition batches, preserving native dtype and placement."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from tomojax._typed_arrays import commit_array


def batches(
    counts: Any, views: int, batch: int, *, after_read: Callable[[np.ndarray], None] | None = None
) -> Any:
    """``(start, stop, frames)`` for each batch of views, the next read while one is used."""
    starts = list(range(0, views, batch))
    counts = commit_array(counts)

    def read(start: int) -> np.ndarray:
        frames = np.asarray(counts[start : min(start + batch, views)])
        # Device arrays use native byte order; leave ordinary batches as views.
        frames = frames.astype(frames.dtype.newbyteorder("="), copy=False)
        if after_read is not None:
            after_read(frames)
        return frames

    with ThreadPoolExecutor(max_workers=1) as reader:
        pending = reader.submit(read, starts[0]) if starts else None
        for k, start in enumerate(starts):
            assert pending is not None
            frames = pending.result()
            if k + 1 < len(starts):
                pending = reader.submit(read, starts[k + 1])
            yield start, min(start + batch, views), frames
