"""Stream view batches from host arrays into compiled solver loops.

A host array is registered under an integer key for the duration of a solve.
Compiled code receives only the key, as an ordinary traced integer, and reads
one batch at a time through a pure callback. The compiled program therefore
depends on shapes alone, and the full projection stack never occupies the
device.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
import itertools

import jax
import jax.numpy as jnp
import numpy as np

_SOURCES: dict[int, np.ndarray] = {}
_KEYS = itertools.count(1)


@contextmanager
def host_source(array: np.ndarray) -> Iterator[jax.Array]:
    """Register ``array`` and yield its key; reads are valid inside the block only."""
    key = next(_KEYS)
    _SOURCES[key] = array
    try:
        yield jnp.int32(key)
    finally:
        _SOURCES.pop(key, None)


def _read(key: np.ndarray, start: np.ndarray, *, size: int) -> np.ndarray:
    source, first = _SOURCES[int(key)], int(start)
    return np.ascontiguousarray(source[first : first + size], dtype=np.float32)


def read_views(key: jax.Array, start: jax.Array, shape: tuple[int, int, int]) -> jax.Array:
    """Return views ``start : start + shape[0]`` of the registered host array."""
    out = jax.ShapeDtypeStruct(shape, jnp.float32)
    return jax.pure_callback(partial(_read, size=shape[0]), out, key, start)


def should_stream(projections: object, *, fraction: float = 0.4) -> bool:
    """Stream host arrays that would take more than ``fraction`` of free device memory."""
    if isinstance(projections, jax.Array) or jax.default_backend() != "gpu":
        return False
    from tomojax.backends import device_free_memory_bytes

    free = device_free_memory_bytes()
    nbytes = int(np.prod(np.shape(projections))) * 4
    return free is not None and nbytes > fraction * free
