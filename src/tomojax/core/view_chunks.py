"""Fixed-size view chunks for scanning a projection stack inside a JAX loop.

Every chunk has ``chunk_size`` views so the loop body compiles once; the last
chunk is shifted back to end at the final view, and ``valid_mask`` zeroes the
views it repeats from the chunk before.
"""

from __future__ import annotations

import jax.numpy as jnp


def chunk_size(n_views: int, views_per_batch: int | None, *, default: int) -> int:
    """Views per chunk: ``views_per_batch`` when positive, else ``default``, at most ``n_views``."""
    requested = (
        int(views_per_batch)
        if views_per_batch is not None and int(views_per_batch) > 0
        else default
    )
    return max(1, min(int(requested), int(n_views)))


def chunk_schedule(
    i: jnp.ndarray,
    *,
    n_views: int,
    chunk_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """First view, float32 validity mask and view indices of chunk ``i``.

    Chunk ``i`` nominally starts at ``i * chunk_size``; a chunk that would run
    past ``n_views`` starts earlier so that it ends at the last view, and the
    mask is zero for the views it shares with the previous chunk.
    """
    i = jnp.asarray(i, dtype=jnp.int32)
    b = jnp.int32(chunk_size)
    start = i * b
    remaining = jnp.maximum(0, jnp.int32(n_views) - start)
    valid = jnp.minimum(b, remaining)
    shift = b - valid
    start_shifted = jnp.maximum(0, start - shift)
    idx = jnp.arange(chunk_size, dtype=jnp.int32)
    valid_mask = (idx >= (b - valid)).astype(jnp.float32)
    return start_shifted, valid_mask, start_shifted + idx


__all__ = ["chunk_schedule", "chunk_size"]
