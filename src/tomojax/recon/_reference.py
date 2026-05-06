"""Minimal reference reconstruction helpers."""
# pyright: reportAny=false, reportUnknownMemberType=false

from __future__ import annotations

import jax
import jax.numpy as jnp


def reconstruct_average_reference(projections: jax.Array, *, depth: int | None = None) -> jax.Array:
    """Return a tiny deterministic preview volume by average backprojection."""
    proj = jnp.asarray(projections, dtype=jnp.float32)
    if proj.ndim != 3:
        raise ValueError("projections must have shape (views, rows, cols)")
    target_depth = int(depth or proj.shape[1])
    mean_projection = jnp.mean(proj, axis=0)
    volume = jnp.repeat(mean_projection[:, None, :], target_depth, axis=1)
    return volume.astype(jnp.float32)
