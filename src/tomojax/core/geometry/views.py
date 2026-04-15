from __future__ import annotations

import jax.numpy as jnp

from .base import Geometry


def stack_view_poses(
    geometry: Geometry,
    n_views: int,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    return jnp.stack(
        [jnp.asarray(geometry.pose_for_view(i), dtype=dtype) for i in range(int(n_views))],
        axis=0,
    )
