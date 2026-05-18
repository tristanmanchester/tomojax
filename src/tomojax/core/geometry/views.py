"""Helpers for materialising per-view pose stacks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from .base import Geometry


def stack_view_poses(
    geometry: Geometry,
    n_views: int,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Stack world-from-object poses for the first ``n_views`` views."""
    from .parallel import ParallelGeometry

    if isinstance(geometry, ParallelGeometry):
        thetas = np.asarray(geometry.thetas_deg[: int(n_views)], dtype=np.float32)
        phi = np.deg2rad(thetas).astype(np.float32)
        c = np.cos(phi).astype(np.float32)
        s = np.sin(phi).astype(np.float32)
        poses = np.zeros((int(n_views), 4, 4), dtype=np.float32)
        poses[:, 0, 0] = c
        poses[:, 0, 1] = -s
        poses[:, 1, 0] = s
        poses[:, 1, 1] = c
        poses[:, 2, 2] = 1.0
        poses[:, 3, 3] = 1.0
        return jnp.asarray(poses, dtype=dtype)

    return jnp.stack(
        [jnp.asarray(geometry.pose_for_view(i), dtype=dtype) for i in range(int(n_views))],
        axis=0,
    )
