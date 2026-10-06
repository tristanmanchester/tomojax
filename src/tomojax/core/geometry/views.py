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
    """Stack world-from-object poses for the first ``n_views`` views.

    A geometry class may define ``stack_poses(n_views, dtype)`` to build the
    stack in one vectorised step (looked up on the class, so wrappers that
    forward attributes to a base geometry do not inherit it).
    """
    stack = getattr(type(geometry), "stack_poses", None)
    if callable(stack):
        return stack(geometry, int(n_views), dtype)

    from .cone import ConeGeometry
    from .lamino import LaminographyGeometry
    from .parallel import ParallelGeometry

    if type(geometry) is ConeGeometry:
        thetas = np.asarray(geometry.thetas_deg[: int(n_views)], dtype=np.float64)
        return jnp.asarray(geometry.poses(thetas).astype(dtype))

    # Subclasses may override pose_for_view (for example to add calibrated
    # shifts). Only specialize the exact built-in implementation.
    if type(geometry) is ParallelGeometry:
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
        # Convert on the host: a device-side cast would compile a separate program.
        return jnp.asarray(poses.astype(dtype))

    if type(geometry) is LaminographyGeometry:
        from .transforms import align_u_to_v

        angles = np.deg2rad(np.asarray(geometry.thetas_deg[: int(n_views)], dtype=np.float64))
        rotation = np.zeros((int(n_views), 3, 3), dtype=np.float64)
        rotation[:, 0, 0] = rotation[:, 1, 1] = np.cos(angles)
        rotation[:, 1, 0] = np.sin(angles)
        rotation[:, 0, 1] = -np.sin(angles)
        rotation[:, 2, 2] = 1.0
        alignment = align_u_to_v(np.array([0.0, 0.0, 1.0]), geometry._axis_unit())
        poses = np.zeros((int(n_views), 4, 4), dtype=np.float64)
        poses[:, :3, :3] = alignment @ rotation
        poses[:, 3, 3] = 1.0
        return jnp.asarray(poses.astype(dtype))

    return jnp.stack(
        [jnp.asarray(geometry.pose_for_view(i), dtype=dtype) for i in range(int(n_views))],
        axis=0,
    )
