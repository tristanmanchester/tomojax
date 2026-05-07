"""Minimal reference reconstruction helpers."""
# pyright: reportAny=false, reportUnknownMemberType=false

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from tomojax.core.projector import sum_backproject_views_T
from tomojax.forward import core_projection_geometry_from_state

if TYPE_CHECKING:
    from tomojax.geometry import GeometryState


def reconstruct_average_reference(projections: jax.Array, *, depth: int | None = None) -> jax.Array:
    """Return a tiny deterministic preview volume by average backprojection."""
    proj = jnp.asarray(projections, dtype=jnp.float32)
    if proj.ndim != 3:
        raise ValueError("projections must have shape (views, rows, cols)")
    target_depth = int(depth or proj.shape[1])
    mean_projection = jnp.mean(proj, axis=0)
    volume = jnp.repeat(mean_projection[:, None, :], target_depth, axis=1)
    return volume.astype(jnp.float32)


def reconstruct_backprojection_reference(
    projections: jax.Array,
    geometry: GeometryState,
    *,
    depth: int | None = None,
) -> jax.Array:
    """Return a deterministic geometry-aware core-adjoint backprojection volume."""
    proj = jnp.asarray(projections, dtype=jnp.float32)
    if proj.ndim != 3:
        raise ValueError("projections must have shape (views, rows, cols)")
    target_depth = int(depth or proj.shape[1])
    if geometry.pose.n_views != proj.shape[0]:
        raise ValueError("geometry view count must match projections")
    core = core_projection_geometry_from_state(
        (int(proj.shape[1]), target_depth, int(proj.shape[2])),
        geometry,
        detector_shape=(int(proj.shape[1]), int(proj.shape[2])),
    )
    volume = sum_backproject_views_T(
        core.t_all,
        core.grid,
        core.detector,
        proj,
        step_size=core.step_size,
        n_steps=core.n_steps,
        unroll=core.projector_unroll,
        gather_dtype=core.gather_dtype,
        det_grid=core.det_grid,
    )
    return (volume / jnp.asarray(max(int(proj.shape[0]), 1), dtype=jnp.float32)).astype(jnp.float32)
