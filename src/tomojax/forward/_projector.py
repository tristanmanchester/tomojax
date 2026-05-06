"""Minimal JAX reference projector."""
# pyright: reportAny=false, reportUnknownMemberType=false

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from tomojax.geometry import GeometryState


def project_parallel_reference(volume: jax.Array, geometry: GeometryState) -> jax.Array:
    """Project a cubic volume with a minimal differentiable parallel-beam model."""
    vol = jnp.asarray(volume, dtype=jnp.float32)
    if vol.ndim != 3:
        raise ValueError("volume must be 3D")

    theta = geometry.setup.theta_offset_rad.value + geometry.pose.phi_residual_rad
    views = [
        _project_one_view(
            vol,
            theta_rad=float(theta_i),
            dx_px=float(geometry.pose.dx_px[idx]),
            dz_px=float(geometry.pose.dz_px[idx]),
        )
        for idx, theta_i in enumerate(theta)
    ]
    return jnp.stack(views, axis=0)


def _project_one_view(
    volume: jax.Array,
    *,
    theta_rad: float,
    dx_px: float,
    dz_px: float,
) -> jax.Array:
    quadrant = int(jnp.floor((theta_rad % jnp.pi) / (jnp.pi / 4.0))) % 4
    rotated = jnp.rot90(volume, k=quadrant, axes=(0, 1))
    projection = jnp.sum(rotated, axis=1)
    projection = jnp.roll(projection, shift=round(dx_px), axis=1)
    projection = jnp.roll(projection, shift=round(dz_px), axis=0)
    return projection.astype(jnp.float32)
