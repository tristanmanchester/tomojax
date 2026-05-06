"""Minimal JAX reference projector."""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

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

    theta = jnp.asarray(geometry.setup.theta_offset_rad.value + geometry.pose.phi_residual_rad)
    dx = jnp.asarray(geometry.setup.det_u_px.value + geometry.pose.dx_px)
    dz_setup = geometry.setup.det_v_px.value if geometry.setup.det_v_px.active else 0.0
    dz = jnp.asarray(dz_setup + geometry.pose.dz_px)
    return project_parallel_reference_arrays(vol, theta_rad=theta, dx_px=dx, dz_px=dz)


def project_parallel_reference_arrays(
    volume: jax.Array,
    *,
    theta_rad: jax.Array,
    dx_px: jax.Array,
    dz_px: jax.Array,
) -> jax.Array:
    """Project a volume from JAX pose arrays with differentiable detector shifts."""
    vol = jnp.asarray(volume, dtype=jnp.float32)
    theta = jnp.asarray(theta_rad, dtype=jnp.float32)
    dx = jnp.asarray(dx_px, dtype=jnp.float32)
    dz = jnp.asarray(dz_px, dtype=jnp.float32)
    if theta.shape != dx.shape or theta.shape != dz.shape:
        raise ValueError("theta_rad, dx_px, and dz_px must have matching shapes")
    return jax.vmap(_project_one_view, in_axes=(None, 0, 0, 0))(vol, theta, dx, dz)


def _project_one_view(
    volume: jax.Array,
    theta_rad: jax.Array,
    dx_px: jax.Array,
    dz_px: jax.Array,
) -> jax.Array:
    quadrant = jnp.asarray(jnp.floor((theta_rad % jnp.pi) / (jnp.pi / 4.0)), dtype=jnp.int32) % 4
    rotated = jax.lax.switch(
        quadrant,
        (
            lambda x: x,
            lambda x: jnp.rot90(x, k=1, axes=(0, 1)),
            lambda x: jnp.rot90(x, k=2, axes=(0, 1)),
            lambda x: jnp.rot90(x, k=3, axes=(0, 1)),
        ),
        volume,
    )
    projection = jnp.sum(rotated, axis=1)
    projection = _shift_periodic_linear(projection, dx_px=dx_px, dz_px=dz_px)
    return projection.astype(jnp.float32)


def _shift_periodic_linear(image: jax.Array, *, dx_px: jax.Array, dz_px: jax.Array) -> jax.Array:
    rows = jnp.arange(image.shape[0], dtype=jnp.float32) - dz_px
    cols = jnp.arange(image.shape[1], dtype=jnp.float32) - dx_px
    row0 = jnp.floor(rows).astype(jnp.int32)
    col0 = jnp.floor(cols).astype(jnp.int32)
    row_frac = rows - jnp.floor(rows)
    col_frac = cols - jnp.floor(cols)

    col1 = col0 + 1
    row1 = row0 + 1
    horizontally_shifted = (1.0 - col_frac) * jnp.take(
        image, col0, axis=1, mode="wrap"
    ) + col_frac * jnp.take(image, col1, axis=1, mode="wrap")
    top_source = jnp.take(horizontally_shifted, row0, axis=0, mode="wrap")
    bottom_values = jnp.take(horizontally_shifted, row1, axis=0, mode="wrap")
    return (1.0 - row_frac)[:, None] * top_source + row_frac[:, None] * bottom_values
