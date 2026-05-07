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

    theta = jnp.asarray(geometry.theta_total_rad())
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
    rotated = _rotate_xy_linear(volume, theta_rad)
    projection = jnp.sum(rotated, axis=1)
    projection = _shift_periodic_linear(projection, dx_px=dx_px, dz_px=dz_px)
    return projection.astype(jnp.float32)


def _rotate_xy_linear(volume: jax.Array, theta_rad: jax.Array) -> jax.Array:
    x_coords = jnp.arange(volume.shape[0], dtype=jnp.float32)
    y_coords = jnp.arange(volume.shape[1], dtype=jnp.float32)
    x_grid, y_grid = jnp.meshgrid(x_coords, y_coords, indexing="ij")
    x_center = (jnp.asarray(volume.shape[0], dtype=jnp.float32) - 1.0) / 2.0
    y_center = (jnp.asarray(volume.shape[1], dtype=jnp.float32) - 1.0) / 2.0
    x_rel = x_grid - x_center
    y_rel = y_grid - y_center
    cos_t = jnp.cos(theta_rad)
    sin_t = jnp.sin(theta_rad)
    source_x = cos_t * x_rel + sin_t * y_rel + x_center
    source_y = -sin_t * x_rel + cos_t * y_rel + y_center

    def sample_slice(slice_xy: jax.Array) -> jax.Array:
        return _sample_image_linear_zero(slice_xy, source_x=source_x, source_y=source_y)

    return jax.vmap(sample_slice, in_axes=2, out_axes=2)(volume)


def _sample_image_linear_zero(
    image: jax.Array,
    *,
    source_x: jax.Array,
    source_y: jax.Array,
) -> jax.Array:
    x0 = jnp.floor(source_x).astype(jnp.int32)
    y0 = jnp.floor(source_y).astype(jnp.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    x_frac = source_x - jnp.floor(source_x)
    y_frac = source_y - jnp.floor(source_y)
    top = (1.0 - y_frac) * _take2d_zero(image, x0, y0) + y_frac * _take2d_zero(image, x0, y1)
    bottom = (1.0 - y_frac) * _take2d_zero(image, x1, y0) + y_frac * _take2d_zero(
        image,
        x1,
        y1,
    )
    return (1.0 - x_frac) * top + x_frac * bottom


def _take2d_zero(image: jax.Array, x_index: jax.Array, y_index: jax.Array) -> jax.Array:
    valid = (
        (x_index >= 0) & (x_index < image.shape[0]) & (y_index >= 0) & (y_index < image.shape[1])
    )
    clipped_x = jnp.clip(x_index, 0, image.shape[0] - 1)
    clipped_y = jnp.clip(y_index, 0, image.shape[1] - 1)
    values = image[clipped_x, clipped_y]
    return jnp.where(valid, values, 0.0)


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
