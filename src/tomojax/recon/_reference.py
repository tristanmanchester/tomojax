"""Minimal reference reconstruction helpers."""
# pyright: reportAny=false, reportUnknownMemberType=false

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

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
    """Return a deterministic geometry-aware reference backprojection volume."""
    proj = jnp.asarray(projections, dtype=jnp.float32)
    if proj.ndim != 3:
        raise ValueError("projections must have shape (views, rows, cols)")
    target_depth = int(depth or proj.shape[1])
    theta = jnp.asarray(geometry.theta_total_rad())
    dx = jnp.asarray(geometry.setup.det_u_px.value + geometry.pose.dx_px)
    dz_setup = geometry.setup.det_v_px.value if geometry.setup.det_v_px.active else 0.0
    dz = jnp.asarray(dz_setup + geometry.pose.dz_px)
    if theta.shape[0] != proj.shape[0]:
        raise ValueError("geometry view count must match projections")
    backprojected = [
        _backproject_one_view(
            proj[view],
            theta_rad=theta[view],
            dx_px=dx[view],
            dz_px=dz[view],
            depth=target_depth,
        )
        for view in range(int(proj.shape[0]))
    ]
    return jnp.mean(jnp.stack(backprojected, axis=0), axis=0).astype(jnp.float32)


def _backproject_one_view(
    projection: jax.Array,
    *,
    theta_rad: jax.Array,
    dx_px: jax.Array,
    dz_px: jax.Array,
    depth: int,
) -> jax.Array:
    unshifted = _shift_periodic_linear(projection, dx_px=-dx_px, dz_px=-dz_px)
    slab = jnp.repeat(unshifted[:, None, :], int(depth), axis=1) / jnp.asarray(
        max(int(depth), 1),
        dtype=jnp.float32,
    )
    return _rotate_xy_linear(slab, -theta_rad)


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
