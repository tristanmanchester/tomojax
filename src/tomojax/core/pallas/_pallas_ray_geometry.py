"""Shared ray-box traversal for Pallas projection, adjoint, and loss kernels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp

from ._pallas_variants import _LAYOUT_VARIANT_IDS


def ray_traversal(
    tload: Callable[[int, int], jnp.ndarray],
    xr: jnp.ndarray,
    zr: jnp.ndarray,
    *,
    shape: tuple[int, int, int],
    origin: tuple[float, float, float],
    voxel_size: tuple[float, float, float],
    step_size: float,
) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Return the same voxel-coordinate recurrence for every inline kernel.

    The pose is world-from-object, with parallel rays along world +y. Its
    transpose rotation maps detector points and ray directions into the object.
    Intersect the full zero-padded trilinear support, including its boundary halo.
    """
    nx, ny, nz = shape
    vol_origin_x, vol_origin_y, vol_origin_z = origin
    vx, vy, vz = voxel_size
    t00, t01, t02, t03 = tload(0, 0), tload(0, 1), tload(0, 2), tload(0, 3)
    t10, t11, t12, t13 = tload(1, 0), tload(1, 1), tload(1, 2), tload(1, 3)
    t20, t21, t22, t23 = tload(2, 0), tload(2, 1), tload(2, 2), tload(2, 3)
    ey_x = t10
    ey_y = t11
    ey_z = t12
    tinv_x = -(t00 * t03 + t10 * t13 + t20 * t23)
    tinv_y = -(t01 * t03 + t11 * t13 + t21 * t23)
    tinv_z = -(t02 * t03 + t12 * t13 + t22 * t23)
    base_x = t00 * xr + t20 * zr + tinv_x
    base_y = t01 * xr + t21 * zr + tinv_y
    base_z = t02 * xr + t22 * zr + tinv_z

    lower_x = jnp.float32(vol_origin_x - vx)
    lower_y = jnp.float32(vol_origin_y - vy)
    lower_z = jnp.float32(vol_origin_z - vz)
    upper_x = jnp.float32(vol_origin_x + nx * vx)
    upper_y = jnp.float32(vol_origin_y + ny * vy)
    upper_z = jnp.float32(vol_origin_z + nz * vz)

    lo_x, hi_x = slab_intersection(base_x, ey_x, lower_x, upper_x)
    lo_y, hi_y = slab_intersection(base_y, ey_y, lower_y, upper_y)
    lo_z, hi_z = slab_intersection(base_z, ey_z, lower_z, upper_z)
    y_entry = jnp.maximum(jnp.maximum(lo_x, lo_y), lo_z)
    y_exit = jnp.minimum(jnp.minimum(hi_x, hi_y), hi_z)
    path_length = jnp.maximum(jnp.float32(0.0), y_exit - y_entry)
    valid_rays = path_length > jnp.float32(0.0)
    y_start = jnp.where(valid_rays, y_entry, jnp.float32(0.0))
    step_size32 = jnp.float32(step_size)
    n_steps_ray = jnp.where(
        valid_rays,
        jnp.ceil(path_length / step_size32).astype(jnp.int32),
        jnp.int32(0),
    )

    q0_x = base_x + y_start * ey_x
    q0_y = base_y + y_start * ey_y
    q0_z = base_z + y_start * ey_z
    ix0 = (q0_x - jnp.float32(vol_origin_x)) / jnp.float32(vx)
    iy0 = (q0_y - jnp.float32(vol_origin_y)) / jnp.float32(vy)
    iz0 = (q0_z - jnp.float32(vol_origin_z)) / jnp.float32(vz)
    dix = step_size32 * ey_x / jnp.float32(vx)
    diy = step_size32 * ey_y / jnp.float32(vy)
    diz = step_size32 * ey_z / jnp.float32(vz)

    return ix0, iy0, iz0, dix, diy, diz, n_steps_ray


def slab_intersection(
    base: jnp.ndarray, denom: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Intersect a ray with a slab, including parallel rays on its boundary."""
    eps = jnp.float32(1e-8)
    parallel = jnp.abs(denom) < eps
    safe_denom = jnp.where(parallel, jnp.float32(1.0), denom)
    t1 = (lower - base) / safe_denom
    t2 = (upper - base) / safe_denom
    lo = jnp.minimum(t1, t2)
    hi = jnp.maximum(t1, t2)
    inside = (base >= lower) & (base <= upper)
    inf = jnp.asarray(jnp.inf, dtype=jnp.float32)
    neg_inf = jnp.asarray(float("-inf"), dtype=jnp.float32)
    lo = jnp.where(parallel, jnp.where(inside, neg_inf, inf), lo)
    hi = jnp.where(parallel, jnp.where(inside, inf, neg_inf), hi)
    return lo, hi


def detector_tile(
    tile_shape: tuple[int, int],
    *,
    stacked: bool,
    layout_variant_id: int = _LAYOUT_VARIANT_IDS["detector_vu"],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return broadcastable detector indices in the kernel's lane layout."""
    tile_v, tile_u = tile_shape
    tile_v_start = pl.program_id(1 if stacked else 0) * tile_v
    tile_u_start = pl.program_id(2 if stacked else 1) * tile_u
    if layout_variant_id == _LAYOUT_VARIANT_IDS["detector_uv"]:
        det_u = tile_u_start + jnp.arange(tile_u, dtype=jnp.int32)[:, jnp.newaxis]
        det_v = tile_v_start + jnp.arange(tile_v, dtype=jnp.int32)[jnp.newaxis, :]
    else:
        det_u = tile_u_start + jnp.arange(tile_u, dtype=jnp.int32)[jnp.newaxis, :]
        det_v = tile_v_start + jnp.arange(tile_v, dtype=jnp.int32)[:, jnp.newaxis]
    return det_u, det_v


def detector_ray_traversal(
    T_ref: Any,
    det_u: jnp.ndarray,
    det_v: jnp.ndarray,
    *,
    detector_shape: tuple[int, int],
    detector_spacing: tuple[float, float],
    detector_center: tuple[float, float],
    shape: tuple[int, int, int],
    origin: tuple[float, float, float],
    voxel_size: tuple[float, float, float],
    step_size: float,
    stacked: bool,
) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Map detector pixel centers and a single/stacked pose to ray traversal."""
    nv, nu = detector_shape
    du, dv = detector_spacing
    det_center_x, det_center_z = detector_center
    xr = (det_u.astype(jnp.float32) - jnp.float32(nu / 2.0 - 0.5)) * jnp.float32(du) + jnp.float32(
        det_center_x
    )
    zr = (det_v.astype(jnp.float32) - jnp.float32(nv / 2.0 - 0.5)) * jnp.float32(dv) + jnp.float32(
        det_center_z
    )

    def tload(row: int, col: int):
        if stacked:
            return plt.load(T_ref.at[pl.program_id(0), jnp.int32(row), jnp.int32(col)])
        return plt.load(T_ref.at[jnp.int32(row), jnp.int32(col)])

    return ray_traversal(
        tload, xr, zr, shape=shape, origin=origin, voxel_size=voxel_size, step_size=step_size
    )
