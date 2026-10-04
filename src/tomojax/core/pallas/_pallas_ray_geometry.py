"""Shared ray-box traversal for Pallas projection, adjoint, and loss kernels."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp


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

    def slab(base: jnp.ndarray, denom: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray):
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

    lo_x, hi_x = slab(base_x, ey_x, lower_x, upper_x)
    lo_y, hi_y = slab(base_y, ey_y, lower_y, upper_y)
    lo_z, hi_z = slab(base_z, ey_z, lower_z, upper_z)
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
