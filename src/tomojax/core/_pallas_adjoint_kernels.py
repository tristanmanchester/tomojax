from __future__ import annotations

from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp

from ._pallas_config import _LAYOUT_VARIANT_IDS
from ._pallas_sampling import _trilinear_atomic_add, _trilinear_load_when_tile_active


def _backproject_kernel(
    T_ref: Any,
    image_ref: Any,
    _init_ref: Any,
    out_ref: Any,
    *,
    nx: int,
    ny: int,
    nz: int,
    nu: int,
    nv: int,
    du: float,
    dv: float,
    det_center_x: float,
    det_center_z: float,
    vol_origin_x: float,
    vol_origin_y: float,
    vol_origin_z: float,
    vx: float,
    vy: float,
    vz: float,
    step_size: float,
    n_steps: int,
    tile_v: int,
    tile_u: int,
    layout_variant_id: int,
    unroll: int | None,
) -> None:
    tile_v_start = pl.program_id(0) * tile_v
    tile_u_start = pl.program_id(1) * tile_u
    if layout_variant_id == _LAYOUT_VARIANT_IDS["detector_uv"]:
        det_u = tile_u_start + jnp.arange(tile_u, dtype=jnp.int32)[:, jnp.newaxis]
        det_v = tile_v_start + jnp.arange(tile_v, dtype=jnp.int32)[jnp.newaxis, :]
    else:
        det_u = tile_u_start + jnp.arange(tile_u, dtype=jnp.int32)[jnp.newaxis, :]
        det_v = tile_v_start + jnp.arange(tile_v, dtype=jnp.int32)[:, jnp.newaxis]
    in_detector = (det_u < nu) & (det_v < nv)

    xr = (det_u.astype(jnp.float32) - jnp.float32(nu / 2.0 - 0.5)) * jnp.float32(du) + jnp.float32(
        det_center_x
    )
    zr = (det_v.astype(jnp.float32) - jnp.float32(nv / 2.0 - 0.5)) * jnp.float32(dv) + jnp.float32(
        det_center_z
    )

    def tload(row: int, col: int):
        return plt.load(T_ref.at[row, col])

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
    ray_vals = plt.load(image_ref.at[det_v, det_u], mask=in_detector, other=0.0) * step_size32

    def body(s, carry):
        ix, iy, iz = carry
        original_step = jnp.int32(n_steps - 1) - s
        active = in_detector & (original_step < n_steps_ray)
        _trilinear_atomic_add(
            out_ref,
            ray_vals,
            ix,
            iy,
            iz,
            nx=nx,
            ny=ny,
            nz=nz,
            active=active,
        )
        return ix - dix, iy - diy, iz - diz

    if unroll is None:
        tile_steps = jnp.minimum(
            jnp.max(jnp.where(in_detector, n_steps_ray, 0)),
            jnp.asarray(n_steps, dtype=jnp.int32),
        )
        tile_last_step = jnp.maximum(tile_steps - jnp.int32(1), jnp.int32(0)).astype(jnp.float32)

        def tile_body(s, carry):
            ix, iy, iz = carry
            original_step = tile_steps - jnp.int32(1) - s
            active = in_detector & (original_step < n_steps_ray)
            _trilinear_atomic_add(
                out_ref,
                ray_vals,
                ix,
                iy,
                iz,
                nx=nx,
                ny=ny,
                nz=nz,
                active=active,
            )
            return ix - dix, iy - diy, iz - diz

        tile_init = (
            ix0 + dix * tile_last_step,
            iy0 + diy * tile_last_step,
            iz0 + diz * tile_last_step,
        )
        jax.lax.fori_loop(0, tile_steps, tile_body, tile_init)
    else:
        last_step = jnp.float32(max(n_steps - 1, 0))
        init = (
            ix0 + dix * last_step,
            iy0 + diy * last_step,
            iz0 + diz * last_step,
        )
        jax.lax.fori_loop(0, n_steps, body, init, unroll=unroll)


def _projector_residual_sse_kernel(
    T_ref: Any,
    volume_ref: Any,
    target_ref: Any,
    out_ref: Any,
    *,
    nx: int,
    ny: int,
    nz: int,
    nu: int,
    nv: int,
    du: float,
    dv: float,
    det_center_x: float,
    det_center_z: float,
    vol_origin_x: float,
    vol_origin_y: float,
    vol_origin_z: float,
    vx: float,
    vy: float,
    vz: float,
    step_size: float,
    n_steps: int,
    tile_v: int,
    tile_u: int,
    kernel_variant_id: int,
    layout_variant_id: int,
    unroll: int | None,
) -> None:
    view_idx = pl.program_id(0)
    tile_v_idx = pl.program_id(1)
    tile_u_idx = pl.program_id(2)
    tile_v_start = tile_v_idx * tile_v
    tile_u_start = tile_u_idx * tile_u
    if layout_variant_id == _LAYOUT_VARIANT_IDS["detector_uv"]:
        det_u = tile_u_start + jnp.arange(tile_u, dtype=jnp.int32)[:, jnp.newaxis]
        det_v = tile_v_start + jnp.arange(tile_v, dtype=jnp.int32)[jnp.newaxis, :]
    else:
        det_u = tile_u_start + jnp.arange(tile_u, dtype=jnp.int32)[jnp.newaxis, :]
        det_v = tile_v_start + jnp.arange(tile_v, dtype=jnp.int32)[:, jnp.newaxis]
    in_detector = (det_u < nu) & (det_v < nv)

    xr = (det_u.astype(jnp.float32) - jnp.float32(nu / 2.0 - 0.5)) * jnp.float32(du) + jnp.float32(
        det_center_x
    )
    zr = (det_v.astype(jnp.float32) - jnp.float32(nv / 2.0 - 0.5)) * jnp.float32(dv) + jnp.float32(
        det_center_z
    )

    def tload(row: int, col: int):
        return plt.load(T_ref.at[view_idx, row, col])

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

    def body(step_idx, carry):
        acc, ix, iy, iz = carry
        active = step_idx < n_steps_ray
        sample = _trilinear_load_when_tile_active(
            volume_ref,
            ix,
            iy,
            iz,
            nx=nx,
            ny=ny,
            nz=nz,
            active=active,
            kernel_variant_id=kernel_variant_id,
        )
        return (
            acc + sample.astype(jnp.float32) * active.astype(jnp.float32) * step_size32,
            ix + dix,
            iy + diy,
            iz + diz,
        )

    init = (
        jnp.zeros_like(ix0, dtype=jnp.float32),
        ix0,
        iy0,
        iz0,
    )
    if unroll is None:
        tile_steps = jnp.minimum(
            jnp.max(jnp.where(in_detector, n_steps_ray, 0)),
            jnp.asarray(n_steps, dtype=jnp.int32),
        )
        acc, _, _, _ = jax.lax.fori_loop(0, tile_steps, body, init)
    else:
        acc, _, _, _ = jax.lax.fori_loop(0, n_steps, body, init, unroll=unroll)
    target = plt.load(target_ref.at[view_idx, det_v, det_u], mask=in_detector, other=0.0)
    residual = jnp.where(in_detector, acc.astype(jnp.float32) - target.astype(jnp.float32), 0.0)
    out_ref[0, 0, 0] = jnp.sum(residual * residual).astype(jnp.float32)


def _projector_loss_grad_kernel(
    T_ref: Any,
    volume_ref: Any,
    target_ref: Any,
    weights_ref: Any,
    _grad_init_ref: Any,
    loss_ref: Any,
    grad_ref: Any,
    *,
    nx: int,
    ny: int,
    nz: int,
    nu: int,
    nv: int,
    du: float,
    dv: float,
    det_center_x: float,
    det_center_z: float,
    vol_origin_x: float,
    vol_origin_y: float,
    vol_origin_z: float,
    vx: float,
    vy: float,
    vz: float,
    step_size: float,
    n_steps: int,
    tile_v: int,
    tile_u: int,
    kernel_variant_id: int,
    layout_variant_id: int,
    unroll: int | None,
    compute_loss: bool,
) -> None:
    view_idx = pl.program_id(0)
    tile_v_idx = pl.program_id(1)
    tile_u_idx = pl.program_id(2)
    tile_v_start = tile_v_idx * tile_v
    tile_u_start = tile_u_idx * tile_u
    if layout_variant_id == _LAYOUT_VARIANT_IDS["detector_uv"]:
        det_u = tile_u_start + jnp.arange(tile_u, dtype=jnp.int32)[:, jnp.newaxis]
        det_v = tile_v_start + jnp.arange(tile_v, dtype=jnp.int32)[jnp.newaxis, :]
    else:
        det_u = tile_u_start + jnp.arange(tile_u, dtype=jnp.int32)[jnp.newaxis, :]
        det_v = tile_v_start + jnp.arange(tile_v, dtype=jnp.int32)[:, jnp.newaxis]
    in_detector = (det_u < nu) & (det_v < nv)

    xr = (det_u.astype(jnp.float32) - jnp.float32(nu / 2.0 - 0.5)) * jnp.float32(du) + jnp.float32(
        det_center_x
    )
    zr = (det_v.astype(jnp.float32) - jnp.float32(nv / 2.0 - 0.5)) * jnp.float32(dv) + jnp.float32(
        det_center_z
    )

    def tload(row: int, col: int):
        return plt.load(T_ref.at[view_idx, row, col])

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

    def fwd_body(step_idx, carry):
        acc, ix, iy, iz = carry
        active = step_idx < n_steps_ray
        sample = _trilinear_load_when_tile_active(
            volume_ref,
            ix,
            iy,
            iz,
            nx=nx,
            ny=ny,
            nz=nz,
            active=active,
            kernel_variant_id=kernel_variant_id,
        )
        return (
            acc + sample.astype(jnp.float32) * active.astype(jnp.float32) * step_size32,
            ix + dix,
            iy + diy,
            iz + diz,
        )

    init = (
        jnp.zeros_like(ix0, dtype=jnp.float32),
        ix0,
        iy0,
        iz0,
    )
    if unroll is None:
        tile_steps = jnp.minimum(
            jnp.max(jnp.where(in_detector, n_steps_ray, 0)),
            jnp.asarray(n_steps, dtype=jnp.int32),
        )
        acc, _, _, _ = jax.lax.fori_loop(0, tile_steps, fwd_body, init)
    else:
        acc, _, _, _ = jax.lax.fori_loop(0, n_steps, fwd_body, init, unroll=unroll)

    target = plt.load(target_ref.at[view_idx, det_v, det_u], mask=in_detector, other=0.0)
    weight = plt.load(weights_ref.at[view_idx, 0, 0])
    raw_residual = jnp.where(in_detector, acc.astype(jnp.float32) - target.astype(jnp.float32), 0.0)
    weighted_residual = raw_residual * weight
    if compute_loss:
        loss_ref[0, 0, 0] = jnp.float32(0.5) * jnp.sum(
            weighted_residual * weighted_residual
        ).astype(jnp.float32)
    else:
        loss_ref[0, 0, 0] = jnp.float32(0.0)
    grad_residual = raw_residual * weight * weight * step_size32

    def bwd_body(s, carry):
        ix, iy, iz = carry
        original_step = jnp.int32(max(n_steps - 1, 0)) - s
        active = in_detector & (original_step < n_steps_ray)
        _trilinear_atomic_add(
            grad_ref,
            grad_residual,
            ix,
            iy,
            iz,
            nx=nx,
            ny=ny,
            nz=nz,
            active=active,
        )
        return ix - dix, iy - diy, iz - diz

    if unroll is None:
        tile_steps = jnp.minimum(
            jnp.max(jnp.where(in_detector, n_steps_ray, 0)),
            jnp.asarray(n_steps, dtype=jnp.int32),
        )
        tile_last_step = jnp.maximum(tile_steps - jnp.int32(1), jnp.int32(0)).astype(jnp.float32)

        def bwd_tile_body(s, carry):
            ix, iy, iz = carry
            original_step = tile_steps - jnp.int32(1) - s
            active = in_detector & (original_step < n_steps_ray)
            _trilinear_atomic_add(
                grad_ref,
                grad_residual,
                ix,
                iy,
                iz,
                nx=nx,
                ny=ny,
                nz=nz,
                active=active,
            )
            return ix - dix, iy - diy, iz - diz

        bwd_tile_init = (
            ix0 + dix * tile_last_step,
            iy0 + diy * tile_last_step,
            iz0 + diz * tile_last_step,
        )
        jax.lax.fori_loop(0, tile_steps, bwd_tile_body, bwd_tile_init)
    else:
        last_step = jnp.float32(max(n_steps - 1, 0))
        bwd_init = (
            ix0 + dix * last_step,
            iy0 + diy * last_step,
            iz0 + diz * last_step,
        )
        jax.lax.fori_loop(0, n_steps, bwd_body, bwd_init, unroll=unroll)
