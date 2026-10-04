from __future__ import annotations

from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp

from ._pallas_config import _LAYOUT_VARIANT_IDS
from ._pallas_loop import paired_fori_loop, static_fori_loop
from ._pallas_ray_geometry import ray_traversal
from ._pallas_sampling import _trilinear_load_active, _trilinear_load_z_integer


def _projector_kernel(
    T_ref: Any,
    volume_ref: Any,
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
        return plt.load(
            T_ref.at[jnp.asarray(row, dtype=jnp.int32), jnp.asarray(col, dtype=jnp.int32)]
        )

    # Rinv = T[:3, :3].T. The existing projector evaluates world y as the ray
    # parameter, so ``base`` is object_from_world([x, 0, z]).
    ix0, iy0, iz0, dix, diy, diz, n_steps_ray = ray_traversal(
        tload,
        xr,
        zr,
        shape=(nx, ny, nz),
        origin=(vol_origin_x, vol_origin_y, vol_origin_z),
        voxel_size=(vx, vy, vz),
        step_size=step_size,
    )
    step_size32 = jnp.float32(step_size)

    def body(step_idx, carry):
        acc, ix, iy, iz = carry
        active = in_detector & (step_idx < n_steps_ray)
        sample = _trilinear_load_active(
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
        acc, _, _, _ = paired_fori_loop(tile_steps, body, init)
    else:
        acc, _, _, _ = static_fori_loop(n_steps, body, init, unroll=unroll)
    if layout_variant_id == _LAYOUT_VARIANT_IDS["detector_uv"]:
        plt.store(out_ref, acc.T.astype(jnp.float32), mask=in_detector.T)
    else:
        plt.store(out_ref, acc.astype(jnp.float32), mask=in_detector)


def _projector_views_kernel(
    T_ref: Any,
    volume_ref: Any,
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
    tile_v_start = pl.program_id(1) * tile_v
    tile_u_start = pl.program_id(2) * tile_u
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
        return plt.load(
            T_ref.at[
                jnp.asarray(view_idx, dtype=jnp.int32),
                jnp.asarray(row, dtype=jnp.int32),
                jnp.asarray(col, dtype=jnp.int32),
            ]
        )

    ix0, iy0, iz0, dix, diy, diz, n_steps_ray = ray_traversal(
        tload,
        xr,
        zr,
        shape=(nx, ny, nz),
        origin=(vol_origin_x, vol_origin_y, vol_origin_z),
        voxel_size=(vx, vy, vz),
        step_size=step_size,
    )
    step_size32 = jnp.float32(step_size)

    def body(step_idx, carry):
        acc, ix, iy, iz = carry
        active = in_detector & (step_idx < n_steps_ray)
        sample = _trilinear_load_active(
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
        acc, _, _, _ = paired_fori_loop(tile_steps, body, init)
    else:
        acc, _, _, _ = static_fori_loop(n_steps, body, init, unroll=unroll)
    if layout_variant_id == _LAYOUT_VARIANT_IDS["detector_uv"]:
        plt.store(
            out_ref,
            acc.T[jnp.newaxis, :, :].astype(jnp.float32),
            mask=in_detector.T[jnp.newaxis, :, :],
        )
    else:
        plt.store(
            out_ref, acc[jnp.newaxis, :, :].astype(jnp.float32), mask=in_detector[jnp.newaxis, :, :]
        )


def _projector_parallel_z_views_kernel(
    cos_ref: Any,
    sin_ref: Any,
    volume_ref: Any,
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
    unroll: int | None,
) -> None:
    view_idx = pl.program_id(0)
    tile_v_start = pl.program_id(1) * tile_v
    tile_u_start = pl.program_id(2) * tile_u
    det_u = tile_u_start + jnp.arange(tile_u, dtype=jnp.int32)[jnp.newaxis, :]
    det_v = tile_v_start + jnp.arange(tile_v, dtype=jnp.int32)[:, jnp.newaxis]
    in_detector = (det_u < nu) & (det_v < nv)

    xr = (det_u.astype(jnp.float32) - jnp.float32(nu / 2.0 - 0.5)) * jnp.float32(du) + jnp.float32(
        det_center_x
    )
    zr = (det_v.astype(jnp.float32) - jnp.float32(nv / 2.0 - 0.5)) * jnp.float32(dv) + jnp.float32(
        det_center_z
    )
    c = plt.load(cos_ref.at[view_idx])
    s = plt.load(sin_ref.at[view_idx])

    base_x = c * xr
    base_y = -s * xr
    base_z = zr
    ey_x = s
    ey_y = c

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
    z_inside = (base_z >= lower_z) & (base_z <= upper_z)
    y_entry = jnp.maximum(lo_x, lo_y)
    y_exit = jnp.minimum(hi_x, hi_y)
    path_length = jnp.maximum(jnp.float32(0.0), y_exit - y_entry)
    valid_rays = z_inside & (path_length > jnp.float32(0.0))
    y_start = jnp.where(valid_rays, y_entry, jnp.float32(0.0))
    step_size32 = jnp.float32(step_size)
    n_steps_ray = jnp.where(
        valid_rays,
        jnp.ceil(path_length / step_size32).astype(jnp.int32),
        jnp.int32(0),
    )

    q0_x = base_x + y_start * ey_x
    q0_y = base_y + y_start * ey_y
    ix0 = (q0_x - jnp.float32(vol_origin_x)) / jnp.float32(vx)
    iy0 = (q0_y - jnp.float32(vol_origin_y)) / jnp.float32(vy)
    iz0 = (base_z - jnp.float32(vol_origin_z)) / jnp.float32(vz)
    dix = step_size32 * ey_x / jnp.float32(vx)
    diy = step_size32 * ey_y / jnp.float32(vy)

    def body(step_idx, carry):
        acc, ix, iy = carry
        active = in_detector & (step_idx < n_steps_ray)
        sample = _trilinear_load_z_integer(
            volume_ref,
            ix,
            iy,
            iz0,
            nx=nx,
            ny=ny,
            nz=nz,
            active=active,
        )
        return (
            acc + sample.astype(jnp.float32) * active.astype(jnp.float32) * step_size32,
            ix + dix,
            iy + diy,
        )

    init = (
        jnp.zeros_like(ix0, dtype=jnp.float32),
        ix0,
        iy0,
    )
    if unroll is None:
        acc, _, _ = jax.lax.fori_loop(0, n_steps, body, init)
    else:
        acc, _, _ = static_fori_loop(n_steps, body, init, unroll=unroll)
    plt.store(
        out_ref, acc.astype(jnp.float32)[jnp.newaxis, :, :], mask=in_detector[jnp.newaxis, :, :]
    )
