from __future__ import annotations

from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp

from ._pallas_config import _LAYOUT_VARIANT_IDS
from ._pallas_loop import static_fori_loop
from ._pallas_ray_geometry import detector_ray_traversal, detector_tile, slab_intersection
from ._pallas_ray_integral import integrate_rays
from ._pallas_sampling import _trilinear_load_z_integer


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
    stacked: bool = False,
) -> None:
    det_u, det_v = detector_tile(
        (tile_v, tile_u), stacked=stacked, layout_variant_id=layout_variant_id
    )
    in_detector = (det_u < nu) & (det_v < nv)

    ix0, iy0, iz0, dix, diy, diz, n_steps_ray = detector_ray_traversal(
        T_ref,
        det_u,
        det_v,
        detector_shape=(nv, nu),
        detector_spacing=(du, dv),
        detector_center=(det_center_x, det_center_z),
        shape=(nx, ny, nz),
        origin=(vol_origin_x, vol_origin_y, vol_origin_z),
        voxel_size=(vx, vy, vz),
        step_size=step_size,
        stacked=stacked,
    )
    acc = integrate_rays(
        volume_ref,
        (ix0, iy0, iz0),
        (dix, diy, diz),
        n_steps_ray,
        shape=(nx, ny, nz),
        in_detector=in_detector,
        step_size=step_size,
        n_steps=n_steps,
        kernel_variant_id=kernel_variant_id,
        unroll=unroll,
        mask_detector=True,
    )
    if layout_variant_id == _LAYOUT_VARIANT_IDS["detector_uv"]:
        acc, in_detector = acc.T, in_detector.T
    if stacked:
        acc, in_detector = acc[jnp.newaxis, :, :], in_detector[jnp.newaxis, :, :]
    plt.store(out_ref, acc.astype(jnp.float32), mask=in_detector)


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
    det_u, det_v = detector_tile((tile_v, tile_u), stacked=True)
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

    lo_x, hi_x = slab_intersection(base_x, ey_x, lower_x, upper_x)
    lo_y, hi_y = slab_intersection(base_y, ey_y, lower_y, upper_y)
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
