from __future__ import annotations

from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp

from ._pallas_config import _LAYOUT_VARIANT_IDS
from ._pallas_loop import paired_fori_loop, static_fori_loop
from ._pallas_ray_geometry import ray_traversal
from ._pallas_sampling import _trilinear_atomic_add, _trilinear_load_active


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
    interpret: bool,
    stacked: bool = False,
) -> None:
    tile_v_start = pl.program_id(1 if stacked else 0) * tile_v
    tile_u_start = pl.program_id(2 if stacked else 1) * tile_u
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
        if stacked:
            return plt.load(T_ref.at[pl.program_id(0), jnp.int32(row), jnp.int32(col)])
        return plt.load(
            T_ref.at[jnp.asarray(row, dtype=jnp.int32), jnp.asarray(col, dtype=jnp.int32)]
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
    image_slice = (
        image_ref.at[pl.program_id(0), det_v, det_u] if stacked else image_ref.at[det_v, det_u]
    )
    ray_vals = plt.load(image_slice, mask=in_detector, other=0.0) * step_size32

    def body(step_idx, carry):
        ix, iy, iz = carry
        active = in_detector & (step_idx < n_steps_ray)
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
            interpret=interpret,
        )
        return ix + dix, iy + diy, iz + diz

    # Replay the forward positions; reverse stepping drifts away in fp32.
    init = (ix0, iy0, iz0)
    if unroll is None:
        tile_steps = jnp.minimum(
            jnp.max(jnp.where(in_detector, n_steps_ray, 0)),
            jnp.asarray(n_steps, dtype=jnp.int32),
        )
        jax.lax.fori_loop(0, tile_steps, body, init)
    else:
        static_fori_loop(n_steps, body, init, unroll=unroll)


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
        active = step_idx < n_steps_ray
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
    interpret: bool,
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

    def fwd_body(step_idx, carry):
        acc, ix, iy, iz = carry
        active = step_idx < n_steps_ray
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
        acc, _, _, _ = paired_fori_loop(tile_steps, fwd_body, init)
    else:
        acc, _, _, _ = static_fori_loop(n_steps, fwd_body, init, unroll=unroll)

    target = plt.load(target_ref.at[view_idx, det_v, det_u], mask=in_detector, other=0.0)
    zero = jnp.asarray(0, dtype=jnp.int32)
    weight = plt.load(weights_ref.at[view_idx, zero, zero])
    raw_residual = jnp.where(in_detector, acc.astype(jnp.float32) - target.astype(jnp.float32), 0.0)
    weighted_residual = raw_residual * weight
    if compute_loss:
        loss_ref[0, 0, 0] = jnp.float32(0.5) * jnp.sum(
            weighted_residual * weighted_residual
        ).astype(jnp.float32)
    else:
        loss_ref[0, 0, 0] = jnp.float32(0.0)
    grad_residual = raw_residual * weight * weight * step_size32

    def bwd_body(step_idx, carry):
        ix, iy, iz = carry
        active = in_detector & (step_idx < n_steps_ray)
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
            interpret=interpret,
        )
        return ix + dix, iy + diy, iz + diz

    bwd_init = (ix0, iy0, iz0)
    if unroll is None:
        tile_steps = jnp.minimum(
            jnp.max(jnp.where(in_detector, n_steps_ray, 0)),
            jnp.asarray(n_steps, dtype=jnp.int32),
        )
        jax.lax.fori_loop(0, tile_steps, bwd_body, bwd_init)
    else:
        static_fori_loop(n_steps, bwd_body, bwd_init, unroll=unroll)
