from __future__ import annotations

from typing import Any

import jax
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp

from ._pallas_config import _KERNEL_VARIANT_IDS


def _trilinear_load_when_tile_active(
    volume_ref: Any,
    ix: jnp.ndarray,
    iy: jnp.ndarray,
    iz: jnp.ndarray,
    *,
    nx: int,
    ny: int,
    nz: int,
    active: jnp.ndarray,
    kernel_variant_id: int,
) -> jnp.ndarray:
    def do_load(_):
        if kernel_variant_id == _KERNEL_VARIANT_IDS["z_integer4"]:
            return _trilinear_load_z_integer(
                volume_ref,
                ix,
                iy,
                iz,
                nx=nx,
                ny=ny,
                nz=nz,
            )
        return _trilinear_load(
            volume_ref,
            ix,
            iy,
            iz,
            nx=nx,
            ny=ny,
            nz=nz,
        )

    # Pallas/Triton lowering in JAX 0.10.0 supports reduce_max but not reduce_or.
    tile_active = jnp.max(active.astype(jnp.int32)) != jnp.int32(0)
    return jax.lax.cond(
        tile_active,
        do_load,
        lambda _: jnp.zeros_like(ix, dtype=jnp.float32),
        operand=None,
    )


def _trilinear_load(
    volume_ref: Any,
    ix_f: jnp.ndarray,
    iy_f: jnp.ndarray,
    iz_f: jnp.ndarray,
    *,
    nx: int,
    ny: int,
    nz: int,
) -> jnp.ndarray:
    fx = jnp.floor(ix_f).astype(jnp.int32)
    fy = jnp.floor(iy_f).astype(jnp.int32)
    fz = jnp.floor(iz_f).astype(jnp.int32)
    cx, cy, cz = fx + 1, fy + 1, fz + 1

    wx1 = ix_f - fx.astype(jnp.float32)
    wy1 = iy_f - fy.astype(jnp.float32)
    wz1 = iz_f - fz.astype(jnp.float32)
    wx0 = jnp.float32(1.0) - wx1
    wy0 = jnp.float32(1.0) - wy1
    wz0 = jnp.float32(1.0) - wz1

    def gather(ix: jnp.ndarray, iy: jnp.ndarray, iz: jnp.ndarray) -> jnp.ndarray:
        inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
        idx = ix * (ny * nz) + iy * nz + iz
        idx = jnp.clip(idx, 0, (nx * ny * nz) - 1)
        return plt.load(volume_ref.at[idx], mask=inb, other=0.0)

    c000 = gather(fx, fy, fz) * (wx0 * wy0 * wz0)
    c001 = gather(fx, fy, cz) * (wx0 * wy0 * wz1)
    c010 = gather(fx, cy, fz) * (wx0 * wy1 * wz0)
    c011 = gather(fx, cy, cz) * (wx0 * wy1 * wz1)
    c100 = gather(cx, fy, fz) * (wx1 * wy0 * wz0)
    c101 = gather(cx, fy, cz) * (wx1 * wy0 * wz1)
    c110 = gather(cx, cy, fz) * (wx1 * wy1 * wz0)
    c111 = gather(cx, cy, cz) * (wx1 * wy1 * wz1)
    return c000 + c001 + c010 + c011 + c100 + c101 + c110 + c111


def _trilinear_load_z_integer(
    volume_ref: Any,
    ix_f: jnp.ndarray,
    iy_f: jnp.ndarray,
    iz_f: jnp.ndarray,
    *,
    nx: int,
    ny: int,
    nz: int,
) -> jnp.ndarray:
    fx = jnp.floor(ix_f).astype(jnp.int32)
    fy = jnp.floor(iy_f).astype(jnp.int32)
    iz = jnp.floor(iz_f + jnp.float32(0.5)).astype(jnp.int32)
    cx, cy = fx + 1, fy + 1

    wx1 = ix_f - fx.astype(jnp.float32)
    wy1 = iy_f - fy.astype(jnp.float32)
    wx0 = jnp.float32(1.0) - wx1
    wy0 = jnp.float32(1.0) - wy1

    def gather(ix: jnp.ndarray, iy: jnp.ndarray) -> jnp.ndarray:
        inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
        idx = ix * (ny * nz) + iy * nz + iz
        idx = jnp.clip(idx, 0, (nx * ny * nz) - 1)
        return plt.load(volume_ref.at[idx], mask=inb, other=0.0)

    c00 = gather(fx, fy) * (wx0 * wy0)
    c01 = gather(fx, cy) * (wx0 * wy1)
    c10 = gather(cx, fy) * (wx1 * wy0)
    c11 = gather(cx, cy) * (wx1 * wy1)
    return c00 + c01 + c10 + c11


def _trilinear_atomic_add(
    out_ref: Any,
    ray_vals: jnp.ndarray,
    ix_f: jnp.ndarray,
    iy_f: jnp.ndarray,
    iz_f: jnp.ndarray,
    *,
    nx: int,
    ny: int,
    nz: int,
    active: jnp.ndarray,
) -> None:
    fx = jnp.floor(ix_f).astype(jnp.int32)
    fy = jnp.floor(iy_f).astype(jnp.int32)
    fz = jnp.floor(iz_f).astype(jnp.int32)
    cx, cy, cz = fx + 1, fy + 1, fz + 1

    wx1 = ix_f - fx.astype(jnp.float32)
    wy1 = iy_f - fy.astype(jnp.float32)
    wz1 = iz_f - fz.astype(jnp.float32)
    wx0 = jnp.float32(1.0) - wx1
    wy0 = jnp.float32(1.0) - wy1
    wz0 = jnp.float32(1.0) - wz1

    def add(ix: jnp.ndarray, iy: jnp.ndarray, iz: jnp.ndarray, weight: jnp.ndarray) -> None:
        inb = active & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
        idx = ix * (ny * nz) + iy * nz + iz
        idx = jnp.clip(idx, 0, (nx * ny * nz) - 1)
        plt.atomic_add(out_ref, (idx,), ray_vals * weight, mask=inb)

    add(fx, fy, fz, wx0 * wy0 * wz0)
    add(fx, fy, cz, wx0 * wy0 * wz1)
    add(fx, cy, fz, wx0 * wy1 * wz0)
    add(fx, cy, cz, wx0 * wy1 * wz1)
    add(cx, fy, fz, wx1 * wy0 * wz0)
    add(cx, fy, cz, wx1 * wy0 * wz1)
    add(cx, cy, fz, wx1 * wy1 * wz0)
    add(cx, cy, cz, wx1 * wy1 * wz1)
