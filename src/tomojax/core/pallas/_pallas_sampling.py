from __future__ import annotations

from typing import Any

import jax
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp

from ._pallas_config import _KERNEL_VARIANT_IDS


def _trilinear_load_active(
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
    # The outer ray loop already bounds the tile's active traversal. Mask each
    # load instead of reducing all lanes and branching again at every sample.
    if kernel_variant_id == _KERNEL_VARIANT_IDS["z_integer4"]:
        return _trilinear_load_z_integer(
            volume_ref,
            ix,
            iy,
            iz,
            nx=nx,
            ny=ny,
            nz=nz,
            active=active,
        )
    return _trilinear_load(
        volume_ref,
        ix,
        iy,
        iz,
        nx=nx,
        ny=ny,
        nz=nz,
        active=active,
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
    active: jnp.ndarray,
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
        inb = active & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
        idx = ix * (ny * nz) + iy * nz + iz
        # Masked lanes never dereference their addresses. Clamping the flattened
        # index adds work to all eight loads without changing boundary values.
        return plt.load(volume_ref.at[idx], mask=inb, other=0.0)

    # Factor the tensor product: apply x/y weights after interpolating in z.
    # Keep float32 weights and zero extension at every face, edge, and corner.
    c00 = gather(fx, fy, fz) * wz0 + gather(fx, fy, cz) * wz1
    c01 = gather(fx, cy, fz) * wz0 + gather(fx, cy, cz) * wz1
    c10 = gather(cx, fy, fz) * wz0 + gather(cx, fy, cz) * wz1
    c11 = gather(cx, cy, fz) * wz0 + gather(cx, cy, cz) * wz1
    return (c00 * wy0 + c01 * wy1) * wx0 + (c10 * wy0 + c11 * wy1) * wx1


def _trilinear_load_z_integer(
    volume_ref: Any,
    ix_f: jnp.ndarray,
    iy_f: jnp.ndarray,
    iz_f: jnp.ndarray,
    *,
    nx: int,
    ny: int,
    nz: int,
    active: jnp.ndarray,
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
        inb = active & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
        idx = ix * (ny * nz) + iy * nz + iz
        return plt.load(volume_ref.at[idx], mask=inb, other=0.0)

    c0 = gather(fx, fy) * wy0 + gather(fx, cy) * wy1
    c1 = gather(cx, fy) * wy0 + gather(cx, cy) * wy1
    return c0 * wx0 + c1 * wx1


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
    interpret: bool = False,
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
        inb = (
            active
            & (weight != 0.0)
            & (ix >= 0)
            & (ix < nx)
            & (iy >= 0)
            & (iy < ny)
            & (iz >= 0)
            & (iz < nz)
        )
        idx = ix * (ny * nz) + iy * nz + iz
        idx = jnp.clip(idx, 0, (nx * ny * nz) - 1)
        if interpret:
            # The Triton interpreter neither accepts atomic masks nor reduces
            # repeated indices in a vector atomic. Scalar updates model both
            # correctly, while CUDA retains the native masked atomic below.
            indices = idx.ravel()
            values = jnp.where(inb, ray_vals * weight, 0.0).ravel()

            def scalar_add(i, _):
                plt.atomic_add(out_ref, (indices[i],), values[i])

            jax.lax.fori_loop(0, indices.size, scalar_add, None)
        else:
            plt.atomic_add(out_ref, (idx,), ray_vals * weight, mask=inb)

    add(fx, fy, fz, wx0 * wy0 * wz0)
    add(fx, fy, cz, wx0 * wy0 * wz1)
    add(fx, cy, fz, wx0 * wy1 * wz0)
    add(fx, cy, cz, wx0 * wy1 * wz1)
    add(cx, fy, fz, wx1 * wy0 * wz0)
    add(cx, fy, cz, wx1 * wy0 * wz1)
    add(cx, cy, fz, wx1 * wy1 * wz0)
    add(cx, cy, cz, wx1 * wy1 * wz1)
