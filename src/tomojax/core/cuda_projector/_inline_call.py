"""Cached inline projection launches; cache keys never contain device arrays."""

from __future__ import annotations

from collections.abc import Callable
import functools

import jax
import jax.numpy as jnp

from tomojax.core._sampled_cuda import run


@functools.lru_cache(maxsize=32)
def inline_projector_call(
    *,
    nx: int,
    ny: int,
    nz: int,
    nv: int,
    nu: int,
    n_views: int | None,
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
    num_warps: int,
    kernel_variant_id: int,
    layout_variant_id: int,
    unroll: int | None,
    interpret: bool,
    residual: bool = False,
) -> Callable[..., jnp.ndarray]:
    params = locals()

    @jax.jit
    def call(poses, volume, target):
        stacked = n_views is not None
        out = run(
            poses if stacked else poses[None],
            volume,
            target if stacked or target is None else target[None],
            mode=3 if residual else 0,
            params=params,
        )
        return out if stacked else out[0]

    return call
