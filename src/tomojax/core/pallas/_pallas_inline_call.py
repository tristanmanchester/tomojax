"""Cached inline projection launches; cache keys never contain device arrays."""

from __future__ import annotations

from collections.abc import Callable
import functools
import math

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp

from ._pallas_kernels import _projector_kernel


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
    """Bind geometry and output layout; absent targets consume no device buffer."""
    stacked = n_views is not None
    kernel = functools.partial(
        _projector_kernel,
        stacked=stacked,
        nx=int(nx),
        ny=int(ny),
        nz=int(nz),
        nu=int(nu),
        nv=int(nv),
        du=float(du),
        dv=float(dv),
        det_center_x=float(det_center_x),
        det_center_z=float(det_center_z),
        vol_origin_x=float(vol_origin_x),
        vol_origin_y=float(vol_origin_y),
        vol_origin_z=float(vol_origin_z),
        vx=float(vx),
        vy=float(vy),
        vz=float(vz),
        step_size=float(step_size),
        n_steps=int(n_steps),
        tile_v=int(tile_v),
        tile_u=int(tile_u),
        kernel_variant_id=int(kernel_variant_id),
        layout_variant_id=int(layout_variant_id),
        unroll=unroll,
    )
    grid_shape = (math.ceil(nv / tile_v), math.ceil(nu / tile_u))
    output_shape = grid_shape if residual else (nv, nu)
    output_tile = (1, 1) if residual else (tile_v, tile_u)
    if n_views is not None:
        grid_shape = (n_views, *grid_shape)
        output_shape = (n_views, *output_shape)
        output_tile = (1, *output_tile)
    name = "residual_sse" if residual else ("views" if stacked else "view")
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(output_shape, jnp.float32),
        grid=grid_shape,
        in_specs=[pl.no_block_spec] * 2 + [pl.no_block_spec if residual else None],
        out_specs=pl.BlockSpec(output_tile, lambda *indices: indices),
        interpret=bool(interpret),
        compiler_params=plt.CompilerParams(num_warps=int(num_warps)),
        name=f"tomojax_forward_project_{name}_T_pallas",
    )
