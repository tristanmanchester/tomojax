from __future__ import annotations

from collections.abc import Callable
import functools
import math
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp

from tomojax.core.geometry.base import Grid
from tomojax.core.validation import validate_volume

from ._pallas_config import (
    PallasForwardProjectorStackTraversalState,
    PallasProjectorUnsupported,
    _ensure_float32_volume,
    _prepare_volume_for_pallas_gather,
    _unsupported,
)
from ._pallas_ray_geometry import detector_tile
from ._pallas_ray_integral import integrate_rays


def _projector_views_kernel_cached(
    ix0_ref: Any,
    iy0_ref: Any,
    iz0_ref: Any,
    n_steps_ray_ref: Any,
    dix_ref: Any,
    diy_ref: Any,
    diz_ref: Any,
    volume_ref: Any,
    out_ref: Any,
    *,
    nx: int,
    ny: int,
    nz: int,
    nu: int,
    nv: int,
    n_views: int,
    step_size: float,
    n_steps: int,
    tile_v: int,
    tile_u: int,
    kernel_variant_id: int,
    unroll: int | None,
) -> None:
    view_idx = pl.program_id(0)
    det_u, det_v = detector_tile((tile_v, tile_u), stacked=True)
    in_detector = (det_u < nu) & (det_v < nv) & (view_idx < n_views)
    state_idx = view_idx * jnp.int32(nv * nu) + jnp.clip(det_v * nu + det_u, 0, (nu * nv) - 1)

    ix0 = plt.load(ix0_ref.at[state_idx], mask=in_detector, other=0.0)
    iy0 = plt.load(iy0_ref.at[state_idx], mask=in_detector, other=0.0)
    iz0 = plt.load(iz0_ref.at[state_idx], mask=in_detector, other=0.0)
    n_steps_ray = plt.load(n_steps_ray_ref.at[state_idx], mask=in_detector, other=0)
    dix = plt.load(dix_ref.at[view_idx])
    diy = plt.load(diy_ref.at[view_idx])
    diz = plt.load(diz_ref.at[view_idx])
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
    )
    plt.store(
        out_ref, acc.astype(jnp.float32)[jnp.newaxis, :, :], mask=in_detector[jnp.newaxis, :, :]
    )


def _projector_residual_sse_kernel_cached(
    ix0_ref: Any,
    iy0_ref: Any,
    iz0_ref: Any,
    n_steps_ray_ref: Any,
    dix_ref: Any,
    diy_ref: Any,
    diz_ref: Any,
    volume_ref: Any,
    target_ref: Any,
    out_ref: Any,
    *,
    nx: int,
    ny: int,
    nz: int,
    nu: int,
    nv: int,
    n_views: int,
    step_size: float,
    n_steps: int,
    tile_v: int,
    tile_u: int,
    kernel_variant_id: int,
    unroll: int | None,
) -> None:
    view_idx = pl.program_id(0)
    det_u, det_v = detector_tile((tile_v, tile_u), stacked=True)
    in_detector = (det_u < nu) & (det_v < nv) & (view_idx < n_views)
    state_idx = view_idx * jnp.int32(nv * nu) + jnp.clip(det_v * nu + det_u, 0, (nu * nv) - 1)

    ix0 = plt.load(ix0_ref.at[state_idx], mask=in_detector, other=0.0)
    iy0 = plt.load(iy0_ref.at[state_idx], mask=in_detector, other=0.0)
    iz0 = plt.load(iz0_ref.at[state_idx], mask=in_detector, other=0.0)
    n_steps_ray = plt.load(n_steps_ray_ref.at[state_idx], mask=in_detector, other=0)
    dix = plt.load(dix_ref.at[view_idx])
    diy = plt.load(diy_ref.at[view_idx])
    diz = plt.load(diz_ref.at[view_idx])
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
    )
    target = plt.load(target_ref.at[view_idx, det_v, det_u], mask=in_detector, other=0.0)
    residual = jnp.where(in_detector, acc.astype(jnp.float32) - target.astype(jnp.float32), 0.0)
    out_ref[0, 0, 0] = jnp.sum(residual * residual).astype(jnp.float32)


@functools.lru_cache(maxsize=32)
def _cached_projector_views_state_pallas_call(
    *,
    nx: int,
    ny: int,
    nz: int,
    nv: int,
    nu: int,
    n_views: int,
    step_size: float,
    n_steps: int,
    tile_v: int,
    tile_u: int,
    num_warps: int,
    kernel_variant_id: int,
    unroll: int | None,
    interpret: bool,
) -> Callable[
    [
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ],
    jnp.ndarray,
]:
    kernel = functools.partial(
        _projector_views_kernel_cached,
        nx=int(nx),
        ny=int(ny),
        nz=int(nz),
        nu=int(nu),
        nv=int(nv),
        n_views=int(n_views),
        step_size=float(step_size),
        n_steps=int(n_steps),
        tile_v=int(tile_v),
        tile_u=int(tile_u),
        kernel_variant_id=int(kernel_variant_id),
        unroll=unroll,
    )
    grid_shape = (int(n_views), math.ceil(int(nv) / int(tile_v)), math.ceil(int(nu) / int(tile_u)))
    out_nv = int(grid_shape[1]) * int(tile_v)
    out_nu = int(grid_shape[2]) * int(tile_u)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((int(n_views), out_nv, out_nu), jnp.float32),
        grid=grid_shape,
        in_specs=[
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ],
        out_specs=pl.BlockSpec(
            (1, int(tile_v), int(tile_u)),
            lambda view, pv, pu: (view, pv, pu),
        ),
        interpret=bool(interpret),
        compiler_params=plt.CompilerParams(num_warps=int(num_warps)),
        name="tomojax_forward_project_views_T_pallas_cached_state",
    )


@functools.lru_cache(maxsize=32)
def _cached_projector_residual_sse_state_pallas_call(
    *,
    nx: int,
    ny: int,
    nz: int,
    nv: int,
    nu: int,
    n_views: int,
    step_size: float,
    n_steps: int,
    tile_v: int,
    tile_u: int,
    num_warps: int,
    kernel_variant_id: int,
    unroll: int | None,
    interpret: bool,
) -> Callable[
    [
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ],
    jnp.ndarray,
]:
    kernel = functools.partial(
        _projector_residual_sse_kernel_cached,
        nx=int(nx),
        ny=int(ny),
        nz=int(nz),
        nu=int(nu),
        nv=int(nv),
        n_views=int(n_views),
        step_size=float(step_size),
        n_steps=int(n_steps),
        tile_v=int(tile_v),
        tile_u=int(tile_u),
        kernel_variant_id=int(kernel_variant_id),
        unroll=unroll,
    )
    grid_shape = (int(n_views), math.ceil(int(nv) / int(tile_v)), math.ceil(int(nu) / int(tile_u)))
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(
            (int(n_views), math.ceil(int(nv) / int(tile_v)), math.ceil(int(nu) / int(tile_u))),
            jnp.float32,
        ),
        grid=grid_shape,
        in_specs=[
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ],
        out_specs=pl.BlockSpec((1, 1, 1), lambda view, pv, pu: (view, pv, pu)),
        interpret=bool(interpret),
        compiler_params=plt.CompilerParams(num_warps=int(num_warps)),
        name="tomojax_forward_project_residual_sse_T_pallas_cached_state",
    )


def forward_project_views_T_pallas_with_state(
    state: PallasForwardProjectorStackTraversalState,
    volume: jnp.ndarray,
    *,
    interpret: bool = False,
    unroll: int | None = None,
) -> jnp.ndarray:
    """Forward project a stack with prepared traversal state."""
    nx, ny, nz = validate_volume(
        volume,
        Grid(nx=state.nx, ny=state.ny, nz=state.nz, vx=1.0, vy=1.0, vz=1.0),
        context="forward_project_views_T_pallas_with_state",
        name="volume",
    )
    _ensure_float32_volume(volume)
    if (nx, ny, nz) != (state.nx, state.ny, state.nz):
        raise PallasProjectorUnsupported(
            _unsupported(
                "volume shape does not match cached traversal state: "
                f"got {(nx, ny, nz)}, expected {(state.nx, state.ny, state.nz)}"
            )
        )
    if not interpret and jax.default_backend() == "cpu":
        raise PallasProjectorUnsupported(
            _unsupported("real Pallas lowering is unavailable on CPU; pass interpret=True")
        )
    tile_v, tile_u = state.tile_shape
    call = _cached_projector_views_state_pallas_call(
        nx=int(state.nx),
        ny=int(state.ny),
        nz=int(state.nz),
        nv=int(state.nv),
        nu=int(state.nu),
        n_views=int(state.n_views),
        step_size=float(state.step_size),
        n_steps=int(state.n_steps),
        tile_v=int(tile_v),
        tile_u=int(tile_u),
        num_warps=int(state.num_warps),
        kernel_variant_id=int(state.kernel_variant_id),
        unroll=unroll,
        interpret=bool(interpret),
    )
    return call(
        state.ix0,
        state.iy0,
        state.iz0,
        state.n_steps_ray,
        state.dix,
        state.diy,
        state.diz,
        _prepare_volume_for_pallas_gather(volume, state.gather_dtype),
    )[:, : int(state.nv), : int(state.nu)]


def forward_project_residual_sse_T_pallas_with_state(
    state: PallasForwardProjectorStackTraversalState,
    volume: jnp.ndarray,
    target: jnp.ndarray,
    *,
    interpret: bool = False,
    unroll: int | None = None,
) -> jnp.ndarray:
    """Return residual SSE for a stack using prepared traversal state."""
    nx, ny, nz = validate_volume(
        volume,
        Grid(nx=state.nx, ny=state.ny, nz=state.nz, vx=1.0, vy=1.0, vz=1.0),
        context="forward_project_residual_sse_T_pallas_with_state",
        name="volume",
    )
    _ensure_float32_volume(volume)
    if (nx, ny, nz) != (state.nx, state.ny, state.nz):
        raise PallasProjectorUnsupported(
            _unsupported(
                "volume shape does not match cached traversal state: "
                f"got {(nx, ny, nz)}, expected {(state.nx, state.ny, state.nz)}"
            )
        )
    if tuple(int(dim) for dim in target.shape) != (state.n_views, state.nv, state.nu):
        raise PallasProjectorUnsupported(
            _unsupported(
                "target shape does not match cached traversal state: "
                f"got {tuple(int(dim) for dim in target.shape)}, "
                f"expected {(state.n_views, state.nv, state.nu)}"
            )
        )
    if not interpret and jax.default_backend() == "cpu":
        raise PallasProjectorUnsupported(
            _unsupported("real Pallas lowering is unavailable on CPU; pass interpret=True")
        )
    tile_v, tile_u = state.tile_shape
    call = _cached_projector_residual_sse_state_pallas_call(
        nx=int(state.nx),
        ny=int(state.ny),
        nz=int(state.nz),
        nv=int(state.nv),
        nu=int(state.nu),
        n_views=int(state.n_views),
        step_size=float(state.step_size),
        n_steps=int(state.n_steps),
        tile_v=int(tile_v),
        tile_u=int(tile_u),
        num_warps=int(state.num_warps),
        kernel_variant_id=int(state.kernel_variant_id),
        unroll=unroll,
        interpret=bool(interpret),
    )
    partials = call(
        state.ix0,
        state.iy0,
        state.iz0,
        state.n_steps_ray,
        state.dix,
        state.diy,
        state.diz,
        _prepare_volume_for_pallas_gather(volume, state.gather_dtype),
        jnp.asarray(target, dtype=jnp.float32),
    )
    return jnp.sum(partials, dtype=jnp.float32)
