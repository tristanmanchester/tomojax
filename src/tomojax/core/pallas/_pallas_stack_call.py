from __future__ import annotations

from collections.abc import Callable
import functools

import jax
import jax.numpy as jnp

from tomojax.core._sampled_cuda import run

from ._pallas_config import (
    PallasForwardProjectorStackTraversalState,
    PallasProjectorUnsupported,
    _prepare_volume_for_pallas_gather,
    _unsupported,
    _validate_cached_volume,
)


@functools.lru_cache(maxsize=32)
def _cached_projector_state_pallas_call(
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
    residual: bool,
) -> Callable[..., jnp.ndarray]:
    params = locals()

    @jax.jit
    def call(ix, iy, iz, count, dx, dy, dz, volume, target):
        return run(
            dx[:, None],
            volume,
            target,
            mode=3 if residual else 0,
            params=params,
            state=(ix, iy, iz, count, dx, dy, dz),
        )

    return call


def forward_project_views_T_pallas_with_state(
    state: PallasForwardProjectorStackTraversalState,
    volume: jnp.ndarray,
    *,
    interpret: bool = False,
    unroll: int | None = None,
) -> jnp.ndarray:
    """Forward project a stack with prepared traversal state."""
    _validate_cached_volume(
        state.traversal, volume, context="forward_project_views_T_pallas_with_state"
    )
    if not interpret and jax.default_backend() == "cpu":
        raise PallasProjectorUnsupported(
            _unsupported("real Pallas lowering is unavailable on CPU; pass interpret=True")
        )
    tile_v, tile_u = state.tile_shape
    call = _cached_projector_state_pallas_call(
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
        residual=False,
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
        None,
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
    _validate_cached_volume(
        state.traversal, volume, context="forward_project_residual_sse_T_pallas_with_state"
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
    call = _cached_projector_state_pallas_call(
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
        residual=True,
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
