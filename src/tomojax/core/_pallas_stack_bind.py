from __future__ import annotations

from collections.abc import Callable
import functools
import math

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp

from ._pallas_config import (
    PallasForwardProjectorStackTraversalState,
    PallasProjectorUnsupported,
    _normalize_state_mode,
    _prepare_volume_for_pallas_gather,
    _unsupported,
    _validate_public_sinogram_call,
)
from ._pallas_kernels import _projector_views_kernel
from ._pallas_stack_call import (
    forward_project_residual_sse_T_pallas_with_state,
    forward_project_views_T_pallas_with_state,
)
from ._pallas_stack_state import (
    block_forward_project_views_T_pallas_state,
    prepare_forward_project_views_T_pallas_state,
)
from .geometry.base import Detector, Grid, _grid_volume_origin
from .validation import validate_projection_stack


class BoundForwardProjectViewsTPallas:
    """Fixed pose-stack Pallas projector callable for repeated-volume workflows."""

    def __init__(
        self,
        state: PallasForwardProjectorStackTraversalState,
        *,
        interpret: bool = False,
        unroll: int | None = None,
    ) -> None:
        self.state = state
        self.interpret = bool(interpret)
        self.unroll = unroll

    def __call__(self, volume: jnp.ndarray) -> jnp.ndarray:
        """Project ``volume`` for the cached pose stack."""
        return forward_project_views_T_pallas_with_state(
            self.state,
            volume,
            interpret=self.interpret,
            unroll=self.unroll,
        )


class BoundForwardProjectResidualSseTPallas:
    """Fixed pose-stack Pallas residual callable for repeated-volume workflows."""

    def __init__(
        self,
        state: PallasForwardProjectorStackTraversalState,
        target: jnp.ndarray,
        *,
        interpret: bool = False,
        unroll: int | None = None,
    ) -> None:
        self.state = state
        self.target = jnp.asarray(target, dtype=jnp.float32)
        self.interpret = bool(interpret)
        self.unroll = unroll

    def __call__(self, volume: jnp.ndarray) -> jnp.ndarray:
        """Return residual SSE for ``volume`` against the cached target stack."""
        return forward_project_residual_sse_T_pallas_with_state(
            self.state,
            volume,
            self.target,
            interpret=self.interpret,
            unroll=self.unroll,
        )


def bind_forward_project_views_T_pallas(
    T_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
    unroll: int | None = None,
    gather_dtype: str = "fp32",
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    interpret: bool = False,
    tile_shape: tuple[int, int] = (8, 16),
    num_warps: int = 4,
    kernel_variant: str = "auto",
    layout_variant: str = "detector_vu",
    block_state: bool = True,
) -> BoundForwardProjectViewsTPallas:
    """Bind fixed pose-stack traversal state once and return a sinogram callable."""
    state = prepare_forward_project_views_T_pallas_state(
        T_stack,
        grid,
        detector,
        step_size=step_size,
        n_steps=n_steps,
        gather_dtype=gather_dtype,
        det_grid=det_grid,
        tile_shape=tile_shape,
        num_warps=num_warps,
        kernel_variant=kernel_variant,
        layout_variant=layout_variant,
    )
    if block_state:
        block_forward_project_views_T_pallas_state(state)
    return BoundForwardProjectViewsTPallas(state, interpret=interpret, unroll=unroll)


def bind_forward_project_residual_sse_T_pallas(
    T_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    target: jnp.ndarray,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
    unroll: int | None = None,
    gather_dtype: str = "fp32",
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    interpret: bool = False,
    tile_shape: tuple[int, int] = (8, 16),
    num_warps: int = 4,
    kernel_variant: str = "auto",
    layout_variant: str = "detector_vu",
    block_state: bool = True,
) -> BoundForwardProjectResidualSseTPallas:
    """Bind fixed pose-stack traversal state once and return a residual SSE callable."""
    validate_projection_stack(
        target,
        detector,
        context="bind_forward_project_residual_sse_T_pallas",
    )
    if int(target.shape[0]) != int(T_stack.shape[0]):
        raise PallasProjectorUnsupported(
            _unsupported(
                "target view count does not match pose stack: "
                f"got {int(target.shape[0])}, expected {int(T_stack.shape[0])}"
            )
        )
    state = prepare_forward_project_views_T_pallas_state(
        T_stack,
        grid,
        detector,
        step_size=step_size,
        n_steps=n_steps,
        gather_dtype=gather_dtype,
        det_grid=det_grid,
        tile_shape=tile_shape,
        num_warps=num_warps,
        kernel_variant=kernel_variant,
        layout_variant=layout_variant,
    )
    if block_state:
        block_forward_project_views_T_pallas_state(state)
    return BoundForwardProjectResidualSseTPallas(
        state,
        target,
        interpret=interpret,
        unroll=unroll,
    )


def forward_project_views_T_pallas(
    T_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
    unroll: int | None = None,
    gather_dtype: str = "fp32",
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    interpret: bool = False,
    tile_shape: tuple[int, int] = (8, 16),
    num_warps: int = 4,
    kernel_variant: str = "auto",
    layout_variant: str = "detector_vu",
    state_mode: str = "inline",
) -> jnp.ndarray:
    """Forward project a stack of views using one experimental batched Pallas launch."""
    (
        nx,
        ny,
        nz,
        nv,
        nu,
        n_views,
        _volume_size,
        step_size_value,
        _n_steps_value,
        effective_n_steps_value,
        (tile_v, tile_u),
        num_warps_value,
        kernel_variant_id,
        layout_variant_id,
    ) = _validate_public_sinogram_call(
        T_stack,
        grid,
        detector,
        volume,
        step_size=step_size,
        n_steps=n_steps,
        gather_dtype=gather_dtype,
        det_grid=det_grid,
        interpret=interpret,
        tile_shape=tile_shape,
        num_warps=num_warps,
        kernel_variant=kernel_variant,
        layout_variant=layout_variant,
        state_mode=state_mode,
    )
    if _normalize_state_mode(state_mode) != "inline":
        state = prepare_forward_project_views_T_pallas_state(
            T_stack,
            grid,
            detector,
            step_size=step_size,
            n_steps=n_steps,
            gather_dtype=gather_dtype,
            det_grid=det_grid,
            tile_shape=tile_shape,
            num_warps=num_warps,
            kernel_variant=kernel_variant,
            layout_variant=layout_variant,
        )
        return forward_project_views_T_pallas_with_state(
            state,
            volume,
            interpret=interpret,
            unroll=unroll,
        )
    vol_origin = _grid_volume_origin(grid)
    call = _cached_projector_views_pallas_call(
        nx=nx,
        ny=ny,
        nz=nz,
        nv=nv,
        nu=nu,
        n_views=n_views,
        du=float(detector.du),
        dv=float(detector.dv),
        det_center_x=float(detector.det_center[0]),
        det_center_z=float(detector.det_center[1]),
        vol_origin_x=float(vol_origin[0]),
        vol_origin_y=float(vol_origin[1]),
        vol_origin_z=float(vol_origin[2]),
        vx=float(grid.vx),
        vy=float(grid.vy),
        vz=float(grid.vz),
        step_size=float(step_size_value),
        n_steps=int(effective_n_steps_value),
        tile_v=int(tile_v),
        tile_u=int(tile_u),
        num_warps=int(num_warps_value),
        kernel_variant_id=int(kernel_variant_id),
        layout_variant_id=int(layout_variant_id),
        unroll=unroll,
        interpret=bool(interpret),
    )
    return call(
        jnp.asarray(T_stack, dtype=jnp.float32),
        _prepare_volume_for_pallas_gather(volume, gather_dtype),
    )


@functools.lru_cache(maxsize=32)
def _cached_projector_views_pallas_call(
    *,
    nx: int,
    ny: int,
    nz: int,
    nv: int,
    nu: int,
    n_views: int,
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
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    kernel = functools.partial(
        _projector_views_kernel,
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
    grid_shape = (int(n_views), math.ceil(int(nv) / int(tile_v)), math.ceil(int(nu) / int(tile_u)))
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((int(n_views), int(nv), int(nu)), jnp.float32),
        grid=grid_shape,
        in_specs=[
            pl.no_block_spec,
            pl.no_block_spec,
        ],
        out_specs=pl.BlockSpec(
            (1, int(tile_v), int(tile_u)),
            lambda view, pv, pu: (view, pv, pu),
        ),
        interpret=bool(interpret),
        compiler_params=plt.CompilerParams(num_warps=int(num_warps)),
        name="tomojax_forward_project_views_T_pallas",
    )
