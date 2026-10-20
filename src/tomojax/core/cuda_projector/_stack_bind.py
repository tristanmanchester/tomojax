from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from tomojax.core.geometry.base import Detector, Grid, grid_volume_origin
from tomojax.core.validation import validate_pose_matrix, validate_projection_stack

from ._config import (
    CudaForwardProjectorStackTraversalState,
    CudaProjectorUnsupported,
    _normalize_state_mode,
    _prepare_volume_for_cuda_gather,
    _unsupported,
    _validate_public_sinogram_call,
)
from ._inline_call import inline_projector_call
from ._stack_call import (
    forward_project_residual_sse_T_cuda_with_state,
    forward_project_views_T_cuda_with_state,
)
from ._stack_state import (
    block_forward_project_views_T_cuda_state,
    prepare_forward_project_views_T_cuda_state,
)


class BoundForwardProjectViewsTCuda:
    """Fixed pose-stack CUDA projector callable for repeated-volume workflows."""

    def __init__(
        self,
        state: CudaForwardProjectorStackTraversalState,
        *,
        interpret: bool = False,
        unroll: int | None = None,
    ) -> None:
        self.state = state
        self.interpret = bool(interpret)
        self.unroll = unroll

    def __call__(self, volume: jnp.ndarray) -> jnp.ndarray:
        """Project ``volume`` for the cached pose stack."""
        return forward_project_views_T_cuda_with_state(
            self.state,
            volume,
            interpret=self.interpret,
            unroll=self.unroll,
        )


class BoundForwardProjectResidualSseTCuda:
    """Fixed pose-stack CUDA residual callable for repeated-volume workflows."""

    def __init__(
        self,
        state: CudaForwardProjectorStackTraversalState,
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
        return forward_project_residual_sse_T_cuda_with_state(
            self.state,
            volume,
            self.target,
            interpret=self.interpret,
            unroll=self.unroll,
        )


def bind_forward_project_views_T_cuda(
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
) -> BoundForwardProjectViewsTCuda:
    """Bind fixed pose-stack traversal state once and return a sinogram callable."""
    state = prepare_forward_project_views_T_cuda_state(
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
        block_forward_project_views_T_cuda_state(state)
    return BoundForwardProjectViewsTCuda(state, interpret=interpret, unroll=unroll)


def bind_forward_project_residual_sse_T_cuda(
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
) -> BoundForwardProjectResidualSseTCuda:
    """Bind fixed pose-stack traversal state once and return a residual SSE callable."""
    validate_projection_stack(
        target,
        detector,
        context="bind_forward_project_residual_sse_T_cuda",
    )
    if int(target.shape[0]) != int(T_stack.shape[0]):
        raise CudaProjectorUnsupported(
            _unsupported(
                "target view count does not match pose stack: "
                f"got {int(target.shape[0])}, expected {int(T_stack.shape[0])}"
            )
        )
    state = prepare_forward_project_views_T_cuda_state(
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
        block_forward_project_views_T_cuda_state(state)
    return BoundForwardProjectResidualSseTCuda(
        state,
        target,
        interpret=interpret,
        unroll=unroll,
    )


def forward_project_views_T_cuda(
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
    """Forward project a stack of views using one experimental batched CUDA launch."""
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
        state = prepare_forward_project_views_T_cuda_state(
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
        return forward_project_views_T_cuda_with_state(
            state,
            volume,
            interpret=interpret,
            unroll=unroll,
        )
    vol_origin = grid_volume_origin(grid)
    return inline_projector_call(
        nx=nx,
        ny=ny,
        nz=nz,
        nv=nv,
        nu=nu,
        n_views=n_views,
        du=float(detector.du),
        dv=float(detector.dv),
        det_center_x=float(detector.center[0]),
        det_center_z=float(detector.center[1]),
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
    )(
        jnp.asarray(T_stack, dtype=jnp.float32),
        _prepare_volume_for_cuda_gather(volume, gather_dtype),
        None,
    )


def forward_project_view_T_cuda(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    **options: Any,
) -> jnp.ndarray:
    """Forward project one view: the pose stack of that single view."""
    validate_pose_matrix(T, context="forward_project_view_T_cuda")
    T_stack = jnp.asarray(T, dtype=jnp.float32)[None]
    return forward_project_views_T_cuda(T_stack, grid, detector, volume, **options)[0]
