"""Facade for the optional CUDA projector backend."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import jax.numpy as jnp

    from tomojax.core.geometry.base import Detector, Grid

from ._config import (
    CudaForwardProjectorStackTraversalState,
    CudaForwardProjectorTraversalState,
    CudaProjectorTraversalMetadata,
    CudaProjectorUnsupported,
    cuda_projector_actual_sinogram_variant_metadata,
    cuda_projector_actual_variant_metadata,
    cuda_projector_sinogram_unsupported_reason as _cuda_projector_sinogram_unsupported_reason,
    cuda_projector_traversal_metadata,
    cuda_projector_unsupported_reason as _cuda_projector_unsupported_reason,
    cuda_projector_variant_metadata,
)
from ._single_view import (
    BoundForwardProjectViewTCuda,
    bind_forward_project_view_T_cuda as _bind_forward_project_view_T_cuda,
    block_forward_project_view_T_cuda_state,
    forward_project_view_T_cuda as _forward_project_view_T_cuda,
    forward_project_view_T_cuda_with_state,
    prepare_forward_project_view_T_cuda_state,
)
from ._views import (
    BoundForwardProjectResidualSseTCuda,
    BoundForwardProjectViewsTCuda,
    backproject_view_T_cuda,
    bind_forward_project_residual_sse_T_cuda as _bind_forward_project_residual_sse_T_cuda,
    bind_forward_project_views_T_cuda as _bind_forward_project_views_T_cuda,
    block_forward_project_views_T_cuda_state,
    forward_project_loss_and_grad_T_cuda as _forward_project_loss_and_grad_T_cuda,
    forward_project_parallel_z_views_cuda,
    forward_project_residual_sse_T_cuda as _forward_project_residual_sse_T_cuda,
    forward_project_residual_sse_T_cuda_with_state,
    forward_project_views_T_cuda as _forward_project_views_T_cuda,
    forward_project_views_T_cuda_with_state,
    prepare_forward_project_views_T_cuda_state,
    sum_backproject_views_T_cuda,
)


@dataclass(frozen=True, slots=True)
class CudaProjectorOptions:
    """Runtime and kernel options shared by public projector entry points.

    ``interpret`` runs the portable JAX reference. CUDA uses at least
    ``num_warps`` warps, increasing that count when needed to cover a detector tile.
    """

    step_size: float | None = None
    n_steps: int | None = None
    unroll: int | None = None
    gather_dtype: str = "fp32"
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None
    interpret: bool = False
    tile_shape: tuple[int, int] = (8, 16)
    num_warps: int = 4
    kernel_variant: str = "auto"
    layout_variant: str = "detector_vu"
    state_mode: str = "inline"
    block_state: bool = True


def _opts(options: CudaProjectorOptions | None) -> CudaProjectorOptions:
    return CudaProjectorOptions() if options is None else options


def _projector_kwargs(options: CudaProjectorOptions) -> dict[str, Any]:
    """Forward shared kernel options without copying detector-coordinate arrays."""
    return {
        "step_size": options.step_size,
        "n_steps": options.n_steps,
        "gather_dtype": options.gather_dtype,
        "det_grid": options.det_grid,
        "tile_shape": options.tile_shape,
        "num_warps": options.num_warps,
        "kernel_variant": options.kernel_variant,
        "layout_variant": options.layout_variant,
    }


def _opts_with_overrides(
    options: CudaProjectorOptions | None, **overrides: Any
) -> CudaProjectorOptions:
    changes = {name: value for name, value in overrides.items() if value is not None}
    opts = _opts(options)
    return replace(opts, **changes) if changes else opts


def forward_project_view_T_cuda(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    *,
    options: CudaProjectorOptions | None = None,
) -> jnp.ndarray:
    """Forward project one view using the configured CUDA projector options."""
    opts = _opts(options)
    return _forward_project_view_T_cuda(
        T,
        grid,
        detector,
        volume,
        **_projector_kwargs(opts),
        unroll=opts.unroll,
        interpret=opts.interpret,
        state_mode=opts.state_mode,
    )


def bind_forward_project_view_T_cuda(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    *,
    options: CudaProjectorOptions | None = None,
) -> BoundForwardProjectViewTCuda:
    """Bind fixed single-view CUDA traversal state using one options object."""
    opts = _opts(options)
    return _bind_forward_project_view_T_cuda(
        T,
        grid,
        detector,
        **_projector_kwargs(opts),
        unroll=opts.unroll,
        interpret=opts.interpret,
        block_state=opts.block_state,
    )


def forward_project_views_T_cuda(
    T_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    *,
    options: CudaProjectorOptions | None = None,
    step_size: float | None = None,
    n_steps: int | None = None,
    unroll: int | None = None,
    gather_dtype: str | None = None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    interpret: bool | None = None,
    tile_shape: tuple[int, int] | None = None,
    num_warps: int | None = None,
    kernel_variant: str | None = None,
    layout_variant: str | None = None,
    state_mode: str | None = None,
) -> jnp.ndarray:
    """Forward project a stack of views using one configured CUDA options object."""
    opts = _opts_with_overrides(
        options,
        step_size=step_size,
        n_steps=n_steps,
        unroll=unroll,
        gather_dtype=gather_dtype,
        det_grid=det_grid,
        interpret=interpret,
        tile_shape=tile_shape,
        num_warps=num_warps,
        kernel_variant=kernel_variant,
        layout_variant=layout_variant,
        state_mode=state_mode,
    )
    return _forward_project_views_T_cuda(
        T_stack,
        grid,
        detector,
        volume,
        **_projector_kwargs(opts),
        unroll=opts.unroll,
        interpret=opts.interpret,
        state_mode=opts.state_mode,
    )


def bind_forward_project_views_T_cuda(
    T_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    *,
    options: CudaProjectorOptions | None = None,
) -> BoundForwardProjectViewsTCuda:
    """Bind fixed stack traversal state using one CUDA options object."""
    opts = _opts(options)
    return _bind_forward_project_views_T_cuda(
        T_stack,
        grid,
        detector,
        **_projector_kwargs(opts),
        unroll=opts.unroll,
        interpret=opts.interpret,
        block_state=opts.block_state,
    )


def bind_forward_project_residual_sse_T_cuda(
    T_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    target: jnp.ndarray,
    *,
    options: CudaProjectorOptions | None = None,
) -> BoundForwardProjectResidualSseTCuda:
    """Bind fixed stack residual state using one CUDA options object."""
    opts = _opts(options)
    return _bind_forward_project_residual_sse_T_cuda(
        T_stack,
        grid,
        detector,
        target,
        **_projector_kwargs(opts),
        unroll=opts.unroll,
        interpret=opts.interpret,
        block_state=opts.block_state,
    )


def forward_project_loss_and_grad_T_cuda(
    T_all: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    target: jnp.ndarray,
    *,
    weights: jnp.ndarray | None = None,
    compute_loss: bool = True,
    options: CudaProjectorOptions | None = None,
    step_size: float | None = None,
    n_steps: int | None = None,
    unroll: int | None = None,
    gather_dtype: str | None = None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    interpret: bool | None = None,
    tile_shape: tuple[int, int] | None = None,
    num_warps: int | None = None,
    kernel_variant: str | None = None,
    layout_variant: str | None = None,
    state_mode: str | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return CUDA projection loss/gradient using one shared options object."""
    opts = _opts_with_overrides(
        options,
        step_size=step_size,
        n_steps=n_steps,
        unroll=unroll,
        gather_dtype=gather_dtype,
        det_grid=det_grid,
        interpret=interpret,
        tile_shape=tile_shape,
        num_warps=num_warps,
        kernel_variant=kernel_variant,
        layout_variant=layout_variant,
        state_mode=state_mode,
    )
    return _forward_project_loss_and_grad_T_cuda(
        T_all,
        grid,
        detector,
        volume,
        target,
        weights=weights,
        **_projector_kwargs(opts),
        unroll=opts.unroll,
        interpret=opts.interpret,
        compute_loss=compute_loss,
    )


def forward_project_residual_sse_T_cuda(
    T_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    target: jnp.ndarray,
    *,
    options: CudaProjectorOptions | None = None,
) -> jnp.ndarray:
    """Return CUDA residual SSE using one shared options object."""
    opts = _opts(options)
    return _forward_project_residual_sse_T_cuda(
        T_stack,
        grid,
        detector,
        volume,
        target,
        **_projector_kwargs(opts),
        unroll=opts.unroll,
        interpret=opts.interpret,
        state_mode=opts.state_mode,
    )


def cuda_projector_unsupported_reason(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    *,
    options: CudaProjectorOptions | None = None,
) -> str | None:
    """Return why single-view CUDA projection is unsupported, if it is."""
    opts = _opts(options)
    return _cuda_projector_unsupported_reason(
        T,
        grid,
        detector,
        volume,
        **_projector_kwargs(opts),
        state_mode=opts.state_mode,
    )


def cuda_projector_sinogram_unsupported_reason(
    T_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    *,
    options: CudaProjectorOptions | None = None,
) -> str | None:
    """Return why batched CUDA projection is unsupported, if it is."""
    opts = _opts(options)
    return _cuda_projector_sinogram_unsupported_reason(
        T_stack,
        grid,
        detector,
        volume,
        **_projector_kwargs(opts),
        state_mode=opts.state_mode,
    )


__all__ = [
    "BoundForwardProjectResidualSseTCuda",
    "BoundForwardProjectViewTCuda",
    "BoundForwardProjectViewsTCuda",
    "CudaForwardProjectorStackTraversalState",
    "CudaForwardProjectorTraversalState",
    "CudaProjectorOptions",
    "CudaProjectorTraversalMetadata",
    "CudaProjectorUnsupported",
    "backproject_view_T_cuda",
    "bind_forward_project_residual_sse_T_cuda",
    "bind_forward_project_view_T_cuda",
    "bind_forward_project_views_T_cuda",
    "block_forward_project_view_T_cuda_state",
    "block_forward_project_views_T_cuda_state",
    "cuda_projector_actual_sinogram_variant_metadata",
    "cuda_projector_actual_variant_metadata",
    "cuda_projector_sinogram_unsupported_reason",
    "cuda_projector_traversal_metadata",
    "cuda_projector_unsupported_reason",
    "cuda_projector_variant_metadata",
    "forward_project_loss_and_grad_T_cuda",
    "forward_project_parallel_z_views_cuda",
    "forward_project_residual_sse_T_cuda",
    "forward_project_residual_sse_T_cuda_with_state",
    "forward_project_view_T_cuda",
    "forward_project_view_T_cuda_with_state",
    "forward_project_views_T_cuda",
    "forward_project_views_T_cuda_with_state",
    "prepare_forward_project_view_T_cuda_state",
    "prepare_forward_project_views_T_cuda_state",
    "sum_backproject_views_T_cuda",
]
