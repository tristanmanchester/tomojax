"""Facade for the optional Pallas projector backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax.numpy as jnp

    from tomojax.core.geometry.base import Detector, Grid

from ._pallas_config import (
    PallasForwardProjectorStackTraversalState,
    PallasForwardProjectorTraversalState,
    PallasProjectorTraversalMetadata,
    PallasProjectorUnsupported,
    pallas_projector_actual_sinogram_variant_metadata,
    pallas_projector_actual_variant_metadata,
    pallas_projector_sinogram_traversal_metadata,
    pallas_projector_sinogram_unsupported_reason as _pallas_projector_sinogram_unsupported_reason,
    pallas_projector_traversal_metadata,
    pallas_projector_unsupported_reason as _pallas_projector_unsupported_reason,
    pallas_projector_variant_metadata,
)
from ._pallas_single_view import (
    BoundForwardProjectViewTPallas,
    bind_forward_project_view_T_pallas as _bind_forward_project_view_T_pallas,
    block_forward_project_view_T_pallas_state,
    forward_project_view_T_pallas as _forward_project_view_T_pallas,
    forward_project_view_T_pallas_with_state,
    prepare_forward_project_view_T_pallas_state,
)
from ._pallas_views import (
    BoundForwardProjectResidualSseTPallas,
    BoundForwardProjectViewsTPallas,
    backproject_view_T_pallas,
    bind_forward_project_residual_sse_T_pallas as _bind_forward_project_residual_sse_T_pallas,
    bind_forward_project_views_T_pallas as _bind_forward_project_views_T_pallas,
    block_forward_project_views_T_pallas_state,
    forward_project_loss_and_grad_T_pallas as _forward_project_loss_and_grad_T_pallas,
    forward_project_parallel_z_views_pallas,
    forward_project_residual_sse_T_pallas as _forward_project_residual_sse_T_pallas,
    forward_project_residual_sse_T_pallas_with_state,
    forward_project_views_T_pallas as _forward_project_views_T_pallas,
    forward_project_views_T_pallas_with_state,
    prepare_forward_project_views_T_pallas_state,
    sum_backproject_views_T_pallas,
)


@dataclass(frozen=True, slots=True)
class PallasProjectorOptions:
    """Runtime and kernel options shared by public Pallas projector entry points."""

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


def _opts(options: PallasProjectorOptions | None) -> PallasProjectorOptions:
    return PallasProjectorOptions() if options is None else options


def forward_project_view_T_pallas(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    *,
    options: PallasProjectorOptions | None = None,
) -> jnp.ndarray:
    """Forward project one view using the configured Pallas projector options."""
    opts = _opts(options)
    return _forward_project_view_T_pallas(
        T,
        grid,
        detector,
        volume,
        step_size=opts.step_size,
        n_steps=opts.n_steps,
        unroll=opts.unroll,
        gather_dtype=opts.gather_dtype,
        det_grid=opts.det_grid,
        interpret=opts.interpret,
        tile_shape=opts.tile_shape,
        num_warps=opts.num_warps,
        kernel_variant=opts.kernel_variant,
        layout_variant=opts.layout_variant,
        state_mode=opts.state_mode,
    )


def bind_forward_project_view_T_pallas(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    *,
    options: PallasProjectorOptions | None = None,
) -> BoundForwardProjectViewTPallas:
    """Bind fixed single-view Pallas traversal state using one options object."""
    opts = _opts(options)
    return _bind_forward_project_view_T_pallas(
        T,
        grid,
        detector,
        step_size=opts.step_size,
        n_steps=opts.n_steps,
        unroll=opts.unroll,
        gather_dtype=opts.gather_dtype,
        det_grid=opts.det_grid,
        interpret=opts.interpret,
        tile_shape=opts.tile_shape,
        num_warps=opts.num_warps,
        kernel_variant=opts.kernel_variant,
        layout_variant=opts.layout_variant,
        block_state=opts.block_state,
    )


def forward_project_views_T_pallas(
    T_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    *,
    options: PallasProjectorOptions | None = None,
) -> jnp.ndarray:
    """Forward project a stack of views using one configured Pallas options object."""
    opts = _opts(options)
    return _forward_project_views_T_pallas(
        T_stack,
        grid,
        detector,
        volume,
        step_size=opts.step_size,
        n_steps=opts.n_steps,
        unroll=opts.unroll,
        gather_dtype=opts.gather_dtype,
        det_grid=opts.det_grid,
        interpret=opts.interpret,
        tile_shape=opts.tile_shape,
        num_warps=opts.num_warps,
        kernel_variant=opts.kernel_variant,
        layout_variant=opts.layout_variant,
        state_mode=opts.state_mode,
    )


def bind_forward_project_views_T_pallas(
    T_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    *,
    options: PallasProjectorOptions | None = None,
) -> BoundForwardProjectViewsTPallas:
    """Bind fixed stack traversal state using one Pallas options object."""
    opts = _opts(options)
    return _bind_forward_project_views_T_pallas(
        T_stack,
        grid,
        detector,
        step_size=opts.step_size,
        n_steps=opts.n_steps,
        unroll=opts.unroll,
        gather_dtype=opts.gather_dtype,
        det_grid=opts.det_grid,
        interpret=opts.interpret,
        tile_shape=opts.tile_shape,
        num_warps=opts.num_warps,
        kernel_variant=opts.kernel_variant,
        layout_variant=opts.layout_variant,
        block_state=opts.block_state,
    )


def bind_forward_project_residual_sse_T_pallas(
    T_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    target: jnp.ndarray,
    *,
    options: PallasProjectorOptions | None = None,
) -> BoundForwardProjectResidualSseTPallas:
    """Bind fixed stack residual state using one Pallas options object."""
    opts = _opts(options)
    return _bind_forward_project_residual_sse_T_pallas(
        T_stack,
        grid,
        detector,
        target,
        step_size=opts.step_size,
        n_steps=opts.n_steps,
        unroll=opts.unroll,
        gather_dtype=opts.gather_dtype,
        det_grid=opts.det_grid,
        interpret=opts.interpret,
        tile_shape=opts.tile_shape,
        num_warps=opts.num_warps,
        kernel_variant=opts.kernel_variant,
        layout_variant=opts.layout_variant,
        block_state=opts.block_state,
    )


def forward_project_loss_and_grad_T_pallas(
    T_all: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    target: jnp.ndarray,
    *,
    weights: jnp.ndarray | None = None,
    compute_loss: bool = True,
    options: PallasProjectorOptions | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return Pallas projection loss/gradient using one shared options object."""
    opts = _opts(options)
    return _forward_project_loss_and_grad_T_pallas(
        T_all,
        grid,
        detector,
        volume,
        target,
        weights=weights,
        step_size=opts.step_size,
        n_steps=opts.n_steps,
        unroll=opts.unroll,
        gather_dtype=opts.gather_dtype,
        det_grid=opts.det_grid,
        interpret=opts.interpret,
        tile_shape=opts.tile_shape,
        num_warps=opts.num_warps,
        kernel_variant=opts.kernel_variant,
        layout_variant=opts.layout_variant,
        compute_loss=compute_loss,
    )


def forward_project_residual_sse_T_pallas(
    T_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    target: jnp.ndarray,
    *,
    options: PallasProjectorOptions | None = None,
) -> jnp.ndarray:
    """Return Pallas residual SSE using one shared options object."""
    opts = _opts(options)
    return _forward_project_residual_sse_T_pallas(
        T_stack,
        grid,
        detector,
        volume,
        target,
        step_size=opts.step_size,
        n_steps=opts.n_steps,
        unroll=opts.unroll,
        gather_dtype=opts.gather_dtype,
        det_grid=opts.det_grid,
        interpret=opts.interpret,
        tile_shape=opts.tile_shape,
        num_warps=opts.num_warps,
        kernel_variant=opts.kernel_variant,
        layout_variant=opts.layout_variant,
        state_mode=opts.state_mode,
    )


def pallas_projector_unsupported_reason(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    *,
    options: PallasProjectorOptions | None = None,
) -> str | None:
    """Return why single-view Pallas projection is unsupported, if it is."""
    opts = _opts(options)
    return _pallas_projector_unsupported_reason(
        T,
        grid,
        detector,
        volume,
        step_size=opts.step_size,
        n_steps=opts.n_steps,
        gather_dtype=opts.gather_dtype,
        det_grid=opts.det_grid,
        tile_shape=opts.tile_shape,
        num_warps=opts.num_warps,
        kernel_variant=opts.kernel_variant,
        layout_variant=opts.layout_variant,
        state_mode=opts.state_mode,
    )


def pallas_projector_sinogram_unsupported_reason(
    T_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    *,
    options: PallasProjectorOptions | None = None,
) -> str | None:
    """Return why batched Pallas projection is unsupported, if it is."""
    opts = _opts(options)
    return _pallas_projector_sinogram_unsupported_reason(
        T_stack,
        grid,
        detector,
        volume,
        step_size=opts.step_size,
        n_steps=opts.n_steps,
        gather_dtype=opts.gather_dtype,
        det_grid=opts.det_grid,
        tile_shape=opts.tile_shape,
        num_warps=opts.num_warps,
        kernel_variant=opts.kernel_variant,
        layout_variant=opts.layout_variant,
        state_mode=opts.state_mode,
    )


__all__ = [
    "BoundForwardProjectResidualSseTPallas",
    "BoundForwardProjectViewTPallas",
    "BoundForwardProjectViewsTPallas",
    "PallasForwardProjectorStackTraversalState",
    "PallasForwardProjectorTraversalState",
    "PallasProjectorOptions",
    "PallasProjectorTraversalMetadata",
    "PallasProjectorUnsupported",
    "backproject_view_T_pallas",
    "bind_forward_project_residual_sse_T_pallas",
    "bind_forward_project_view_T_pallas",
    "bind_forward_project_views_T_pallas",
    "block_forward_project_view_T_pallas_state",
    "block_forward_project_views_T_pallas_state",
    "forward_project_loss_and_grad_T_pallas",
    "forward_project_parallel_z_views_pallas",
    "forward_project_residual_sse_T_pallas",
    "forward_project_residual_sse_T_pallas_with_state",
    "forward_project_view_T_pallas",
    "forward_project_view_T_pallas_with_state",
    "forward_project_views_T_pallas",
    "forward_project_views_T_pallas_with_state",
    "pallas_projector_actual_sinogram_variant_metadata",
    "pallas_projector_actual_variant_metadata",
    "pallas_projector_sinogram_traversal_metadata",
    "pallas_projector_sinogram_unsupported_reason",
    "pallas_projector_traversal_metadata",
    "pallas_projector_unsupported_reason",
    "pallas_projector_variant_metadata",
    "prepare_forward_project_view_T_pallas_state",
    "prepare_forward_project_views_T_pallas_state",
    "sum_backproject_views_T_pallas",
]
