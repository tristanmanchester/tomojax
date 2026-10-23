"""Optional PyTorch autograd bridge to TomoJAX's exact matched projectors.

Import this module explicitly; importing :mod:`tomojax` does not load PyTorch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.dlpack
import torch
from torch.utils import dlpack

from tomojax.recon import Projector as _Projector

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomojax.geometry import Grid, ScanGeometry

__all__ = ["Projector"]


class _Projection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        values: torch.Tensor,
        forward: Callable[[jax.Array], jax.Array],
        transpose: Callable[[jax.Array], jax.Array],
    ) -> torch.Tensor:
        ctx.forward = forward
        ctx.transpose = transpose
        # Detach only the DLPack view, not the operation: backward below is the
        # exact other half of the same pair. No volume is saved for backward.
        array = jax.dlpack.from_dlpack(values.detach(), copy=False)
        result = forward(array)
        # The object protocol exchanges stream dependencies in both directions.
        # Do not use a legacy capsule or a device-wide synchronization here.
        return dlpack.from_dlpack(result)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        # Autograd can supply broadcast/strided gradients (e.g. y.sum()). These
        # alone need materializing; contiguous data cross DLPack without copies.
        result = _Projection.apply(gradient.contiguous(), ctx.transpose, ctx.forward)
        return result, None, None


class Projector(torch.nn.Module):
    """A fixed-geometry CUDA projector and its exact transpose, with autograd.

    ``operator(volume)`` maps ``(nz, ny, nx)`` to ``(views, v, u)``;
    ``operator.adjoint(projections)`` maps back. When ``grid.nz == detector.nv == 1``,
    2D inputs map ``(ny, nx)`` to ``(views, nu)`` and back, without copies.
    Inputs must be contiguous
    CUDA float32 tensors with 16-byte-aligned storage (standard allocations).
    Clone an unaligned offset view before passing it. The DLPack boundary
    shares storage in both directions;
    neither inputs nor outputs pass through the host. Noncontiguous gradients
    produced by PyTorch are made contiguous before applying the transpose.

    Geometry and grid are snapshotted at construction and are not trainable.
    Each direction compiles on first use, then reuses its compiled executable.
    Both directions support differentiation, including higher derivatives of
    compositions. Install PyTorch separately for your CUDA version; it is not
    required by the rest of TomoJAX.
    """

    def __init__(self, geometry: ScanGeometry, *, grid: Grid | None = None) -> None:
        super().__init__()
        self._operator = _Projector(geometry, grid=grid)
        self.geometry = self._operator.geometry
        self.grid = self._operator.grid
        self.volume_shape = self._operator.volume_shape
        self.projection_shape = self._operator.projection_shape

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """Project a CUDA float32 volume; its gradient uses the exact transpose."""
        self._validate(volume)
        return _Projection.apply(volume, self._operator, self._operator.adjoint)

    def adjoint(self, projections: torch.Tensor) -> torch.Tensor:
        """Backproject with the exact transpose; its gradient uses the projector."""
        self._validate(projections)
        return _Projection.apply(projections, self._operator.adjoint, self._operator)

    @staticmethod
    def _validate(values: torch.Tensor) -> None:
        if not values.is_cuda or not values.is_floating_point() or values.element_size() != 4:
            raise ValueError("Projector inputs must be CUDA float32 tensors")
        if not values.is_contiguous():
            raise ValueError("Projector inputs must be contiguous; call tensor.contiguous()")
