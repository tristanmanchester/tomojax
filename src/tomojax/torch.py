"""Optional PyTorch autograd bridge to TomoJAX's exact matched projectors.

Import this module explicitly; importing :mod:`tomojax` does not load PyTorch.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import jax
import jax.dlpack
import torch
from torch.utils import dlpack

from tomojax.recon import backproject, project

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

    ``operator(volume)`` maps ``(nx, ny, nz)`` to ``(views, v, u)``;
    ``operator.adjoint(projections)`` maps back. Inputs must be contiguous
    CUDA float32 tensors. The DLPack boundary shares storage in both directions;
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
        self.geometry = copy.deepcopy(geometry)
        self.grid = self.geometry.grid if grid is None else copy.deepcopy(grid)
        self.volume_shape = (self.grid.nx, self.grid.ny, self.grid.nz)
        self.projection_shape = (
            len(self.geometry.angles),
            self.geometry.detector.nv,
            self.geometry.detector.nu,
        )
        # Build through the public operators, preserving their geometry dispatch
        # and batching. The outer jit keeps their construction off the hot path.
        self._forward = jax.jit(lambda x: project(self.geometry, x, grid=self.grid))
        self._adjoint = jax.jit(lambda y: backproject(self.geometry, y, grid=self.grid))

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """Project a CUDA float32 volume; its gradient uses the exact transpose."""
        self._validate(volume, self.volume_shape)
        return _Projection.apply(volume, self._forward, self._adjoint)

    def adjoint(self, projections: torch.Tensor) -> torch.Tensor:
        """Backproject with the exact transpose; its gradient uses the projector."""
        self._validate(projections, self.projection_shape)
        return _Projection.apply(projections, self._adjoint, self._forward)

    @staticmethod
    def _validate(values: torch.Tensor, shape: tuple[int, int, int]) -> None:
        if not values.is_cuda or not values.is_floating_point() or values.element_size() != 4:
            raise ValueError("Projector inputs must be CUDA float32 tensors")
        if tuple(values.shape) != shape:
            raise ValueError(f"Projector expected shape {shape}, got {tuple(values.shape)}")
        if not values.is_contiguous():
            raise ValueError("Projector inputs must be contiguous; call tensor.contiguous()")
