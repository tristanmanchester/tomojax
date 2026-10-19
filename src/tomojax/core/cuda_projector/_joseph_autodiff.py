"""Physical rules for first-order transformations of the CUDA Joseph projector."""

from __future__ import annotations

import jax

from tomojax.core.projector_autodiff import FirstOrderProjector

from ._joseph import adjoint_cuda, forward_cuda
from ._joseph_derivatives import coefficient_vjp, projection_jvp


class _PlaneProjector(FirstOrderProjector):
    interpolation: str

    def expand(self, coeff, volume):
        return forward_cuda(
            coeff, volume, self.grid, self.detector, interpolation=self.interpolation
        )

    def coefficient_jvp(self, coeff, volume, dcoeff, dvolume):
        return projection_jvp(
            coeff,
            volume,
            dcoeff,
            dvolume,
            self.grid,
            self.detector,
            interpolation=self.interpolation,
        )

    def coefficient_vjp(self, coeff, volume, cotangent):
        return coefficient_vjp(
            coeff,
            volume,
            cotangent,
            self.grid,
            self.detector,
            interpolation=self.interpolation,
        )

    def adjoint(self, coeff, cotangent):
        return adjoint_cuda(
            coeff, cotangent, self.grid, self.detector, interpolation=self.interpolation
        )


def differentiable_forward(coeff, volume, grid, detector, *, interpolation="linear") -> jax.Array:
    """Apply CUDA projection with explicit first-order derivatives and batching."""
    return _PlaneProjector(
        (jax.typeof(coeff), jax.typeof(volume)),
        grid=grid,
        detector=detector,
        interpolation=interpolation,
    )(coeff, volume)
