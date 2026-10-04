"""First-order JAX transformations for the CUDA Joseph projector.

Use JAX 0.11's HiPrim interface so forward mode, reverse mode and batching have
explicit rules. Reverse mode retains the input volume and coefficients and
recomputes sample derivatives, rather than saving a ray-by-plane tape.
"""

from __future__ import annotations

import jax
from jax.experimental.hijax import GradAccum, HiPrim, ShapedArray, Zero, instantiate_zeros
import jax.numpy as jnp

from ._pallas_joseph import adjoint_pallas, forward_pallas
from ._pallas_joseph_derivatives import coefficient_vjp, projection_jvp


class _PlaneTangent(HiPrim):
    def __init__(self, avals, grid, detector, has_dcoeff, has_dvolume, interpolation) -> None:
        self.in_avals = avals
        self.out_aval = ShapedArray((avals[0].shape[0], detector.nv, detector.nu), jnp.float32)
        self.params = {
            "grid": grid,
            "detector": detector,
            "has_dcoeff": has_dcoeff,
            "has_dvolume": has_dvolume,
            "interpolation": interpolation,
        }
        super().__init__()

    def expand(self, coeff, volume, dcoeff, dvolume):
        if not self.has_dcoeff:
            if self.has_dvolume:
                return forward_pallas(
                    coeff, dvolume, self.grid, self.detector, interpolation=self.interpolation
                )
            return jnp.zeros(self.out_aval.shape, jnp.float32)
        return projection_jvp(
            coeff,
            volume,
            dcoeff,
            dvolume if self.has_dvolume else None,
            self.grid,
            self.detector,
            interpolation=self.interpolation,
        )

    def transpose(self, cotangent, coeff, volume, dcoeff_accum, dvolume_accum):
        if isinstance(coeff, GradAccum) or isinstance(volume, GradAccum):
            raise NotImplementedError("Joseph CUDA derivatives support first-order AD only")
        cotangent = instantiate_zeros(cotangent)
        if self.has_dcoeff and isinstance(dcoeff_accum, GradAccum):
            dcoeff_accum.accum(
                coefficient_vjp(
                    coeff,
                    volume,
                    cotangent,
                    self.grid,
                    self.detector,
                    interpolation=self.interpolation,
                )
            )
        if self.has_dvolume and isinstance(dvolume_accum, GradAccum):
            dvolume_accum.accum(
                adjoint_pallas(
                    coeff, cotangent, self.grid, self.detector, interpolation=self.interpolation
                )
            )

    def batch_dim_rule(self, _axis_data, in_dims):
        return None if all(dim is None for dim in in_dims) else 0


class _PlaneProjector(HiPrim):
    def __init__(self, coeff_type, volume_type, grid, detector, interpolation) -> None:
        self.in_avals = (coeff_type, volume_type)
        self.out_aval = ShapedArray((coeff_type.shape[0], detector.nv, detector.nu), jnp.float32)
        self.params = {"grid": grid, "detector": detector, "interpolation": interpolation}
        super().__init__()

    def expand(self, coeff, volume):
        return forward_pallas(
            coeff, volume, self.grid, self.detector, interpolation=self.interpolation
        )

    def vjp_fwd(self, _nzs_in, coeff, volume):
        return self(coeff, volume), (coeff, volume)

    def vjp_bwd_retval(self, residuals, cotangent):
        coeff, volume = residuals
        cotangent = instantiate_zeros(cotangent)
        return (
            coefficient_vjp(
                coeff, volume, cotangent, self.grid, self.detector, interpolation=self.interpolation
            ),
            adjoint_pallas(
                coeff, cotangent, self.grid, self.detector, interpolation=self.interpolation
            ),
        )

    def lin(self, _nzs_in, coeff, volume):
        return self(coeff, volume), (coeff, volume)

    def linearized(self, residuals, dcoeff, dvolume):
        coeff, volume = residuals
        has_dcoeff, has_dvolume = not isinstance(dcoeff, Zero), not isinstance(dvolume, Zero)
        # Dummy references avoid allocating full zero tangent arrays. The kernel
        # never loads a dummy input; these flags are compile-time parameters.
        args = (coeff, volume, dcoeff if has_dcoeff else coeff, dvolume if has_dvolume else volume)
        return _PlaneTangent(
            tuple(jax.typeof(arg) for arg in args),
            self.grid,
            self.detector,
            has_dcoeff,
            has_dvolume,
            self.interpolation,
        )(*args)

    def jvp(self, primals, tangents):
        return self(*primals), self.linearized(primals, *tangents)

    def transpose(self, cotangent, coeff, volume_accum):
        if isinstance(coeff, GradAccum):
            raise ValueError("Joseph projection is linear only in volume; use jax.vjp for poses")
        if isinstance(volume_accum, GradAccum):
            volume_accum.accum(
                adjoint_pallas(
                    coeff,
                    instantiate_zeros(cotangent),
                    self.grid,
                    self.detector,
                    interpolation=self.interpolation,
                )
            )

    def batch_dim_rule(self, _axis_data, in_dims):
        return None if all(dim is None for dim in in_dims) else 0


def differentiable_forward(coeff, volume, grid, detector, *, interpolation="linear") -> jax.Array:
    """Apply CUDA projection with explicit first-order derivatives and batching."""
    return _PlaneProjector(jax.typeof(coeff), jax.typeof(volume), grid, detector, interpolation)(
        coeff, volume
    )
