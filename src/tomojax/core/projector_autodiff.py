"""First-order transformations for projectors linear in the volume.

Subclasses own their physical forward, combined coefficient/volume JVP,
coefficient VJP and exact adjoint. This module owns transformation bookkeeping;
only static geometry/options live in primitive parameters, never scan arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax
from jax.experimental.hijax import GradAccum, HiPrim, ShapedArray, Zero, instantiate_zeros
import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Hashable

    from tomojax.core.geometry.base import Detector, Grid


type Residuals = tuple[jax.Array, jax.Array | None, tuple[()] | None]
type Tangent = jax.Array | Zero


class FirstOrderProjector(HiPrim):
    """A physical projector, optionally accepting derived non-differentiable caches."""

    grid: Grid
    detector: Detector

    def __init__(
        self, avals: tuple[ShapedArray, ...], *, grid: Grid, detector: Detector, **options: Hashable
    ) -> None:
        self.in_avals = avals
        self.out_aval = ShapedArray((avals[0].shape[0], detector.nv, detector.nu), jnp.float32)
        self.params = {"grid": grid, "detector": detector, **options}
        super().__init__()

    def coefficient_jvp(
        self, coeff: jax.Array, volume: jax.Array, dcoeff: jax.Array, dvolume: jax.Array | None
    ) -> jax.Array:
        """Apply coefficient tangents, with an optional volume tangent in the same pass."""
        raise NotImplementedError

    def coefficient_vjp(
        self, coeff: jax.Array, volume: jax.Array, cotangent: jax.Array
    ) -> jax.Array:
        """Pull back the geometry coefficients without a ray-by-plane tape."""
        raise NotImplementedError

    def adjoint(self, coeff: jax.Array, cotangent: jax.Array) -> jax.Array:
        """Apply the exact volume transpose of ``expand``."""
        raise NotImplementedError

    def lin(
        self, nzs_in: tuple[bool, ...], coeff: jax.Array, volume: jax.Array, *cache: jax.Array
    ) -> tuple[jax.Array, Residuals]:
        """Retain only the primal arrays required by active physical derivatives."""
        # Volume-only reverse mode must not retain the primal image or its cache.
        return self(coeff, volume, *cache), (
            coeff,
            volume if nzs_in[0] else None,
            () if nzs_in[1] else None,
        )

    vjp_fwd = lin

    def vjp_bwd(
        self,
        residuals: Residuals,
        cotangent: Tangent,
        coeff_accum: object,
        volume_accum: object,
        *_cache_accum: object,
    ) -> None:
        """Accumulate physical pullbacks, never differentiating derived caches."""
        coeff, volume, volume_active = residuals
        cotangent = instantiate_zeros(cotangent)
        if volume is not None and isinstance(coeff_accum, GradAccum):
            coeff_accum.accum(self.coefficient_vjp(coeff, volume, cotangent))
        if volume_active is not None and isinstance(volume_accum, GradAccum):
            volume_accum.accum(self.adjoint(coeff, cotangent))

    def linearized(
        self, residuals: Residuals, dcoeff: Tangent, dvolume: Tangent, *_dcache: Tangent
    ) -> jax.Array:
        """Bind a linear tangent operator without allocating inactive zero images."""
        coeff, volume, _ = residuals
        has_dcoeff, has_dvolume = not isinstance(dcoeff, Zero), not isinstance(dvolume, Zero)
        # Dummy references allocate no zero images and are never read by a rule.
        args = (
            coeff,
            cast("jax.Array", volume) if has_dcoeff else coeff,
            cast("jax.Array", dcoeff) if has_dcoeff else coeff,
            cast("jax.Array", dvolume) if has_dvolume else coeff,
        )
        return _ProjectorTangent(
            tuple(jax.typeof(arg) for arg in args),
            self,
            has_dcoeff=has_dcoeff,
            has_dvolume=has_dvolume,
        )(*args)

    def jvp(
        self, primals: tuple[jax.Array, ...], tangents: tuple[Tangent, ...]
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluate the primal and its first-order tangent together."""
        return self(*primals), self.linearized((primals[0], primals[1], None), *tangents)

    def transpose(
        self,
        cotangent: Tangent,
        coeff: jax.Array | GradAccum,
        volume_accum: object,
        *_cache_accum: object,
    ) -> None:
        """Transpose only the projector's linear volume input."""
        if isinstance(coeff, GradAccum):
            raise ValueError("Projection is linear only in volume; use jax.vjp for poses")
        if isinstance(volume_accum, GradAccum):
            volume_accum.accum(self.adjoint(coeff, instantiate_zeros(cotangent)))

    def batch_dim_rule(self, _axis_data: object, in_dims: tuple[int | None, ...]) -> int | None:
        """Put mapped results on the leading axis, leaving unmapped calls alone."""
        return None if all(dim is None for dim in in_dims) else 0


class _ProjectorTangent(HiPrim):
    projector: FirstOrderProjector
    has_dcoeff: bool
    has_dvolume: bool

    def __init__(
        self,
        avals: tuple[ShapedArray, ...],
        projector: FirstOrderProjector,
        *,
        has_dcoeff: bool,
        has_dvolume: bool,
    ) -> None:
        self.in_avals = avals
        self.out_aval = projector.out_aval
        self.params = {"projector": projector, "has_dcoeff": has_dcoeff, "has_dvolume": has_dvolume}
        super().__init__()

    def expand(
        self, coeff: jax.Array, volume: jax.Array, dcoeff: jax.Array, dvolume: jax.Array
    ) -> jax.Array:
        if self.has_dcoeff:
            return self.projector.coefficient_jvp(
                coeff, volume, dcoeff, dvolume if self.has_dvolume else None
            )
        if self.has_dvolume:
            return self.projector.expand(coeff, dvolume)
        return jnp.zeros(self.out_aval.shape, jnp.float32)

    def transpose(
        self,
        cotangent: Tangent,
        coeff: jax.Array | GradAccum,
        volume: jax.Array | GradAccum,
        dcoeff_accum: object,
        dvolume_accum: object,
    ) -> None:
        if isinstance(coeff, GradAccum) or isinstance(volume, GradAccum):
            raise NotImplementedError("Projector derivatives support first-order AD only")
        self.projector.vjp_bwd(
            (coeff, volume if self.has_dcoeff else None, () if self.has_dvolume else None),
            cotangent,
            dcoeff_accum,
            dvolume_accum,
        )

    batch_dim_rule = FirstOrderProjector.batch_dim_rule
