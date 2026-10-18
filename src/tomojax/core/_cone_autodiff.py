"""First-order cone CUDA AD, retaining arrays rather than a ray-by-plane tape."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
from jax.experimental.buffer_callback import buffer_callback
from jax.experimental.hijax import GradAccum, HiPrim, ShapedArray, Zero, instantiate_zeros
import jax.numpy as jnp
import numpy as np

from tomojax.core import cone
from tomojax.core.cone import cuda_module, forward_cuda

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Grid


def _launch_vjp(context, out, coeff, volume, cotangent, *, grid, detector):
    import cupy as cp

    with cone.xla_stream(context, out):
        target = cp.asarray(out)
        target.fill(0)
        cuda_module().get_function("cone_pose_vjp")(
            (coeff.shape[0], -(-detector.nu // 16), -(-detector.nv // 8)),
            (128,),
            (
                cp.asarray(coeff), cp.asarray(volume), cp.asarray(cotangent), target,
                np.int32(grid.nx), np.int32(grid.ny), np.int32(grid.nz),
                np.int32(detector.nu), np.int32(detector.nv),
                np.float32(grid.vx), np.float32(grid.vy), np.float32(grid.vz),
            ),
        )  # fmt: skip


def _coefficient_vjp(coeff, volume, cotangent, grid, detector):
    call = buffer_callback(
        partial(_launch_vjp, grid=grid, detector=detector),
        jax.ShapeDtypeStruct(coeff.shape, jnp.float32),
        vmap_method="sequential",
    )
    return call(coeff, volume, cotangent)


class _ConeTangent(HiPrim):
    grid: Grid
    detector: Detector
    has_dcoeff: bool
    has_dvolume: bool

    def __init__(self, avals, grid, detector, has_dcoeff, has_dvolume) -> None:
        self.in_avals = avals
        self.out_aval = ShapedArray((avals[0].shape[0], detector.nv, detector.nu), jnp.float32)
        self.params = {
            "grid": grid,
            "detector": detector,
            "has_dcoeff": has_dcoeff,
            "has_dvolume": has_dvolume,
        }
        super().__init__()

    def expand(self, coeff, volume, dcoeff, dvolume):
        result = jnp.zeros(self.out_aval.shape, jnp.float32)
        if self.has_dcoeff:
            result = cone.cone_pose_columns(
                volume, coeff, tangent=dcoeff[:, None, :12], grid=self.grid, detector=self.detector
            )[:, 1]
        if self.has_dvolume:
            result = result + forward_cuda(dvolume, coeff, self.grid, self.detector)
        return result

    def transpose(self, cotangent, coeff, volume, dcoeff_accum, dvolume_accum):
        if isinstance(coeff, GradAccum) or isinstance(volume, GradAccum):
            raise NotImplementedError("Cone CUDA derivatives support first-order AD only")
        cotangent = instantiate_zeros(cotangent)
        if self.has_dcoeff and isinstance(dcoeff_accum, GradAccum):
            dcoeff_accum.accum(_coefficient_vjp(coeff, volume, cotangent, self.grid, self.detector))
        if self.has_dvolume and isinstance(dvolume_accum, GradAccum):
            dvolume_accum.accum(
                cone.cone_backproject(cotangent, coeff, self.grid, self.detector, backend="cuda")
            )

    def batch_dim_rule(self, _axis_data, in_dims):
        return None if all(dim is None for dim in in_dims) else 0


class _ConeProjector(HiPrim):
    grid: Grid
    detector: Detector

    def __init__(self, avals, grid, detector) -> None:
        self.in_avals = avals
        self.out_aval = ShapedArray((avals[0].shape[0], detector.nv, detector.nu), jnp.float32)
        self.params = {"grid": grid, "detector": detector}
        super().__init__()

    def expand(self, coeff, volume, *packed: jax.Array):
        return forward_cuda(
            volume, coeff, self.grid, self.detector, packed=packed[0] if packed else None
        )

    def lin(self, nzs_in, coeff, volume, *packed: jax.Array):
        # A volume-only pullback needs only the geometry, not the primal volume
        # or its derived packed cache. Keep neither alive through reconstruction.
        return self(coeff, volume, *packed), (
            coeff,
            volume if nzs_in[0] else None,
            () if nzs_in[1] else None,
        )

    vjp_fwd = lin

    def vjp_bwd(self, residuals, cotangent, coeff_accum, volume_accum, *_packed_accum: GradAccum):
        coeff, volume, volume_active = residuals
        cotangent = instantiate_zeros(cotangent)
        if volume is not None and isinstance(coeff_accum, GradAccum):
            coeff_accum.accum(_coefficient_vjp(coeff, volume, cotangent, self.grid, self.detector))
        if volume_active is not None and isinstance(volume_accum, GradAccum):
            volume_accum.accum(
                cone.cone_backproject(cotangent, coeff, self.grid, self.detector, backend="cuda")
            )
        # Packed storage is a derived cache, never an independent physical input.

    def linearized(self, residuals, dcoeff, dvolume, *_dpacked: jax.Array):
        coeff, volume, _ = residuals
        has_dcoeff, has_dvolume = not isinstance(dcoeff, Zero), not isinstance(dvolume, Zero)
        # Dummy references avoid allocating zero volume tangents.
        args = (
            coeff,
            volume if has_dcoeff else coeff,
            dcoeff if has_dcoeff else coeff,
            dvolume if has_dvolume else coeff,
        )
        return _ConeTangent(
            tuple(jax.typeof(arg) for arg in args),
            self.grid,
            self.detector,
            has_dcoeff,
            has_dvolume,
        )(*args)

    def jvp(self, primals, tangents):
        return self(*primals), self.linearized((*primals[:2], None), *tangents)

    def transpose(self, cotangent, coeff, volume_accum, *_packed_accum: GradAccum):
        if isinstance(coeff, GradAccum):
            raise ValueError("Cone projection is linear only in volume; use jax.vjp for poses")
        if isinstance(volume_accum, GradAccum):
            volume_accum.accum(
                cone.cone_backproject(
                    instantiate_zeros(cotangent), coeff, self.grid, self.detector, backend="cuda"
                )
            )

    def batch_dim_rule(self, _axis_data, in_dims):
        return None if all(dim is None for dim in in_dims) else 0


def differentiable_forward(coeff, volume, grid, detector, *, packed=None):
    """CUDA projection with first-order pose/volume derivatives and optional cache."""
    args = (coeff, volume) if packed is None else (coeff, volume, packed)
    return _ConeProjector(tuple(jax.typeof(arg) for arg in args), grid, detector)(*args)
