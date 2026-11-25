"""First-order cone CUDA AD, retaining arrays rather than a ray-by-plane tape."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np

from tomojax.core import cone
from tomojax.core._cone_runtime import cuda_module
from tomojax.core.cone import adjoint_cuda, forward_cuda, pose_columns_cuda
from tomojax.core.projector_autodiff import FirstOrderProjector
from tomojax.core.volume_layout import swap_private_axes

if TYPE_CHECKING:
    from tomojax.core._cuda_types import CallbackBuffer, CallbackContext

    from .geometry.base import Detector, Grid


def _launch_vjp(
    context: CallbackContext,
    out: CallbackBuffer,
    coeff: CallbackBuffer,
    volume: CallbackBuffer,
    cotangent: CallbackBuffer,
    *,
    grid: Grid,
    detector: Detector,
    z_fast: bool,
) -> None:
    import cupy as cp

    with cone.xla_stream(context, out):
        target = cp.asarray(out)
        target.fill(0)
        module = cuda_module(z_fast=True) if z_fast else cuda_module()
        module.get_function("cone_pose_vjp")(
            (coeff.shape[0], -(-detector.nu // 16), -(-detector.nv // 8)),
            (128,),
            (
                cp.asarray(coeff), cp.asarray(volume), cp.asarray(cotangent), target,
                np.int32(grid.nx), np.int32(grid.ny), np.int32(grid.nz),
                np.int32(detector.nu), np.int32(detector.nv),
                np.float32(grid.vx), np.float32(grid.vy), np.float32(grid.vz),
            ),
        )  # fmt: skip


def _coefficient_vjp(
    coeff: jax.Array,
    volume: jax.Array,
    cotangent: jax.Array,
    grid: Grid,
    detector: Detector,
    z_fast: Literal[True, False],
) -> jax.Array:
    call = buffer_callback(
        partial(_launch_vjp, grid=grid, detector=detector, z_fast=z_fast),
        jax.ShapeDtypeStruct(coeff.shape, jnp.float32),
        vmap_method="sequential",
    )
    return call(coeff, volume, cotangent)


class _ConeProjector(FirstOrderProjector):
    z_fast: bool

    def expand(self, coeff: jax.Array, volume: jax.Array) -> jax.Array:
        return forward_cuda(volume, coeff, self.grid, self.detector, z_fast=self.z_fast)

    def coefficient_jvp(
        self, coeff: jax.Array, volume: jax.Array, dcoeff: jax.Array, dvolume: jax.Array | None
    ) -> jax.Array:
        result = pose_columns_cuda(
            volume,
            coeff,
            tangent=dcoeff[:, None, :12],
            grid=self.grid,
            detector=self.detector,
            z_fast=self.z_fast,
        )[:, 1]
        if dvolume is not None:
            result = result + forward_cuda(
                dvolume, coeff, self.grid, self.detector, z_fast=self.z_fast
            )
        return result

    def coefficient_vjp(
        self, coeff: jax.Array, volume: jax.Array, cotangent: jax.Array
    ) -> jax.Array:
        return _coefficient_vjp(coeff, volume, cotangent, self.grid, self.detector, self.z_fast)

    def adjoint(self, coeff: jax.Array, cotangent: jax.Array) -> jax.Array:
        shape = (self.grid.nx, self.grid.ny, self.grid.nz)
        initial = jnp.zeros(shape if self.z_fast else shape[::-1], jnp.float32)
        return adjoint_cuda(cotangent, coeff, self.grid, self.detector, initial, z_fast=self.z_fast)


def differentiable_forward(
    coeff: jax.Array, volume: jax.Array, grid: Grid, detector: Detector, *, z_fast: bool = False
) -> jax.Array:
    """CUDA projection with first-order pose/volume derivatives."""

    def on_cuda(c: jax.Array, v: jax.Array) -> jax.Array:
        args = (c, v)
        return _ConeProjector(
            tuple(jax.typeof(arg) for arg in args), grid=grid, detector=detector, z_fast=z_fast
        )(*args)

    def on_jax(c: jax.Array, v: jax.Array) -> jax.Array:
        v = swap_private_axes(v, z_fast=z_fast)
        return cone.cone_project(v, c, grid, detector, backend="jax")

    return jax.lax.platform_dependent(coeff, volume, cuda=on_cuda, default=on_jax)
