"""First-order cone CUDA AD, retaining arrays rather than a ray-by-plane tape."""

from __future__ import annotations

from functools import partial

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np

from tomojax.core import cone
from tomojax.core.cone import cuda_module, forward_cuda
from tomojax.core.projector_autodiff import FirstOrderProjector


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


class _ConeProjector(FirstOrderProjector):
    def expand(self, coeff, volume, *packed: jax.Array):
        return forward_cuda(
            volume, coeff, self.grid, self.detector, packed=packed[0] if packed else None
        )

    def coefficient_jvp(self, coeff, volume, dcoeff, dvolume):
        result = cone.cone_pose_columns(
            volume, coeff, tangent=dcoeff[:, None, :12], grid=self.grid, detector=self.detector
        )[:, 1]
        if dvolume is not None:
            result = result + forward_cuda(dvolume, coeff, self.grid, self.detector)
        return result

    def coefficient_vjp(self, coeff, volume, cotangent):
        return _coefficient_vjp(coeff, volume, cotangent, self.grid, self.detector)

    def adjoint(self, coeff, cotangent):
        return cone.cone_backproject(cotangent, coeff, self.grid, self.detector, backend="cuda")


def differentiable_forward(coeff, volume, grid, detector, *, packed=None):
    """CUDA projection with first-order pose/volume derivatives and optional cache."""
    args = (coeff, volume) if packed is None else (coeff, volume, packed)
    return _ConeProjector(tuple(jax.typeof(arg) for arg in args), grid=grid, detector=detector)(
        *args
    )
