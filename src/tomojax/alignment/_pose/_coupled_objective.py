"""Joint free-voxel/pose Gauss-Newton with bounded pose-column storage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.alignment._model.dofs import POSE_WIDTH
from tomojax.core.projector import get_detector_grid_device

from ._coupled_linear import CoupledLinearResult
from ._coupled_program import CoupledArrays, CoupledSpec, run_loss, run_update
from ._pose_context import _PoseObjectiveContext
from ._pose_jacobian import PoseJacobianOptions

# Recomputing a view's five Jacobian columns costs a derivative projection,
# and the joint solve applies them in every conjugate-gradient iteration, so
# cache them whenever five sinograms fit in this share of free device memory.
# Larger acquisitions recompute them per view instead.
_POSE_CACHE_FRACTION = 0.25
# Fallback cap when free device memory cannot be queried.
_POSE_CACHE_BYTES = 512 * 1024**2


def _pose_cache_limit() -> int:
    from tomojax.backends import device_free_memory_bytes

    if _POSE_CACHE_BYTES <= 0:
        return 0
    free = device_free_memory_bytes() if jax.default_backend() == "gpu" else None
    return _POSE_CACHE_BYTES if free is None else int(_POSE_CACHE_FRACTION * free)


@dataclass(frozen=True)
class CoupledObjective:
    update: Callable[[jax.Array, jax.Array], CoupledLinearResult]
    loss: Callable[[jax.Array, jax.Array], jax.Array]
    projector_backend: str
    pose_columns_cached: bool


def build_coupled_objective(ctx: _PoseObjectiveContext) -> CoupledObjective:
    cfg = ctx.cfg
    if not ctx.loss_adapter.supports_gauss_newton:
        raise ValueError("joint GN requires a least-squares alignment loss")
    if cfg.gather_dtype not in {"fp32", "float32"}:
        raise ValueError("joint GN requires gather_dtype='fp32' for matched linear operators")
    # Pallas exact and Joseph integration accept dynamic rigid poses. The
    # sampled reference keeps its matched explicit transpose and bounded view loop.
    backend = (
        "pallas" if cfg.projector_backend == "pallas" and jax.default_backend() == "gpu" else "jax"
    )
    if cfg.ray_integrator not in {"exact", "joseph", "joseph_cubic"}:
        backend = "jax"
    if ctx.beam is not None:
        # Cone beams: the CUDA kernels where available ("pallas" names the CUDA path).
        from tomojax.core.cone import use_cuda_cone

        backend = "pallas" if cfg.projector_backend != "jax" and use_cuda_cone() else "jax"
    canonical = get_detector_grid_device(ctx.detector)
    if (
        backend == "pallas"
        and cfg.ray_integrator == "exact"
        and not all(
            np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(ctx.det_grid, canonical, strict=True)
        )
    ):
        backend = "jax"
    arrays = CoupledArrays(
        poses=ctx.pose_stack,
        projections=ctx.projections,
        weights=ctx.loss_adapter.gauss_newton_weights(
            ctx.projections, ctx.loss_mask if ctx.has_loss_mask else None
        ),
        mask=ctx.volume_mask if ctx.volume_mask is not None else jnp.float32(1),
        active=ctx.active_mask.astype(jnp.float32),
        smoothness=ctx.smoothness_weights,
        det_grid=ctx.det_grid,
    )
    cache_columns = ctx.n_views * ctx.nv * ctx.nu * POSE_WIDTH * 4 <= _pose_cache_limit()
    spec = CoupledSpec(
        grid=ctx.grid,
        detector=ctx.detector,
        backend=backend,
        jacobian=replace(PoseJacobianOptions.from_config(cfg), beam=ctx.beam),
        cache_columns=cache_columns,
        regulariser=cfg.regulariser,
        huber_delta=float(cfg.huber_delta),
        lambda_tv=float(cfg.lambda_tv),
        recon_positivity=bool(cfg.recon_positivity),
        gn_volume_damping=float(cfg.gn_volume_damping),
        gn_damping=float(cfg.gn_damping),
        gn_joint_solver=cfg.gn_joint_solver,
        gn_joint_rtol=float(cfg.gn_joint_rtol),
        gn_joint_iters=int(cfg.gn_joint_iters),
        has_smoothness=bool(cfg.w_rot or cfg.w_trans),
        beam=ctx.beam,
    )
    return CoupledObjective(
        partial(run_update, arrays, spec=spec),
        partial(run_loss, arrays, spec=spec),
        backend,
        cache_columns,
    )
