"""Joint free-voxel/pose Gauss-Newton with bounded pose-column storage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.projector import get_detector_grid_device

from ._coupled_linear import CoupledLinearResult
from ._coupled_program import CoupledArrays, CoupledSpec, run_loss, run_update
from ._pose_context import _PoseObjectiveContext
from ._pose_jacobian import PoseJacobianOptions

# Large acquisitions recompute per-view columns instead of retaining five
# complete sinograms. This caps the optional cache, not the acquisition size.
_POSE_CACHE_BYTES = 64 * 1024**2


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
    # Pallas exact integration accepts dynamic rigid poses. The sampled
    # reference keeps its matched explicit transpose and bounded view loop.
    backend = (
        "pallas" if cfg.projector_backend == "pallas" and jax.default_backend() == "gpu" else "jax"
    )
    if cfg.ray_integrator != "exact":
        backend = "jax"
    canonical = get_detector_grid_device(ctx.detector)
    if backend == "pallas" and not all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(ctx.det_grid, canonical, strict=True)
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
    cache_columns = ctx.n_views * ctx.nv * ctx.nu * 5 * 4 <= _POSE_CACHE_BYTES
    spec = CoupledSpec(
        grid=ctx.grid,
        detector=ctx.detector,
        backend=backend,
        jacobian=PoseJacobianOptions.from_config(cfg),
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
    )
    return CoupledObjective(
        partial(run_update, arrays, spec=spec),
        partial(run_loss, arrays, spec=spec),
        backend,
        cache_columns,
    )
