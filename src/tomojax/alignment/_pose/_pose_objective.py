from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from tomojax.alignment._config import AlignConfig
from tomojax.alignment._geometry.parametrizations import apply_pose_updates
from tomojax.alignment._objectives.fixed_volume import project_and_score_stack
from tomojax.alignment._objectives.loss_adapters import LossAdapter, loss_adapter
from tomojax.alignment._objectives.loss_specs import AlignmentLossSpec
from tomojax.core.geometry.base import Detector, Geometry, Grid
from tomojax.core.view_chunks import chunk_schedule

from ._coupled_objective import CoupledObjective, build_coupled_objective
from ._matching_model import JointRefinement
from ._pose_block import pose_block_solver
from ._pose_context import AlignmentRuntimeContext, _pose_objective_context
from ._pose_jacobian import PoseJacobianOptions, pose_prediction_and_columns


@dataclass(frozen=True)
class PoseObjectiveBundle:
    align_loss_jit: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    gn_update_all: Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]
    coupled: CoupledObjective | None = None


@dataclass(frozen=True)
class PoseSpec:
    """Only choices that change the compiled program, never measurement arrays."""

    grid: Grid
    detector: Detector
    jacobian: PoseJacobianOptions
    loss: AlignmentLossSpec
    backend: str
    chunk_size: int
    gn_damping: float
    # The per-view curvature prior enters the normal equations, not only the loss.
    smoothness_prior: bool


class PoseArrays(NamedTuple):
    """Per-scan numerical inputs; changing values can reuse a compiled program."""

    poses: jax.Array
    projections: jax.Array
    det_grid: tuple[jax.Array, jax.Array]
    active: jax.Array
    smoothness: jax.Array
    loss_mask: jax.Array | None
    # The views' cone-beam lab frames; None for a parallel beam.
    frames: jax.Array | None


def _apply_pose_smoothness_loss(
    pose_params: jnp.ndarray,
    loss: jnp.ndarray,
    smoothness_weights: jnp.ndarray,
) -> jnp.ndarray:
    if int(pose_params.shape[0]) < 3:
        return loss
    d2 = pose_params[:-2] - 2.0 * pose_params[1:-1] + pose_params[2:]
    return loss + jnp.sum((d2 * smoothness_weights) ** 2)


def _apply_pose_smoothness_gradient(
    pose_params: jnp.ndarray,
    total: jnp.ndarray,
    grad: jnp.ndarray,
    smoothness_weights: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if int(pose_params.shape[0]) < 3:
        return total, grad
    d2 = pose_params[:-2] - 2.0 * pose_params[1:-1] + pose_params[2:]
    total = total + jnp.sum((d2 * smoothness_weights) ** 2)
    ww = (smoothness_weights**2) * 2.0
    grad = grad.at[1:-1].add(-2.0 * d2 * ww)
    grad = grad.at[0:-2].add(1.0 * d2 * ww)
    grad = grad.at[2:].add(1.0 * d2 * ww)
    return total, grad


@partial(jax.jit, static_argnames=("spec",))
def align_loss(
    arrays: PoseArrays, pose_params: jax.Array, vol: jax.Array, *, spec: PoseSpec
) -> jax.Array:
    """The fixed-volume objective at ``pose_params``, including the pose smoothness term."""
    options = spec.jacobian
    loss_tot = project_and_score_stack(
        pose_stack=apply_pose_updates(
            arrays.poses, pose_params, translation_frame=options.translation_frame
        ),
        grid=spec.grid,
        detector=spec.detector,
        volume=vol,
        det_grid=arrays.det_grid,
        targets=arrays.projections,
        loss_adapter=loss_adapter(spec.loss, arrays.loss_mask),
        views_per_batch=spec.chunk_size,
        projector_unroll=options.unroll,
        checkpoint_projector=options.checkpoint,
        gather_dtype=options.gather_dtype,
        backend=spec.backend,
        # Gauss-Newton only evaluates this loss; its pose columns are separate.
        require_differentiable_projector=not options.cone_beam,
        ray_integrator=options.integrator,
        frames=arrays.frames,
    )
    return _apply_pose_smoothness_loss(pose_params, loss_tot, arrays.smoothness)


def gn_view_system(
    residual: jax.Array,
    cols: jax.Array,
    *,
    adapter: LossAdapter,
    active: jax.Array,
    damping: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One view's damped Gauss-Newton system, right-hand side and true objective."""
    current_loss, robust_weights = adapter.residual_model(residual)
    residual = residual * robust_weights
    cols = cols * robust_weights
    gradient = jnp.matmul(cols, residual, precision=jax.lax.Precision.HIGHEST)
    hessian = jnp.matmul(cols, cols.T, precision=jax.lax.Precision.HIGHEST)
    active = active.astype(hessian.dtype)
    inactive = jnp.float32(1.0) - active
    hessian_active = hessian * active[:, None] * active[None, :]
    system = hessian_active + jnp.float32(damping) * jnp.diag(active)
    return system + jnp.diag(inactive), -gradient * active, current_loss


@partial(jax.jit, static_argnames=("spec",))
def gn_update(
    arrays: PoseArrays, pose_params: jax.Array, vol: jax.Array, *, spec: PoseSpec
) -> tuple[jax.Array, jax.Array]:
    """The damped per-view Gauss-Newton step and the objective at ``pose_params``."""
    adapter = loss_adapter(spec.loss, arrays.loss_mask)
    n_views, b = int(pose_params.shape[0]), spec.chunk_size
    columns = partial(
        pose_prediction_and_columns,
        grid=spec.grid,
        detector=spec.detector,
        det_grid=arrays.det_grid,
        options=spec.jacobian,
    )

    def one_view(p5_i, t_nom_i, frame_i, y_i, w_i):
        residual, cols = columns(p5_i, t_nom_i, frame_i, vol, y_i, w_i)
        return gn_view_system(
            residual, cols, adapter=adapter, active=arrays.active, damping=spec.gn_damping
        )

    def body(carry, i):
        system_acc, rhs_acc, loss_acc = carry
        start, vmask, views = chunk_schedule(i, n_views=n_views, chunk_size=b)

        def take(a: jax.Array) -> jax.Array:
            return jax.lax.dynamic_slice_in_dim(a, start, b)

        y_chunk = take(arrays.projections)
        mask = None if arrays.loss_mask is None else take(arrays.loss_mask)
        frames = None if arrays.frames is None else take(arrays.frames)
        systems, rhs_values, loss_values = jax.vmap(one_view)(
            take(pose_params),
            take(arrays.poses),
            frames,
            y_chunk,
            adapter.gauss_newton_weights(y_chunk, mask),
        )
        system_acc = system_acc.at[views].add(systems * vmask[:, None, None])
        rhs_acc = rhs_acc.at[views].add(rhs_values * vmask[:, None])
        return (system_acc, rhs_acc, loss_acc + jnp.sum(loss_values * vmask)), None

    width = int(pose_params.shape[1])
    (systems, rhs, current_loss), _ = jax.lax.scan(
        body,
        (
            jnp.zeros((n_views, width, width), dtype=pose_params.dtype),
            jnp.zeros_like(pose_params),
            jnp.float32(0.0),
        ),
        jnp.arange((n_views + b - 1) // b, dtype=jnp.int32),
    )
    if spec.smoothness_prior:
        current_loss, prior_gradient = _apply_pose_smoothness_gradient(
            pose_params, current_loss, jnp.zeros_like(pose_params), arrays.smoothness
        )
        rhs = rhs - prior_gradient * arrays.active
        delta_all = pose_block_solver(systems, arrays.smoothness, has_smoothness=True)(rhs)
    else:
        current_loss = _apply_pose_smoothness_loss(pose_params, current_loss, arrays.smoothness)
        delta_all = jax.vmap(jnp.linalg.solve)(systems, rhs)
    return delta_all * arrays.active, current_loss


def _build_pose_objective_bundle(
    *,
    geometry: Geometry,  # noqa: ARG001
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    cfg: AlignConfig,
    n_views: int,
    active_mask: jnp.ndarray,
    runtime: AlignmentRuntimeContext,
    refinement: JointRefinement | None = None,
) -> PoseObjectiveBundle:
    ctx = _pose_objective_context(
        grid=grid,
        detector=detector,
        projections=projections,
        cfg=cfg,
        n_views=n_views,
        active_mask=active_mask,
        runtime=runtime,
    )
    arrays = PoseArrays(
        poses=ctx.pose_stack,
        projections=ctx.projections,
        det_grid=ctx.det_grid,
        active=ctx.active_mask,
        smoothness=ctx.smoothness_weights,
        loss_mask=ctx.loss_mask,
        frames=ctx.frames,
    )
    spec = PoseSpec(
        grid=grid,
        detector=detector,
        jacobian=replace(PoseJacobianOptions.from_config(cfg), cone_beam=ctx.cone is not None),
        loss=ctx.loss_adapter.spec,
        backend=str(cfg.backend),
        chunk_size=ctx.chunk_size,
        gn_damping=float(cfg.gn_damping),
        smoothness_prior=cfg.pose_model == "per_view" and bool(cfg.w_rot or cfg.w_trans),
    )
    return PoseObjectiveBundle(
        align_loss_jit=partial(align_loss, arrays, spec=spec),
        gn_update_all=partial(gn_update, arrays, spec=spec),
        coupled=build_coupled_objective(ctx, refinement=refinement)
        if cfg.gn_coupling == "joint"
        else None,
    )
