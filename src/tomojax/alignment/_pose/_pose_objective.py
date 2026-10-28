from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from tomojax.alignment._config import AlignConfig
from tomojax.alignment._geometry.parametrizations import apply_pose_update, apply_pose_updates
from tomojax.alignment._objectives.fixed_volume import project_and_score_stack
from tomojax.core.geometry.base import Detector, Geometry, Grid
from tomojax.core.projector import forward_project_view_T
from tomojax.core.view_chunks import chunk_schedule

from ._coupled_objective import CoupledObjective, build_coupled_objective
from ._matching_model import JointRefinement
from ._pose_block import pose_block_solver
from ._pose_context import AlignmentRuntimeContext, _pose_objective_context, _PoseObjectiveContext
from ._pose_jacobian import build_pose_prediction_and_columns


@dataclass(frozen=True)
class PoseObjectiveBundle:
    align_loss: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    align_loss_jit: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    loss_and_grad_manual: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ]
    gn_update_all: Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]
    coupled: CoupledObjective | None = None


def _objective_apply_vol_mask(ctx: _PoseObjectiveContext, vol: jnp.ndarray) -> jnp.ndarray:
    return vol * ctx.volume_mask if ctx.volume_mask is not None else vol


def _objective_loss_mask_chunk(
    ctx: _PoseObjectiveContext, start_shifted: jnp.ndarray
) -> jnp.ndarray:
    if ctx.has_loss_mask:
        return jax.lax.dynamic_slice_in_dim(ctx.loss_mask, start_shifted, ctx.chunk_size)
    return ctx.empty_loss_mask_chunk


def _objective_loss_mask_arg(ctx: _PoseObjectiveContext, mask_i: jnp.ndarray) -> jnp.ndarray | None:
    return mask_i[None, ...] if ctx.has_loss_mask else None


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


def _build_pose_align_loss(
    ctx: _PoseObjectiveContext,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    def align_loss(
        pose_params: jnp.ndarray,
        vol: jnp.ndarray,
        loss_rng_key: jnp.ndarray,
    ) -> jnp.ndarray:
        t_aug = apply_pose_updates(
            ctx.pose_stack, pose_params, translation_frame=ctx.cfg.pose_translation_frame
        )
        loss_tot = project_and_score_stack(
            pose_stack=t_aug,
            grid=ctx.grid,
            detector=ctx.detector,
            volume=_objective_apply_vol_mask(ctx, vol),
            det_grid=ctx.det_grid,
            targets=ctx.projections,
            loss_adapter=ctx.loss_adapter,
            views_per_batch=ctx.chunk_size,
            projector_unroll=int(ctx.cfg.projector_unroll),
            checkpoint_projector=ctx.cfg.checkpoint_projector,
            gather_dtype=ctx.cfg.gather_dtype,
            view_indices=jnp.arange(ctx.n_views, dtype=jnp.int32),
            backend=ctx.cfg.backend,
            # Gauss-Newton only evaluates this loss; its pose columns are separate.
            require_differentiable_projector=ctx.cone is None or ctx.cfg.opt_method != "gn",
            loss_rng_key=loss_rng_key,
            ray_integrator=ctx.cfg.ray_integrator,
            frames=ctx.frames,
        )
        return _apply_pose_smoothness_loss(pose_params, loss_tot, ctx.smoothness_weights)

    return align_loss


def _objective_frames_chunk(ctx: _PoseObjectiveContext, start: jnp.ndarray) -> jnp.ndarray | None:
    """The chunk's cone-beam lab frames, or None for a parallel beam."""
    if ctx.frames is None:
        return None
    return jax.lax.dynamic_slice_in_dim(ctx.frames, start, ctx.chunk_size)


def _build_one_view_value_and_grad_batch(ctx: _PoseObjectiveContext) -> Callable[..., object]:
    def _one_view_loss(
        p5_i: jnp.ndarray,
        t_nom_i: jnp.ndarray,
        frame_i: jnp.ndarray | None,
        y_i: jnp.ndarray,
        masked_vol: jnp.ndarray,
        mask_i: jnp.ndarray,
        view_idx: jnp.ndarray,
        loss_rng_key: jnp.ndarray,
    ) -> jnp.ndarray:
        t_i = apply_pose_update(t_nom_i, p5_i, translation_frame=ctx.cfg.pose_translation_frame)
        pred_i = forward_project_view_T(
            t_i,
            ctx.grid,
            ctx.detector,
            masked_vol,
            use_checkpoint=ctx.cfg.checkpoint_projector,
            unroll=int(ctx.cfg.projector_unroll),
            gather_dtype=ctx.cfg.gather_dtype,
            det_grid=ctx.det_grid,
            ray_integrator=ctx.cfg.ray_integrator,
            frames=frame_i,
        )
        view_indices = jnp.expand_dims(jnp.asarray(view_idx, dtype=jnp.int32), axis=0)
        lvec = ctx.per_view_loss_fn(
            pred_i[None, ...],
            y_i[None, ...],
            _objective_loss_mask_arg(ctx, mask_i),
            view_indices=view_indices,
            rng_key=loss_rng_key,
        )
        return lvec[0]

    return jax.jit(
        jax.vmap(
            jax.value_and_grad(_one_view_loss),
            in_axes=(0, 0, 0, 0, None, 0, 0, None),
        )
    )


def _build_manual_loss_and_grad(
    ctx: _PoseObjectiveContext,
    one_view_val_and_grad_batch: Callable[..., object],
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    def loss_and_grad_manual(
        pose_params: jnp.ndarray, vol: jnp.ndarray, loss_rng_key: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        masked_vol = _objective_apply_vol_mask(ctx, vol)

        def body(
            carry: tuple[jnp.ndarray, jnp.ndarray], i: jnp.ndarray
        ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], None]:
            total, grad = carry
            start_shifted, vmask, view_idx_chunk = chunk_schedule(
                i, n_views=ctx.n_views, chunk_size=ctx.chunk_size
            )
            params_chunk = jax.lax.dynamic_slice_in_dim(pose_params, start_shifted, ctx.chunk_size)
            t_nom_chunk = jax.lax.dynamic_slice_in_dim(
                ctx.pose_stack, start_shifted, ctx.chunk_size
            )
            y_chunk = jax.lax.dynamic_slice_in_dim(ctx.projections, start_shifted, ctx.chunk_size)
            lvec, g_chunk = one_view_val_and_grad_batch(
                params_chunk,
                t_nom_chunk,
                _objective_frames_chunk(ctx, start_shifted),
                y_chunk,
                masked_vol,
                _objective_loss_mask_chunk(ctx, start_shifted),
                view_idx_chunk,
                loss_rng_key,
            )
            total = total + jnp.sum(lvec * vmask)
            grad = grad.at[view_idx_chunk].add(g_chunk * vmask[:, None])
            return (total, grad), None

        init = (jnp.float32(0.0), jnp.zeros_like(pose_params))
        (total, grad), _ = jax.lax.scan(
            body,
            init,
            jnp.arange(ctx.num_chunks, dtype=jnp.int32),
        )
        return _apply_pose_smoothness_gradient(pose_params, total, grad, ctx.smoothness_weights)

    return jax.jit(loss_and_grad_manual)


def _build_gn_update_batch(ctx: _PoseObjectiveContext) -> Callable[..., object]:
    predict_and_columns = build_pose_prediction_and_columns(ctx)

    def _gn_update_one(p5_i, t_nom_i, frame_i, y_i, vol, w_i):
        residual, cols = predict_and_columns(p5_i, t_nom_i, frame_i, vol, y_i, w_i)
        current_loss, robust_weights = ctx.loss_adapter.residual_model(residual)
        residual = residual * robust_weights
        cols = cols * robust_weights
        gradient = jnp.matmul(cols, residual, precision=jax.lax.Precision.HIGHEST)
        hessian = jnp.matmul(cols, cols.T, precision=jax.lax.Precision.HIGHEST)
        lam = jnp.float32(ctx.cfg.gn_damping)
        active = ctx.active_mask.astype(hessian.dtype)
        inactive = jnp.float32(1.0) - active
        hessian_active = hessian * active[:, None] * active[None, :]
        system = hessian_active + lam * jnp.diag(active) + jnp.diag(inactive)
        return system, -gradient * active, current_loss

    return jax.jit(jax.vmap(_gn_update_one, in_axes=(0, 0, 0, 0, None, 0)))


def _build_gn_update_all(
    ctx: _PoseObjectiveContext,
    gn_update_batch: Callable[..., object],
) -> Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    def _ls_weight_chunk(y_chunk: jnp.ndarray, mask_chunk: jnp.ndarray) -> jnp.ndarray:
        return ctx.loss_adapter.gauss_newton_weights(
            y_chunk,
            mask_chunk if ctx.has_loss_mask else None,
        )

    def gn_update_all(
        pose_params: jnp.ndarray, vol: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        masked_vol = _objective_apply_vol_mask(ctx, vol)

        def body(
            carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], i: jnp.ndarray
        ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], None]:
            system_acc, rhs_acc, loss_acc = carry
            start_shifted, vmask, view_idx_chunk = chunk_schedule(
                i, n_views=ctx.n_views, chunk_size=ctx.chunk_size
            )
            params_chunk = jax.lax.dynamic_slice_in_dim(pose_params, start_shifted, ctx.chunk_size)
            t_chunk = jax.lax.dynamic_slice_in_dim(ctx.pose_stack, start_shifted, ctx.chunk_size)
            y_chunk = jax.lax.dynamic_slice_in_dim(ctx.projections, start_shifted, ctx.chunk_size)
            systems, rhs_values, loss_values = gn_update_batch(
                params_chunk,
                t_chunk,
                _objective_frames_chunk(ctx, start_shifted),
                y_chunk,
                masked_vol,
                _ls_weight_chunk(y_chunk, _objective_loss_mask_chunk(ctx, start_shifted)),
            )
            system_acc = system_acc.at[view_idx_chunk].add(systems * vmask[:, None, None])
            rhs_acc = rhs_acc.at[view_idx_chunk].add(rhs_values * vmask[:, None])
            loss_acc = loss_acc + jnp.sum(loss_values * vmask)
            return (system_acc, rhs_acc, loss_acc), None

        width = int(pose_params.shape[1])
        (systems, rhs, current_loss), _ = jax.lax.scan(
            body,
            (
                jnp.zeros((ctx.n_views, width, width), dtype=pose_params.dtype),
                jnp.zeros_like(pose_params),
                jnp.float32(0.0),
            ),
            jnp.arange(ctx.num_chunks, dtype=jnp.int32),
        )
        if ctx.cfg.pose_model == "per_view" and (ctx.cfg.w_rot or ctx.cfg.w_trans):
            # The curvature prior is part of the objective the step scores, so
            # its gradient and (view-coupling) Hessian belong in the normal
            # equations rather than only in the loss.
            current_loss, prior_gradient = _apply_pose_smoothness_gradient(
                pose_params, current_loss, jnp.zeros_like(pose_params), ctx.smoothness_weights
            )
            rhs = rhs - prior_gradient * ctx.active_mask
            delta_all = pose_block_solver(systems, ctx.smoothness_weights, has_smoothness=True)(rhs)
        else:
            current_loss = _apply_pose_smoothness_loss(
                pose_params, current_loss, ctx.smoothness_weights
            )
            delta_all = jax.vmap(jnp.linalg.solve)(systems, rhs)
        return delta_all * ctx.active_mask, current_loss

    return jax.jit(gn_update_all)


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
    align_loss = _build_pose_align_loss(ctx)
    one_view_val_and_grad_batch = _build_one_view_value_and_grad_batch(ctx)
    loss_and_grad_manual_jit = _build_manual_loss_and_grad(ctx, one_view_val_and_grad_batch)
    gn_update_batch = _build_gn_update_batch(ctx)
    return PoseObjectiveBundle(
        align_loss=align_loss,
        align_loss_jit=jax.jit(align_loss),
        loss_and_grad_manual=loss_and_grad_manual_jit,
        gn_update_all=_build_gn_update_all(ctx, gn_update_batch),
        coupled=build_coupled_objective(ctx, refinement=refinement)
        if cfg.gn_coupling == "joint"
        else None,
    )
