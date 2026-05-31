from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from tomojax.align._config import AlignConfig
from tomojax.align._geometry.parametrizations import se3_from_5d
from tomojax.align._objectives.fixed_volume import project_and_score_stack
from tomojax.core.geometry.base import Detector, Geometry, Grid
from tomojax.core.projector import forward_project_view_T

from ._pose_context import AlignmentRuntimeContext, _pose_objective_context, _PoseObjectiveContext


@dataclass(frozen=True)
class PoseObjectiveBundle:
    align_loss: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    align_loss_jit: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    loss_and_grad_manual: Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]
    gn_update_all: Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]


def _objective_chunk_schedule(
    ctx: _PoseObjectiveContext, i: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    i = jnp.asarray(i, dtype=jnp.int32)
    start = i * jnp.int32(ctx.chunk_size)
    remaining = jnp.maximum(0, jnp.int32(ctx.n_views) - start)
    valid = jnp.minimum(jnp.int32(ctx.chunk_size), remaining)
    shift = jnp.int32(ctx.chunk_size) - valid
    start_shifted = jnp.maximum(0, start - shift)
    idx = jnp.arange(ctx.chunk_size, dtype=jnp.int32)
    vmask = (idx >= (jnp.int32(ctx.chunk_size) - valid)).astype(jnp.float32)
    return start_shifted, vmask, start_shifted + idx


def _objective_apply_vol_mask(ctx: _PoseObjectiveContext, vol: jnp.ndarray) -> jnp.ndarray:
    return vol * ctx.volume_mask if ctx.volume_mask is not None else vol


def _objective_loss_mask_chunk(
    ctx: _PoseObjectiveContext, start_shifted: jnp.ndarray
) -> jnp.ndarray:
    if ctx.has_loss_mask:
        return jax.lax.dynamic_slice(
            ctx.loss_mask,
            (start_shifted, 0, 0),
            (ctx.chunk_size, ctx.nv, ctx.nu),
        )
    return ctx.empty_loss_mask_chunk


def _objective_loss_mask_arg(ctx: _PoseObjectiveContext, mask_i: jnp.ndarray) -> jnp.ndarray | None:
    return mask_i[None, ...] if ctx.has_loss_mask else None


def _apply_pose_smoothness_loss(
    params5: jnp.ndarray,
    loss: jnp.ndarray,
    smoothness_weights: jnp.ndarray,
) -> jnp.ndarray:
    if int(params5.shape[0]) < 3:
        return loss
    d2 = params5[:-2] - 2.0 * params5[1:-1] + params5[2:]
    return loss + jnp.sum((d2 * smoothness_weights) ** 2)


def _apply_pose_smoothness_gradient(
    params5: jnp.ndarray,
    total: jnp.ndarray,
    grad: jnp.ndarray,
    smoothness_weights: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if int(params5.shape[0]) < 3:
        return total, grad
    d2 = params5[:-2] - 2.0 * params5[1:-1] + params5[2:]
    total = total + jnp.sum((d2 * smoothness_weights) ** 2)
    ww = (smoothness_weights**2) * 2.0
    grad = grad.at[1:-1].add(-2.0 * d2 * ww)
    grad = grad.at[0:-2].add(1.0 * d2 * ww)
    grad = grad.at[2:].add(1.0 * d2 * ww)
    return total, grad


def _build_pose_align_loss(
    ctx: _PoseObjectiveContext,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    def align_loss(params5: jnp.ndarray, vol: jnp.ndarray) -> jnp.ndarray:
        t_aug = ctx.pose_stack @ jax.vmap(se3_from_5d)(params5)
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
            projector_backend=ctx.cfg.projector_backend,
            require_differentiable_projector=True,
        )
        return _apply_pose_smoothness_loss(params5, loss_tot, ctx.smoothness_weights)

    return align_loss


def _build_one_view_value_and_grad_batch(ctx: _PoseObjectiveContext) -> Callable[..., object]:
    def _one_view_loss(
        p5_i: jnp.ndarray,
        t_nom_i: jnp.ndarray,
        y_i: jnp.ndarray,
        masked_vol: jnp.ndarray,
        mask_i: jnp.ndarray,
        view_idx: jnp.ndarray,
    ) -> jnp.ndarray:
        t_i = t_nom_i @ se3_from_5d(p5_i)
        pred_i = forward_project_view_T(
            t_i,
            ctx.grid,
            ctx.detector,
            masked_vol,
            use_checkpoint=ctx.cfg.checkpoint_projector,
            unroll=int(ctx.cfg.projector_unroll),
            gather_dtype=ctx.cfg.gather_dtype,
            det_grid=ctx.det_grid,
        )
        view_indices = jnp.expand_dims(jnp.asarray(view_idx, dtype=jnp.int32), axis=0)
        lvec = ctx.per_view_loss_fn(
            pred_i[None, ...],
            y_i[None, ...],
            _objective_loss_mask_arg(ctx, mask_i),
            view_indices=view_indices,
        )
        return lvec[0]

    return jax.jit(
        jax.vmap(
            jax.value_and_grad(_one_view_loss),
            in_axes=(0, 0, 0, None, 0, 0),
        )
    )


def _build_manual_loss_and_grad(
    ctx: _PoseObjectiveContext,
    one_view_val_and_grad_batch: Callable[..., object],
) -> Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    def loss_and_grad_manual(
        params5: jnp.ndarray, vol: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        masked_vol = _objective_apply_vol_mask(ctx, vol)

        def body(
            carry: tuple[jnp.ndarray, jnp.ndarray], i: jnp.ndarray
        ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], None]:
            total, grad = carry
            start_shifted, vmask, view_idx_chunk = _objective_chunk_schedule(ctx, i)
            params_chunk = jax.lax.dynamic_slice(
                params5,
                (start_shifted, 0),
                (ctx.chunk_size, params5.shape[1]),
            )
            t_nom_chunk = jax.lax.dynamic_slice(
                ctx.pose_stack,
                (start_shifted, 0, 0),
                (ctx.chunk_size, 4, 4),
            )
            y_chunk = jax.lax.dynamic_slice(
                ctx.projections,
                (start_shifted, 0, 0),
                (ctx.chunk_size, ctx.nv, ctx.nu),
            )
            lvec, g_chunk = one_view_val_and_grad_batch(
                params_chunk,
                t_nom_chunk,
                y_chunk,
                masked_vol,
                _objective_loss_mask_chunk(ctx, start_shifted),
                view_idx_chunk,
            )
            total = total + jnp.sum(lvec * vmask)
            grad = grad.at[view_idx_chunk].add(g_chunk * vmask[:, None])
            return (total, grad), None

        init = (jnp.float32(0.0), jnp.zeros_like(params5))
        (total, grad), _ = jax.lax.scan(
            body,
            init,
            jnp.arange(ctx.num_chunks, dtype=jnp.int32),
        )
        return _apply_pose_smoothness_gradient(params5, total, grad, ctx.smoothness_weights)

    return jax.jit(loss_and_grad_manual)


def _build_gn_update_batch(ctx: _PoseObjectiveContext) -> Callable[..., object]:
    def _pred_flat(t_i: jnp.ndarray, masked_vol: jnp.ndarray) -> jnp.ndarray:
        return forward_project_view_T(
            t_i,
            ctx.grid,
            ctx.detector,
            masked_vol,
            use_checkpoint=ctx.cfg.checkpoint_projector,
            unroll=int(ctx.cfg.projector_unroll),
            gather_dtype=ctx.cfg.gather_dtype,
            det_grid=ctx.det_grid,
        ).ravel()

    def _gn_update_one(
        p5_i: jnp.ndarray,
        t_nom_i: jnp.ndarray,
        y_i: jnp.ndarray,
        vol: jnp.ndarray,
        w_i: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        def f(p5: jnp.ndarray) -> jnp.ndarray:
            t_i = t_nom_i @ se3_from_5d(p5)
            residual = _pred_flat(t_i, vol) - y_i.ravel()
            return w_i.ravel() * residual

        residual = f(p5_i)
        current_loss = jnp.float32(0.5) * jnp.vdot(residual, residual).real
        eye5 = jnp.eye(5, dtype=jnp.float32)

        def jvp_col(v: jnp.ndarray) -> jnp.ndarray:
            return jax.jvp(f, (p5_i,), (v,))[1]

        cols = jax.vmap(jvp_col)(eye5)
        gradient = cols @ residual
        hessian = cols @ cols.T
        lam = jnp.float32(ctx.cfg.gn_damping)
        active = ctx.active_mask.astype(hessian.dtype)
        inactive = jnp.float32(1.0) - active
        hessian_active = hessian * active[:, None] * active[None, :]
        system = hessian_active + lam * jnp.diag(active) + jnp.diag(inactive)
        rhs = -gradient * active
        delta = jnp.linalg.solve(system, rhs)
        return delta * active, current_loss

    return jax.jit(jax.vmap(_gn_update_one, in_axes=(0, 0, 0, None, 0)))


def _build_gn_update_all(
    ctx: _PoseObjectiveContext,
    gn_update_batch: Callable[..., object],
) -> Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    def _ls_weight_chunk(y_chunk: jnp.ndarray, mask_chunk: jnp.ndarray) -> jnp.ndarray:
        return ctx.loss_adapter.gauss_newton_weights(
            y_chunk,
            mask_chunk if ctx.has_loss_mask else None,
        )

    def gn_update_all(params5: jnp.ndarray, vol: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        masked_vol = _objective_apply_vol_mask(ctx, vol)

        def body(
            carry: tuple[jnp.ndarray, jnp.ndarray], i: jnp.ndarray
        ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], None]:
            delta_acc, loss_acc = carry
            start_shifted, vmask, view_idx_chunk = _objective_chunk_schedule(ctx, i)
            params_chunk = jax.lax.dynamic_slice(
                params5,
                (start_shifted, 0),
                (ctx.chunk_size, params5.shape[1]),
            )
            t_chunk = jax.lax.dynamic_slice(
                ctx.pose_stack,
                (start_shifted, 0, 0),
                (ctx.chunk_size, 4, 4),
            )
            y_chunk = jax.lax.dynamic_slice(
                ctx.projections,
                (start_shifted, 0, 0),
                (ctx.chunk_size, ctx.nv, ctx.nu),
            )
            dp_values, loss_values = gn_update_batch(
                params_chunk,
                t_chunk,
                y_chunk,
                masked_vol,
                _ls_weight_chunk(y_chunk, _objective_loss_mask_chunk(ctx, start_shifted)),
            )
            delta_acc = delta_acc.at[view_idx_chunk].add(dp_values * vmask[:, None])
            loss_acc = loss_acc + jnp.sum(loss_values * vmask)
            return (delta_acc, loss_acc), None

        delta0 = jnp.zeros_like(params5)
        (delta_all, current_loss), _ = jax.lax.scan(
            body,
            (delta0, jnp.float32(0.0)),
            jnp.arange(ctx.num_chunks, dtype=jnp.int32),
        )
        current_loss = _apply_pose_smoothness_loss(params5, current_loss, ctx.smoothness_weights)
        return delta_all, current_loss

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
    )
