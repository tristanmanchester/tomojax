from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import logging
import math
import re
import time
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core import format_duration, progress_iter
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.projector import forward_project_view_T, get_detector_grid_device
from tomojax.core.validation import (
    validate_grid,
    validate_optional_same_shape,
    validate_pose_stack,
    validate_projection_stack,
    validate_volume,
)
from tomojax.geometry import cylindrical_mask_xy

from ._config import AlignConfig, _active_dof_mask_for_cfg, _active_dofs_for_cfg
from ._geometry.parametrizations import se3_from_5d
from ._model.dofs import bounds_vectors
from ._model.gauge import (
    GaugeFixMode,
    active_gauge_dofs,
    apply_alignment_gauge,
    gauge_stats_to_python,
    normalize_gauge_fix,
    validate_alignment_gauge_feasible,
)
from ._model.motion_models import (
    build_pose_motion_model,
    expand_motion_coefficients,
    fit_motion_coefficients,
    scan_coordinate_from_geometry,
)
from ._objectives.fixed_volume import (
    ObjectiveProvenance,
    alignment_projector_backend_provenance,
    project_and_score_stack,
)
from ._objectives.loss_adapters import LossAdapter, build_loss_adapter
from ._objectives.loss_specs import (
    loss_is_within_relative_tolerance,
    loss_spec_name,
    resolve_loss_for_level,
)
from ._observer import (
    ObserverAction,
    ObserverCallback,
    OuterStat,
    _normalize_observer_action,
    adapt_observer_callback,
)
from ._profiles import profile_policy_from_config
from ._reconstruction_stage import _run_reconstruction_step
from ._results import AlignCheckpointCallback, AlignInfo, AlignResumeState, _set_float_stat
from .optimizers import PoseLbfgsConfig, PoseOptimizationContext, run_pose_lbfgs

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Geometry, Grid


def _should_prefer_gn_candidate(
    loss_before: float,
    current_loss: float,
    candidate_loss: float,
    rel_tol: float,
) -> bool:
    """Accept tolerated GN candidates only when they improve the current best step."""
    candidate_ok = candidate_loss < loss_before or loss_is_within_relative_tolerance(
        loss_before, candidate_loss, rel_tol
    )
    return candidate_ok and candidate_loss < current_loss


def _second_difference_gram(n: int) -> jnp.ndarray:
    if n < 3:
        return jnp.zeros((n, n), dtype=jnp.float32)
    d2 = jnp.zeros((n - 2, n), dtype=jnp.float32)
    rows = jnp.arange(n - 2, dtype=jnp.int32)
    d2 = d2.at[rows, rows].set(1.0)
    d2 = d2.at[rows, rows + 1].set(-2.0)
    d2 = d2.at[rows, rows + 2].set(1.0)
    return d2.T @ d2


def _smooth_gn_candidate(
    params5: jnp.ndarray,
    smoothness_gram: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Project a per-view GN candidate through the quadratic curvature prior."""
    n_views = int(params5.shape[0])
    if n_views < 3:
        return params5

    eye = jnp.eye(n_views, dtype=jnp.float32)

    def solve_one_dim(rhs: jnp.ndarray, weight: jnp.ndarray) -> jnp.ndarray:
        return jax.lax.cond(
            weight > 0.0,
            lambda _: jnp.linalg.solve(eye + 2.0 * weight * smoothness_gram, rhs),
            lambda _: rhs,
            operand=None,
        )

    return jax.vmap(solve_one_dim, in_axes=(1, 0), out_axes=1)(params5, weights)


def _select_gn_candidate(
    params5_prev: jnp.ndarray,
    dp_all: jnp.ndarray,
    *,
    loss_before: float,
    eval_loss: Callable[[jnp.ndarray], float],
    gn_accept_tol: float,
    constrain_candidate: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    smooth_candidate: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
    light_smoothness_weights_sq: jnp.ndarray | None = None,
    medium_smoothness_weights_sq: jnp.ndarray | None = None,
    smoothness_weights_sq: jnp.ndarray | None = None,
    trans_only_smoothness_weights_sq: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, float]:
    """Pick a GN candidate using a small hierarchical full-loss search."""

    def _accepts(candidate_loss: float, current_best_loss: float = float("inf")) -> bool:
        return _should_prefer_gn_candidate(
            loss_before,
            current_best_loss,
            candidate_loss,
            gn_accept_tol,
        )

    def _constrain(candidate: jnp.ndarray) -> jnp.ndarray:
        if constrain_candidate is None:
            return candidate
        return constrain_candidate(candidate)

    raw_params = _constrain(params5_prev + dp_all)
    raw_loss = eval_loss(raw_params)
    if _accepts(raw_loss):
        return raw_params, raw_loss

    half_params = _constrain(params5_prev + jnp.float32(0.5) * dp_all)
    half_loss = eval_loss(half_params)
    if _accepts(half_loss):
        return half_params, half_loss

    base_params = raw_params if raw_loss <= half_loss else half_params

    def _has_active_weights(weights: jnp.ndarray | None) -> bool:
        return weights is not None and bool(jnp.any(weights > 0.0))

    if smooth_candidate is None:
        return params5_prev, loss_before

    smooth_weights = [
        weights
        for weights in (
            light_smoothness_weights_sq,
            medium_smoothness_weights_sq,
            smoothness_weights_sq,
            trans_only_smoothness_weights_sq,
        )
        if _has_active_weights(weights)
    ]

    if not smooth_weights:
        return params5_prev, loss_before

    best_params = params5_prev
    best_loss = float("inf")
    accepted = False
    for weights in smooth_weights:
        candidate_params = _constrain(smooth_candidate(base_params, weights))
        candidate_loss = eval_loss(candidate_params)
        if _accepts(candidate_loss, best_loss):
            best_params = candidate_params
            best_loss = candidate_loss
            accepted = True

    if accepted:
        return best_params, best_loss
    return params5_prev, loss_before


_EXPECTED_ALIGN_EVAL_FAILURE_SNIPPETS = (
    "allocator",
    "cholesky",
    "failed to converge",
    "non-finite",
    "nonfinite",
    "not positive definite",
    "out of memory",
    "resource_exhausted",
    "singular",
    "svd",
)
_EXPECTED_ALIGN_EVAL_NUMERIC_TOKEN_RE = re.compile(
    r"(?<![a-z0-9_])(?:[+-]?(?:inf|nan)s?|infinite|infinity)(?![a-z0-9_])"
)


def _is_expected_align_eval_failure(exc: Exception) -> bool:
    if isinstance(exc, FloatingPointError):
        return True
    msg = str(exc).lower()
    return bool(_EXPECTED_ALIGN_EVAL_NUMERIC_TOKEN_RE.search(msg)) or any(
        snippet in msg for snippet in _EXPECTED_ALIGN_EVAL_FAILURE_SNIPPETS
    )


def _evaluate_align_loss(
    eval_loss: Callable[[], float | jnp.ndarray],
    *,
    fallback: float | None,
    context: str,
) -> float | None:
    try:
        return float(eval_loss())
    except Exception as exc:
        if _is_expected_align_eval_failure(exc):
            logging.warning("%s after expected numeric failure: %s", context, exc)
            return fallback
        raise


def _build_alignment_volume_mask(
    grid: Grid,
    detector: Detector,
    *,
    mask_vol: str,
) -> jnp.ndarray | None:
    mask_mode = str(mask_vol).lower()
    if mask_mode in ("off", "none", ""):
        return None
    if mask_mode not in ("cyl", "cylindrical"):
        raise ValueError("align mask_vol must be one of 'off' or 'cyl'")
    try:
        m_xy = cylindrical_mask_xy(grid, detector)
        return jnp.asarray(m_xy, dtype=jnp.float32)[:, :, None]
    except Exception as exc:
        raise ValueError(f"Failed to apply requested mask_vol={mask_mode!r}") from exc


@dataclass(frozen=True)
class _AlignSetupState:
    cfg: AlignConfig
    observer_fn: ObserverCallback | None
    n_views: int
    x: jnp.ndarray
    params5: jnp.ndarray
    frozen_params5: jnp.ndarray
    active_mask_tuple: tuple[bool, bool, bool, bool, bool]
    active_mask_bool: jnp.ndarray
    active_col_indices_np: np.ndarray
    active_names: tuple[str, ...]
    active_mask: jnp.ndarray
    bounds_lower: jnp.ndarray
    bounds_upper: jnp.ndarray
    gauge_fix: GaugeFixMode
    gauge_dofs: tuple[str, ...]


@dataclass(frozen=True)
class PoseConstraintContext:
    active_mask_tuple: tuple[bool, bool, bool, bool, bool]
    active_mask_bool: jnp.ndarray
    frozen_params5: jnp.ndarray
    bounds_lower: jnp.ndarray
    bounds_upper: jnp.ndarray
    gauge_fix: GaugeFixMode
    gauge_dofs: tuple[str, ...]

    @classmethod
    def from_setup(cls, setup: _AlignSetupState) -> PoseConstraintContext:
        return cls(
            active_mask_tuple=setup.active_mask_tuple,
            active_mask_bool=setup.active_mask_bool,
            frozen_params5=setup.frozen_params5,
            bounds_lower=setup.bounds_lower,
            bounds_upper=setup.bounds_upper,
            gauge_fix=setup.gauge_fix,
            gauge_dofs=setup.gauge_dofs,
        )

    def apply_param_constraints(self, candidate: jnp.ndarray) -> jnp.ndarray:
        clipped = jnp.clip(candidate, self.bounds_lower, self.bounds_upper)
        return jnp.where(self.active_mask_bool, clipped, self.frozen_params5)

    def apply_full_constraints(self, candidate: jnp.ndarray) -> jnp.ndarray:
        constrained = self.apply_param_constraints(candidate)
        gauged, _ = apply_alignment_gauge(
            constrained,
            mode=self.gauge_fix,
            active_mask=self.active_mask_tuple,
            bounds_lower=self.bounds_lower,
            bounds_upper=self.bounds_upper,
        )
        return self.apply_param_constraints(gauged)

    def apply_full_constraints_with_stats(
        self,
        candidate: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict[str, float | str | list[str]]]:
        constrained = self.apply_param_constraints(candidate)
        gauged, stats = apply_alignment_gauge(
            constrained,
            mode=self.gauge_fix,
            active_mask=self.active_mask_tuple,
            bounds_lower=self.bounds_lower,
            bounds_upper=self.bounds_upper,
        )
        gauged = self.apply_param_constraints(gauged)
        final_gauged, final_stats = apply_alignment_gauge(
            gauged,
            mode=self.gauge_fix,
            active_mask=self.active_mask_tuple,
            bounds_lower=self.bounds_lower,
            bounds_upper=self.bounds_upper,
        )
        final_gauged = self.apply_param_constraints(final_gauged)
        stats_py = gauge_stats_to_python(stats)
        final_py = gauge_stats_to_python(final_stats)
        stats_py["dx_mean_after"] = final_py["dx_mean_after"]
        stats_py["dz_mean_after"] = final_py["dz_mean_after"]
        return final_gauged, stats_py

    def description(self) -> str:
        if self.gauge_fix == "none":
            return "none"
        gauge_dofs_label = ",".join(self.gauge_dofs) if self.gauge_dofs else "no translation DOFs"
        return f"{self.gauge_fix} over active {gauge_dofs_label}"


@dataclass(frozen=True)
class PoseMotionContext:
    motion_model: Any
    use_smooth_pose_model: bool
    active_coeff_indices: jnp.ndarray
    params5: jnp.ndarray
    motion_coeffs: jnp.ndarray | None
    constraint_ctx: PoseConstraintContext

    @classmethod
    def build(
        cls,
        *,
        geometry: Geometry,
        cfg: AlignConfig,
        n_views: int,
        active_names: tuple[str, ...],
        params5: jnp.ndarray,
        resume_state: AlignResumeState | None,
        constraint_ctx: PoseConstraintContext,
    ) -> PoseMotionContext:
        scan_coordinate = scan_coordinate_from_geometry(geometry, n_views)
        motion_model = build_pose_motion_model(
            pose_model=str(cfg.pose_model),
            n_views=n_views,
            active_dofs=active_names,
            frozen_params5=constraint_ctx.frozen_params5,
            scan_coordinate=scan_coordinate,
            knot_spacing=int(cfg.knot_spacing),
            degree=int(cfg.degree),
        )
        use_smooth_pose_model = motion_model.name != "per_view"
        active_coeff_indices = jnp.asarray(motion_model.active_indices, dtype=jnp.int32)
        motion_coeffs = None
        constrained_params = params5
        if use_smooth_pose_model:
            motion_coeffs = fit_motion_coefficients(motion_model, constrained_params)
            constrained_params = constraint_ctx.apply_full_constraints(
                expand_motion_coefficients(motion_model, motion_coeffs)
            )
            motion_coeffs = fit_motion_coefficients(motion_model, constrained_params)
            if resume_state is not None and resume_state.motion_coeffs is not None:
                resume_coeffs = jnp.asarray(resume_state.motion_coeffs, dtype=jnp.float32)
                if tuple(resume_coeffs.shape) != tuple(motion_coeffs.shape):
                    raise ValueError(
                        "align resume_state motion_coeffs shape mismatch: "
                        f"expected {tuple(motion_coeffs.shape)}, got {tuple(resume_coeffs.shape)}"
                    )
                motion_coeffs = resume_coeffs
                constrained_params = constraint_ctx.apply_full_constraints(
                    expand_motion_coefficients(motion_model, motion_coeffs)
                )

        return cls(
            motion_model=motion_model,
            use_smooth_pose_model=use_smooth_pose_model,
            active_coeff_indices=active_coeff_indices,
            params5=constrained_params,
            motion_coeffs=motion_coeffs,
            constraint_ctx=constraint_ctx,
        )

    def coeffs_to_constrained_params(self, coeffs: jnp.ndarray) -> jnp.ndarray:
        return self.constraint_ctx.apply_full_constraints(
            expand_motion_coefficients(self.motion_model, coeffs)
        )

    def project_params_to_smooth(self, candidate: jnp.ndarray) -> jnp.ndarray:
        constrained = self.constraint_ctx.apply_full_constraints(candidate)
        coeffs = fit_motion_coefficients(self.motion_model, constrained)
        return self.coeffs_to_constrained_params(coeffs)

    def loss_and_grad_for(
        self,
        align_loss: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    ) -> Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]] | None:
        if not self.use_smooth_pose_model:
            return None

        def motion_align_loss(coeffs: jnp.ndarray, vol: jnp.ndarray) -> jnp.ndarray:
            return align_loss(self.coeffs_to_constrained_params(coeffs), vol)

        return jax.jit(jax.value_and_grad(motion_align_loss))


@dataclass(frozen=True)
class AlignmentRuntimeContext:
    pose_stack: jnp.ndarray
    det_grid: tuple[jnp.ndarray, jnp.ndarray]
    smoothness_weights: jnp.ndarray
    smoothness_gram: jnp.ndarray
    smoothness_weights_sq: jnp.ndarray
    medium_smoothness_weights_sq: jnp.ndarray
    trans_only_smoothness_weights_sq: jnp.ndarray
    light_smoothness_weights_sq: jnp.ndarray
    volume_mask: jnp.ndarray | None
    active_loss_name: str
    loss_adapter: LossAdapter
    loss_mask: jnp.ndarray | None
    has_loss_mask: bool
    supports_gauss_newton: bool
    objective_provenance: dict[str, str]
    nv: int
    nu: int
    chunk_size: int
    num_chunks: int
    empty_loss_mask_chunk: jnp.ndarray


@dataclass(frozen=True)
class PoseObjectiveBundle:
    align_loss: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    align_loss_jit: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    loss_and_grad_manual: Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]
    gn_update_all: Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]


@dataclass(frozen=True)
class _PoseObjectiveContext:
    grid: Grid
    detector: Detector
    projections: jnp.ndarray
    cfg: AlignConfig
    n_views: int
    active_mask: jnp.ndarray
    pose_stack: jnp.ndarray
    det_grid: tuple[jnp.ndarray, jnp.ndarray]
    smoothness_weights: jnp.ndarray
    volume_mask: jnp.ndarray | None
    loss_adapter: LossAdapter
    per_view_loss_fn: Callable[..., jnp.ndarray]
    nv: int
    nu: int
    chunk_size: int
    num_chunks: int
    loss_mask: jnp.ndarray | None
    has_loss_mask: bool
    empty_loss_mask_chunk: jnp.ndarray


def _pose_objective_context(
    *,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    cfg: AlignConfig,
    n_views: int,
    active_mask: jnp.ndarray,
    runtime: AlignmentRuntimeContext,
) -> _PoseObjectiveContext:
    return _PoseObjectiveContext(
        grid=grid,
        detector=detector,
        projections=projections,
        cfg=cfg,
        n_views=n_views,
        active_mask=active_mask,
        pose_stack=runtime.pose_stack,
        det_grid=runtime.det_grid,
        smoothness_weights=runtime.smoothness_weights,
        volume_mask=runtime.volume_mask,
        loss_adapter=runtime.loss_adapter,
        per_view_loss_fn=runtime.loss_adapter.per_view_loss,
        nv=runtime.nv,
        nu=runtime.nu,
        chunk_size=runtime.chunk_size,
        num_chunks=runtime.num_chunks,
        loss_mask=runtime.loss_mask,
        has_loss_mask=runtime.has_loss_mask,
        empty_loss_mask_chunk=runtime.empty_loss_mask_chunk,
    )


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
        _, vjp = jax.vjp(f, p5_i)
        gradient = vjp(residual)[0]
        eye5 = jnp.eye(5, dtype=jnp.float32)

        def jvp_col(v: jnp.ndarray) -> jnp.ndarray:
            return jax.jvp(f, (p5_i,), (v,))[1]

        cols = jax.vmap(jvp_col)(eye5)
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


def _recon_summary_parts(stat: OuterStat, *, compact: bool) -> list[str]:
    parts: list[str] = []
    digits = 2 if compact else 3
    recon_time = stat.get("recon_time")
    if recon_time is not None:
        prefix = "" if compact else "time "
        parts.append(f"{prefix}{format_duration(recon_time)}")
    if stat.get("recon_retry"):
        parts.append("retry" if compact else "fallback retry")
    l_meas = stat.get("L_meas")
    l_next = stat.get("L_next")
    if (l_meas is not None) and (l_next is not None):
        parts.append(f"L {float(l_meas):.{digits}e}->{float(l_next):.{digits}e}")
    f_first = stat.get("recon_loss_first")
    f_last = stat.get("recon_loss_last")
    f_min = stat.get("recon_loss_min")
    if (f_first is not None) and (f_last is not None):
        loss = f"loss {float(f_first):.{digits}e}->{float(f_last):.{digits}e}"
        if f_min is not None:
            loss += f" (min {float(f_min):.{digits}e})"
        parts.append(loss)
    return parts


def _gauge_summary_parts(stat: OuterStat, *, compact: bool) -> list[str]:
    if stat.get("gauge_fix") == "none":
        return ["gauge none"]
    if stat.get("gauge_fix") != "mean_translation":
        return []
    dxm = stat.get("dx_mean_before_gauge")
    dzm = stat.get("dz_mean_before_gauge")
    if compact:
        if dxm is not None and dzm is not None:
            return [f"gauge mean dx,dz {float(dxm):+.2e},{float(dzm):+.2e}->0"]
        return []
    dxa = stat.get("dx_mean_after_gauge")
    dza = stat.get("dz_mean_after_gauge")
    if dxm is not None and dzm is not None and dxa is not None and dza is not None:
        return [
            "gauge mean dx,dz "
            f"{float(dxm):+.3e},{float(dzm):+.3e}->"
            f"{float(dxa):+.3e},{float(dza):+.3e}"
        ]
    return []


def _gn_summary_parts(stat: OuterStat, *, compact: bool, digits: int) -> list[str]:
    parts: list[str] = []
    rot_mean = stat.get("rot_mean")
    trans_mean = stat.get("trans_mean")
    if rot_mean is not None:
        label = "|drot|" if compact else "|drot|_mean"
        suffix = "" if compact else " rad"
        parts.append(f"{label} {float(rot_mean):.{digits}e}{suffix}")
    if trans_mean is not None:
        label = "|dtrans|" if compact else "|dtrans|_mean"
        parts.append(f"{label} {float(trans_mean):.{digits}e}")
    return parts


def _lbfgs_summary_parts(stat: OuterStat, *, compact: bool) -> list[str]:
    parts: list[str] = []
    status = "accepted" if stat.get("lbfgs_accepted") else "rejected"
    if stat.get("lbfgs_fallback_to_gd"):
        status = "fallback->gd" if compact else "fallback to GD"
    parts.append(status if compact else f"L-BFGS {status}")
    if compact:
        best = stat.get("lbfgs_best_loss")
        if best is not None:
            parts.append(f"best {float(best):.2e}")
    else:
        for src, label in (
            ("lbfgs_initial_loss", "initial"),
            ("lbfgs_final_loss", "final"),
            ("lbfgs_best_loss", "best"),
        ):
            value = stat.get(src)
            if value is not None:
                parts.append(f"{label} {float(value):.3e}")
    nit = stat.get("lbfgs_nit")
    nfev = stat.get("lbfgs_nfev")
    if nit is not None:
        parts.append(f"nit {int(nit)}")
    if nfev is not None:
        parts.append(f"nfev {int(nfev)}")
    if not compact:
        message = stat.get("lbfgs_message")
        if message:
            parts.append(str(message))
    for src, label in (("rot_mean", "|drot|"), ("trans_mean", "|dtrans|")):
        value = stat.get(src)
        if value is not None and compact:
            parts.append(f"{label} {float(value):.2e}")
    return parts


def _gd_summary_parts(stat: OuterStat, *, compact: bool, digits: int) -> list[str]:
    parts: list[str] = []
    if stat.get("lbfgs_fallback_to_gd"):
        message = stat.get("lbfgs_message")
        parts.append(
            "lbfgs fallback"
            if compact
            else "L-BFGS fallback to GD" + (f": {message}" if message else "")
        )
    rot_rms = stat.get("rot_rms")
    trans_rms = stat.get("trans_rms")
    if rot_rms is not None:
        label = "rotRMS" if compact else "rot RMS"
        parts.append(f"{label} {float(rot_rms):.{digits}e}")
    if trans_rms is not None:
        label = "transRMS" if compact else "trans RMS"
        parts.append(f"{label} {float(trans_rms):.{digits}e}")
    return parts


def _step_summary_parts(stat: OuterStat, *, compact: bool, digits: int) -> list[str]:
    step_kind = stat.get("step_kind")
    if step_kind == "gn":
        return _gn_summary_parts(stat, compact=compact, digits=digits)
    if step_kind == "lbfgs":
        return _lbfgs_summary_parts(stat, compact=compact)
    if step_kind == "gd":
        return _gd_summary_parts(stat, compact=compact, digits=digits)
    return []


def _align_loss_summary_parts(stat: OuterStat, *, compact: bool, digits: int) -> list[str]:
    loss_before = stat.get("loss_before")
    loss_after = stat.get("loss_after")
    if (loss_before is None) or (loss_after is None):
        return []
    loss_delta = stat.get("loss_delta")
    rel_pct = stat.get("loss_rel_pct")
    sep = " " if compact else ", "
    rel = f"{sep}{float(rel_pct):+.2f}%" if rel_pct is not None else ""
    return [
        f"loss {float(loss_before):.{digits}e}->{float(loss_after):.{digits}e} "
        f"(Δ {float(loss_delta):+.{digits}e}{rel})"
    ]


def _align_summary_parts(stat: OuterStat, *, compact: bool) -> list[str]:
    parts: list[str] = []
    digits = 2 if compact else 3
    align_time = stat.get("align_time")
    if align_time is not None:
        prefix = "" if compact else "time "
        parts.append(f"{prefix}{format_duration(align_time)}")
    parts.extend(_step_summary_parts(stat, compact=compact, digits=digits))
    parts.extend(_align_loss_summary_parts(stat, compact=compact, digits=digits))
    parts.extend(_gauge_summary_parts(stat, compact=compact))
    return parts


def _format_outer_summary_lines(
    stat: OuterStat,
    *,
    cfg: AlignConfig,
    recon_algo: str,
) -> list[str]:
    outer_idx = int(stat.get("outer_idx", 0))
    total_iters = int(cfg.outer_iters)
    total_time = format_duration(stat.get("outer_time"))
    elapsed = format_duration(stat.get("cumulative_time"))
    solver_label = str(stat.get("recon_algo") or recon_algo).upper()
    if cfg.log_compact:
        parts: list[str] = [f"Outer {outer_idx}/{total_iters}"]
        recon_parts = _recon_summary_parts(stat, compact=True)
        align_parts = _align_summary_parts(stat, compact=True)
        if recon_parts:
            parts.append(f"recon {solver_label.lower()} " + " ".join(recon_parts))
        if align_parts:
            parts.append("align " + " ".join(align_parts))
        parts.append(f"elapsed {elapsed}")
        return [" | ".join(parts)]
    recon_parts = _recon_summary_parts(stat, compact=False)
    align_parts = _align_summary_parts(stat, compact=False)
    return [
        f"Outer {outer_idx}/{total_iters} | total {total_time} | elapsed {elapsed}",
        f"  Recon ({solver_label}) | {' | '.join(recon_parts) if recon_parts else '-'}",
        f"  Align | {' | '.join(align_parts) if align_parts else '-'}",
    ]


@dataclass(frozen=True)
class AlignmentStepContext:
    cfg: AlignConfig
    opt_mode: str
    active_loss_name: str
    ls_like: bool
    align_loss: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    align_loss_jit: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    loss_and_grad_manual: Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]
    gn_update_all: Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]
    active_mask: jnp.ndarray
    active_coeff_indices: jnp.ndarray
    active_col_indices_np: np.ndarray
    frozen_params5: jnp.ndarray
    bounds_lower: jnp.ndarray
    bounds_upper: jnp.ndarray
    apply_full_constraints: Callable[[jnp.ndarray], jnp.ndarray]
    apply_full_constraints_with_stats: Callable[
        [jnp.ndarray],
        tuple[jnp.ndarray, dict[str, float | str | list[str]]],
    ]
    project_params_to_smooth: Callable[[jnp.ndarray], jnp.ndarray]
    coeffs_to_constrained_params: Callable[[jnp.ndarray], jnp.ndarray]
    use_smooth_pose_model: bool
    motion_loss_and_grad: (
        Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]] | None
    )
    motion_model: Any
    smoothness_gram: jnp.ndarray
    light_smoothness_weights_sq: jnp.ndarray
    medium_smoothness_weights_sq: jnp.ndarray
    smoothness_weights_sq: jnp.ndarray
    trans_only_smoothness_weights_sq: jnp.ndarray
    gauge_fix: str
    gauge_dofs: tuple[str, ...]


@dataclass(frozen=True)
class _AlignmentStepCoreResult:
    params5: jnp.ndarray
    motion_coeffs: jnp.ndarray | None
    loss_before: float | None
    loss_after: float | None
    step_kind: str
    stat: OuterStat


def _alignment_step_kind(ctx: AlignmentStepContext) -> tuple[str, OuterStat]:
    if ctx.opt_mode == "gn" and ctx.ls_like:
        return "gn", {}
    if ctx.opt_mode == "gn":
        logging.warning(
            "Gauss-Newton is incompatible with loss=%s; falling back to GD for this step",
            ctx.active_loss_name,
        )
        return "gd", {"optimizer_fallback": "gn->gd"}
    if ctx.opt_mode == "lbfgs":
        return "lbfgs", {}
    return "gd", {}


def _pre_alignment_step_loss(
    ctx: AlignmentStepContext,
    *,
    params5: jnp.ndarray,
    vol: jnp.ndarray,
    step_kind: str,
) -> float | None:
    if step_kind == "gn" and not ctx.use_smooth_pose_model:
        return None
    return _evaluate_align_loss(
        lambda: ctx.align_loss_jit(params5, vol),
        fallback=None,
        context="Skipping pre-step alignment loss evaluation",
    )


def _select_gd_step_candidate(
    ctx: AlignmentStepContext,
    *,
    base_params: jnp.ndarray,
    doubled_params: jnp.ndarray,
    previous_params: jnp.ndarray,
    loss_before_value: float | None,
    vol: jnp.ndarray,
) -> tuple[jnp.ndarray, float | None]:
    base_loss = _evaluate_align_loss(
        lambda: ctx.align_loss_jit(base_params, vol),
        fallback=math.inf,
        context="Treating GD base candidate as rejected during alignment loss evaluation",
    )
    doubled_loss = _evaluate_align_loss(
        lambda: ctx.align_loss_jit(doubled_params, vol),
        fallback=math.inf,
        context="Treating GD doubled-step candidate as rejected during alignment loss evaluation",
    )
    base_loss_f = float(base_loss) if base_loss is not None else math.inf
    doubled_loss_f = float(doubled_loss) if doubled_loss is not None else math.inf
    if not math.isfinite(base_loss_f) and not math.isfinite(doubled_loss_f):
        return previous_params, loss_before_value
    if doubled_loss_f < base_loss_f:
        chosen_params = doubled_params
        chosen_loss = doubled_loss_f
    else:
        chosen_params = base_params
        chosen_loss = base_loss_f
    return (
        chosen_params,
        float(chosen_loss) if math.isfinite(chosen_loss) else loss_before_value,
    )


def _run_gd_alignment_step(
    ctx: AlignmentStepContext,
    params5_in: jnp.ndarray,
    motion_coeffs_in: jnp.ndarray | None,
    vol: jnp.ndarray,
    loss_before_value: float | None,
) -> tuple[jnp.ndarray, jnp.ndarray | None, float | None, jnp.ndarray]:
    cfg = ctx.cfg
    scales = jnp.array(
        [cfg.lr_rot, cfg.lr_rot, cfg.lr_rot, cfg.lr_trans, cfg.lr_trans],
        dtype=jnp.float32,
    )
    if ctx.use_smooth_pose_model:
        if motion_coeffs_in is None or ctx.motion_loss_and_grad is None:
            raise RuntimeError("smooth pose model coefficients were not initialized")
        coeffs_in = motion_coeffs_in
        _, g_coeffs = ctx.motion_loss_and_grad(coeffs_in, vol)
        active_scales = scales[ctx.active_coeff_indices]
        rms_active = jnp.sqrt(jnp.mean(jnp.square(g_coeffs), axis=0)) + 1e-6
        eff_scales = active_scales / rms_active
        best_coeffs = coeffs_in - g_coeffs * eff_scales[None, :]
        best_params = ctx.coeffs_to_constrained_params(best_coeffs)
        cand_coeffs = coeffs_in - 2.0 * g_coeffs * eff_scales[None, :]
        cand_params = ctx.coeffs_to_constrained_params(cand_coeffs)
        params5_out, loss_after_value = _select_gd_step_candidate(
            ctx,
            base_params=best_params,
            doubled_params=cand_params,
            previous_params=ctx.coeffs_to_constrained_params(coeffs_in),
            loss_before_value=loss_before_value,
            vol=vol,
        )
        motion_coeffs_out = fit_motion_coefficients(ctx.motion_model, params5_out)
        params5_out = ctx.coeffs_to_constrained_params(motion_coeffs_out)
        rms = jnp.zeros((5,), dtype=jnp.float32).at[ctx.active_coeff_indices].set(rms_active)
        return params5_out, motion_coeffs_out, loss_after_value, rms

    _, g_params = ctx.loss_and_grad_manual(params5_in, vol)
    g_params = g_params * ctx.active_mask
    rms = jnp.sqrt(jnp.mean(jnp.square(g_params), axis=0)) + 1e-6
    eff_scales = scales / rms
    best_params = ctx.apply_full_constraints(params5_in - g_params * eff_scales)
    cand_params = ctx.apply_full_constraints(params5_in - 2.0 * g_params * eff_scales)
    params5_out, loss_after_value = _select_gd_step_candidate(
        ctx,
        base_params=best_params,
        doubled_params=cand_params,
        previous_params=params5_in,
        loss_before_value=loss_before_value,
        vol=vol,
    )
    return params5_out, motion_coeffs_in, loss_after_value, rms


def _run_lbfgs_alignment_step(
    ctx: AlignmentStepContext,
    params5_in: jnp.ndarray,
    motion_coeffs_in: jnp.ndarray | None,
    vol: jnp.ndarray,
    loss_before_value: float | None,
) -> tuple[jnp.ndarray, jnp.ndarray | None, float | None, OuterStat]:
    cfg = ctx.cfg
    result = run_pose_lbfgs(
        params5_in=params5_in,
        motion_coeffs_in=motion_coeffs_in,
        loss_before_value=loss_before_value,
        objective_fn=lambda candidate: ctx.align_loss(candidate, vol),
        eval_loss_fn=lambda candidate, label: _evaluate_align_loss(
            lambda: ctx.align_loss_jit(candidate, vol),
            fallback=math.inf,
            context=f"Treating L-BFGS {label} candidate as rejected "
            "during alignment loss evaluation",
        ),
        is_expected_failure=_is_expected_align_eval_failure,
        cfg=PoseLbfgsConfig(
            maxiter=int(cfg.lbfgs_maxiter),
            ftol=float(cfg.lbfgs_ftol),
            gtol=float(cfg.lbfgs_gtol),
            maxls=int(cfg.lbfgs_maxls),
            memory_size=int(cfg.lbfgs_memory_size),
        ),
        context=PoseOptimizationContext(
            active_cols=ctx.active_col_indices_np,
            frozen_params5=ctx.frozen_params5,
            bounds_lower=ctx.bounds_lower,
            bounds_upper=ctx.bounds_upper,
            apply_param_constraints=ctx.apply_full_constraints,
            motion_model=ctx.motion_model if ctx.use_smooth_pose_model else None,
        ),
    )
    if result.stats.get("lbfgs_fallback_to_gd"):
        logging.warning(
            "%s; falling back to GD for this alignment step",
            result.stats.get("lbfgs_message"),
        )
    return result.params5, result.motion_coeffs, result.loss, result.stats


def _gn_smooth_candidate_fn(
    ctx: AlignmentStepContext,
    constrain_candidate: Callable[[jnp.ndarray], jnp.ndarray],
    params5_in: jnp.ndarray,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None:
    if int(params5_in.shape[0]) < 3:
        return None

    def smooth_candidate(candidate: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        return _smooth_gn_candidate(
            constrain_candidate(candidate),
            ctx.smoothness_gram,
            weights,
        )

    return smooth_candidate


def _run_gn_alignment_step(
    ctx: AlignmentStepContext,
    params5_in: jnp.ndarray,
    motion_coeffs_in: jnp.ndarray | None,
    vol: jnp.ndarray,
    loss_before_value: float | None,
) -> tuple[jnp.ndarray, jnp.ndarray | None, float | None, float | None, OuterStat]:
    cfg = ctx.cfg
    params5_prev = params5_in
    dp_raw, gn_loss_before = ctx.gn_update_all(params5_prev, vol)
    dp_all = dp_raw * ctx.active_mask
    loss_before = loss_before_value
    if not ctx.use_smooth_pose_model:
        loss_before = float(gn_loss_before)
    constrain_candidate = (
        ctx.project_params_to_smooth if ctx.use_smooth_pose_model else ctx.apply_full_constraints
    )
    if cfg.gn_accept_only_improving and (loss_before is not None):
        params5_out, loss_after = _select_gn_candidate(
            params5_prev,
            dp_all,
            loss_before=loss_before,
            eval_loss=lambda candidate: float(
                _evaluate_align_loss(
                    lambda: ctx.align_loss_jit(candidate, vol),
                    fallback=math.inf,
                    context="Treating GN candidate as rejected during alignment loss evaluation",
                )
            ),
            gn_accept_tol=cfg.gn_accept_tol,
            constrain_candidate=constrain_candidate,
            smooth_candidate=_gn_smooth_candidate_fn(ctx, constrain_candidate, params5_in),
            light_smoothness_weights_sq=ctx.light_smoothness_weights_sq,
            medium_smoothness_weights_sq=ctx.medium_smoothness_weights_sq,
            smoothness_weights_sq=ctx.smoothness_weights_sq,
            trans_only_smoothness_weights_sq=ctx.trans_only_smoothness_weights_sq,
        )
        params5_out = constrain_candidate(params5_out)
    else:
        params5_out = constrain_candidate(params5_prev + dp_all)
        candidate_loss = _evaluate_align_loss(
            lambda: ctx.align_loss_jit(params5_out, vol),
            fallback=math.inf,
            context="Treating GN step as rejected during alignment loss evaluation",
        )
        if candidate_loss is not None and math.isfinite(candidate_loss):
            loss_after = candidate_loss
        else:
            params5_out = params5_prev
            loss_after = loss_before
    motion_coeffs_out = motion_coeffs_in
    if ctx.use_smooth_pose_model:
        motion_coeffs_out = fit_motion_coefficients(ctx.motion_model, params5_out)
        params5_out = ctx.coeffs_to_constrained_params(motion_coeffs_out)
    stat: OuterStat = {}
    _set_float_stat(stat, "rot_mean", jnp.mean(jnp.abs(dp_all[:, :3])))
    _set_float_stat(stat, "trans_mean", jnp.mean(jnp.abs(dp_all[:, 3:])))
    return params5_out, motion_coeffs_out, loss_before, loss_after, stat


def _run_alignment_step_core(
    ctx: AlignmentStepContext,
    params5_in: jnp.ndarray,
    motion_coeffs_in: jnp.ndarray | None,
    vol: jnp.ndarray,
) -> _AlignmentStepCoreResult:
    step_kind, stat = _alignment_step_kind(ctx)
    loss_before = _pre_alignment_step_loss(
        ctx,
        params5=params5_in,
        vol=vol,
        step_kind=step_kind,
    )
    if step_kind == "gn":
        params5_out, motion_coeffs_out, loss_before, loss_after, gn_stat = _run_gn_alignment_step(
            ctx, params5_in, motion_coeffs_in, vol, loss_before
        )
        stat.update(gn_stat)
    elif step_kind == "lbfgs":
        params5_out, motion_coeffs_out, loss_after, lbfgs_stats = _run_lbfgs_alignment_step(
            ctx,
            params5_in,
            motion_coeffs_in,
            vol,
            loss_before,
        )
        stat.update(lbfgs_stats)
        if stat.get("lbfgs_fallback_to_gd"):
            step_kind = "gd"
            params5_out, motion_coeffs_out, loss_after, rms = _run_gd_alignment_step(
                ctx,
                params5_out,
                motion_coeffs_out,
                vol,
                loss_before,
            )
            _set_float_stat(stat, "rot_rms", jnp.mean(rms[:3]))
            _set_float_stat(stat, "trans_rms", jnp.mean(rms[3:]))
    else:
        params5_out, motion_coeffs_out, loss_after, rms = _run_gd_alignment_step(
            ctx,
            params5_in,
            motion_coeffs_in,
            vol,
            loss_before,
        )
        _set_float_stat(stat, "rot_rms", jnp.mean(rms[:3]))
        _set_float_stat(stat, "trans_rms", jnp.mean(rms[3:]))
    return _AlignmentStepCoreResult(
        params5=params5_out,
        motion_coeffs=motion_coeffs_out,
        loss_before=loss_before,
        loss_after=loss_after,
        step_kind=step_kind,
        stat=stat,
    )


def _apply_final_alignment_constraints(
    ctx: AlignmentStepContext,
    params5: jnp.ndarray,
    motion_coeffs: jnp.ndarray | None,
) -> tuple[jnp.ndarray, jnp.ndarray | None, dict[str, float | str | list[str]]]:
    params5, gauge_stats = ctx.apply_full_constraints_with_stats(params5)
    motion_coeffs_out = motion_coeffs
    if ctx.use_smooth_pose_model:
        motion_coeffs_out = fit_motion_coefficients(ctx.motion_model, params5)
        params5, gauge_stats = ctx.apply_full_constraints_with_stats(
            expand_motion_coefficients(ctx.motion_model, motion_coeffs_out)
        )
        motion_coeffs_out = fit_motion_coefficients(ctx.motion_model, params5)
    return params5, motion_coeffs_out, gauge_stats


def _record_gauge_stats(
    stat: OuterStat,
    ctx: AlignmentStepContext,
    gauge_stats: dict[str, float | str | list[str]],
) -> None:
    stat["gauge_fix"] = ctx.gauge_fix
    stat["gauge_fix_dofs"] = ",".join(ctx.gauge_dofs)
    if ctx.gauge_fix == "mean_translation":
        stat["dx_mean_before_gauge"] = float(gauge_stats["dx_mean_before"])
        stat["dz_mean_before_gauge"] = float(gauge_stats["dz_mean_before"])
        stat["dx_mean_after_gauge"] = float(gauge_stats["dx_mean_after"])
        stat["dz_mean_after_gauge"] = float(gauge_stats["dz_mean_after"])


def _alignment_total_loss(
    ctx: AlignmentStepContext,
    *,
    params5: jnp.ndarray,
    vol: jnp.ndarray,
    loss_before: float | None,
    loss_after: float | None,
    step_kind: str,
    loss_hist: list[float],
    stat: OuterStat,
) -> float:
    final_loss_fallback = loss_after
    if final_loss_fallback is None:
        final_loss_fallback = loss_before
    if final_loss_fallback is None and loss_hist:
        final_loss_fallback = loss_hist[-1]
    can_reuse_validated_gn_loss = (
        step_kind == "gn"
        and not ctx.use_smooth_pose_model
        and loss_after is not None
        and math.isfinite(float(loss_after))
    )
    if can_reuse_validated_gn_loss:
        stat["loss_after_reused"] = True
        return float(loss_after)
    total_loss_eval = _evaluate_align_loss(
        lambda: ctx.align_loss_jit(params5, vol),
        fallback=final_loss_fallback,
        context="Using fallback for final alignment loss bookkeeping",
    )
    stat["loss_after_reused"] = False
    return float(total_loss_eval) if total_loss_eval is not None else math.nan


def _alignment_relative_improvement(
    stat: OuterStat,
    *,
    loss_before: float | None,
    total_loss: float,
) -> float | None:
    if loss_before is None:
        stat["loss_delta"] = None
        stat["loss_rel_pct"] = None
        return None
    delta = total_loss - loss_before
    stat["loss_delta"] = delta
    if math.isfinite(loss_before) and abs(loss_before) > 1e-12:
        stat["loss_rel_pct"] = (delta / loss_before) * 100.0
    else:
        stat["loss_rel_pct"] = None
    if math.isfinite(loss_before) and math.isfinite(total_loss):
        return (loss_before - total_loss) / max(abs(loss_before), 1e-12)
    return None


def _should_reject_post_constraint_loss(
    *,
    loss_before: float | None,
    total_loss: float,
    rel_tol: float,
) -> bool:
    if loss_before is None:
        return False
    before = float(loss_before)
    after = float(total_loss)
    if not (math.isfinite(before) and math.isfinite(after)):
        return False
    if after <= before:
        return False
    return not (loss_is_within_relative_tolerance(before, after, rel_tol))


def _run_alignment_step(
    ctx: AlignmentStepContext,
    params5_in: jnp.ndarray,
    motion_coeffs_in: jnp.ndarray | None,
    vol: jnp.ndarray,
    *,
    loss_hist: list[float],
) -> tuple[
    jnp.ndarray,
    jnp.ndarray | None,
    dict[str, float | str | list[str]],
    float,
    float | None,
    OuterStat,
]:
    align_start = time.perf_counter()
    result = _run_alignment_step_core(ctx, params5_in, motion_coeffs_in, vol)
    params5_out, motion_coeffs_out, gauge_stats = _apply_final_alignment_constraints(
        ctx,
        result.params5,
        result.motion_coeffs,
    )
    stat = result.stat
    stat["loss_before"] = result.loss_before
    stat["step_kind"] = result.step_kind
    stat["optimizer_kind"] = result.step_kind
    stat["loss_after_step"] = result.loss_after
    _record_gauge_stats(stat, ctx, gauge_stats)
    jax.block_until_ready(params5_out)
    stat["align_time"] = time.perf_counter() - align_start
    total_loss = _alignment_total_loss(
        ctx,
        params5=params5_out,
        vol=vol,
        loss_before=result.loss_before,
        loss_after=result.loss_after,
        step_kind=result.step_kind,
        loss_hist=loss_hist,
        stat=stat,
    )
    if ctx.cfg.gn_accept_only_improving and _should_reject_post_constraint_loss(
        loss_before=result.loss_before,
        total_loss=total_loss,
        rel_tol=float(ctx.cfg.gn_accept_tol),
    ):
        params5_out = params5_in
        motion_coeffs_out = motion_coeffs_in
        total_loss = float(result.loss_before)
        stat["post_constraint_rejected"] = True
        stat["post_constraint_reject_reason"] = "loss_after_constraints_worse_than_before"
    else:
        stat["post_constraint_rejected"] = False
    stat["loss_after"] = total_loss
    rel_impr = _alignment_relative_improvement(
        stat,
        loss_before=result.loss_before,
        total_loss=total_loss,
    )
    stat["rel_impr"] = rel_impr
    return params5_out, motion_coeffs_out, gauge_stats, total_loss, rel_impr, stat


def _prepare_align_setup(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    cfg: AlignConfig | None,
    init_x: jnp.ndarray | None,
    init_params5: jnp.ndarray | None,
    observer: ObserverCallback | None,
    resume_state: AlignResumeState | None,
) -> _AlignSetupState:
    if cfg is None:
        cfg = AlignConfig()
    observer_fn = adapt_observer_callback(observer) if observer is not None else None
    validate_grid(grid, "align grid")
    n_views, _, _ = validate_projection_stack(
        projections,
        detector,
        geometry=geometry,
        context="align projections",
    )
    if resume_state is not None:
        init_x = resume_state.x
        init_params5 = resume_state.params5
    if init_x is not None:
        validate_volume(init_x, grid, context="align init_x", name="init_x")
    validate_optional_same_shape(
        init_params5,
        (n_views, 5),
        context="align init_params5",
        name="init_params5",
        fix="pass one 5-parameter alignment row per projection view.",
    )
    x = (
        jnp.asarray(init_x, dtype=jnp.float32)
        if init_x is not None
        else jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    )
    params5 = (
        jnp.asarray(init_params5, dtype=jnp.float32)
        if init_params5 is not None
        else jnp.zeros((n_views, 5), dtype=jnp.float32)
    )
    frozen_params5 = params5
    active_mask_tuple = _active_dof_mask_for_cfg(cfg)
    active_mask_bool = jnp.asarray(active_mask_tuple, dtype=bool)
    active_col_indices_np = np.asarray(
        [idx for idx, is_active in enumerate(active_mask_tuple) if is_active],
        dtype=np.int32,
    )
    active_names = _active_dofs_for_cfg(cfg)
    active_mask = active_mask_bool.astype(jnp.float32)
    bounds_lower, bounds_upper = bounds_vectors(cfg.bounds)
    gauge_fix = normalize_gauge_fix(cfg.gauge_fix)
    gauge_dofs = active_gauge_dofs(mode=gauge_fix, active_mask=active_mask_tuple)
    validate_alignment_gauge_feasible(
        mode=gauge_fix,
        active_mask=active_mask_tuple,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
    )
    return _AlignSetupState(
        cfg=cfg,
        observer_fn=observer_fn,
        n_views=n_views,
        x=x,
        params5=params5,
        frozen_params5=frozen_params5,
        active_mask_tuple=active_mask_tuple,
        active_mask_bool=active_mask_bool,
        active_col_indices_np=active_col_indices_np,
        active_names=active_names,
        active_mask=active_mask,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        gauge_fix=gauge_fix,
        gauge_dofs=gauge_dofs,
    )


def _build_alignment_runtime_context(
    *,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    cfg: AlignConfig,
    n_views: int,
    active_mask: jnp.ndarray,
    det_grid_override: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> AlignmentRuntimeContext:
    pose_stack = stack_view_poses(geometry, n_views)
    validate_pose_stack(pose_stack, n_views, context="align geometry")
    det_grid = (
        get_detector_grid_device(detector) if det_grid_override is None else det_grid_override
    )

    smoothness_weights = (
        jnp.array(
            [cfg.w_rot, cfg.w_rot, cfg.w_rot, cfg.w_trans, cfg.w_trans],
            dtype=jnp.float32,
        )
        * active_mask
    )
    smoothness_gram = _second_difference_gram(n_views)
    smoothness_weights_sq = smoothness_weights * smoothness_weights
    medium_smoothness_weights_sq = smoothness_weights_sq * jnp.float32(0.4)
    trans_only_smoothness_weights_sq = smoothness_weights_sq.at[:3].set(0.0)
    light_smoothness_weights_sq = smoothness_weights_sq * jnp.float32(0.25)

    active_loss_spec = resolve_loss_for_level(cfg.loss, level_factor=1)
    active_loss_name = loss_spec_name(active_loss_spec)
    loss_adapter = build_loss_adapter(active_loss_spec, projections)
    loss_mask = getattr(loss_adapter.state, "mask", None)
    chunk_size = int(cfg.views_per_batch) if int(cfg.views_per_batch) > 0 else n_views
    chunk_size = min(chunk_size, n_views)
    nv = int(projections.shape[1])
    nu = int(projections.shape[2])

    return AlignmentRuntimeContext(
        pose_stack=pose_stack,
        det_grid=det_grid,
        smoothness_weights=smoothness_weights,
        smoothness_gram=smoothness_gram,
        smoothness_weights_sq=smoothness_weights_sq,
        medium_smoothness_weights_sq=medium_smoothness_weights_sq,
        trans_only_smoothness_weights_sq=trans_only_smoothness_weights_sq,
        light_smoothness_weights_sq=light_smoothness_weights_sq,
        volume_mask=_build_alignment_volume_mask(
            grid,
            detector,
            mask_vol=str(getattr(cfg, "mask_vol", "off")),
        ),
        active_loss_name=active_loss_name,
        loss_adapter=loss_adapter,
        loss_mask=loss_mask,
        has_loss_mask=loss_mask is not None,
        supports_gauss_newton=loss_adapter.supports_gauss_newton,
        objective_provenance=ObjectiveProvenance(
            outer_loss_source="AlignmentLossSpec",
            outer_loss_kind=active_loss_name,
            inner_data_term="current_reconstruction",
            inner_regulariser="none",
            validation_split="none",
            differentiation_mode="none",
            initialization_policy="current_level_volume",
        ).to_dict(),
        nv=nv,
        nu=nu,
        chunk_size=chunk_size,
        num_chunks=(n_views + chunk_size - 1) // chunk_size,
        empty_loss_mask_chunk=jnp.zeros((chunk_size, nv, nu), dtype=jnp.float32),
    )


def _build_alignment_step_context(
    *,
    cfg: AlignConfig,
    opt_mode: str,
    runtime: AlignmentRuntimeContext,
    objective: PoseObjectiveBundle,
    setup: _AlignSetupState,
    motion_ctx: PoseMotionContext,
    motion_model: Any,
    constraint_ctx: PoseConstraintContext,
    motion_loss_and_grad: Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
) -> AlignmentStepContext:
    return AlignmentStepContext(
        cfg=cfg,
        opt_mode=opt_mode,
        active_loss_name=runtime.active_loss_name,
        ls_like=runtime.supports_gauss_newton,
        align_loss=objective.align_loss,
        align_loss_jit=objective.align_loss_jit,
        loss_and_grad_manual=objective.loss_and_grad_manual,
        gn_update_all=objective.gn_update_all,
        active_mask=setup.active_mask,
        active_coeff_indices=motion_ctx.active_coeff_indices,
        active_col_indices_np=setup.active_col_indices_np,
        frozen_params5=setup.frozen_params5,
        bounds_lower=constraint_ctx.bounds_lower,
        bounds_upper=constraint_ctx.bounds_upper,
        apply_full_constraints=constraint_ctx.apply_full_constraints,
        apply_full_constraints_with_stats=constraint_ctx.apply_full_constraints_with_stats,
        project_params_to_smooth=motion_ctx.project_params_to_smooth,
        coeffs_to_constrained_params=motion_ctx.coeffs_to_constrained_params,
        use_smooth_pose_model=motion_ctx.use_smooth_pose_model,
        motion_loss_and_grad=motion_loss_and_grad,
        motion_model=motion_model,
        smoothness_gram=runtime.smoothness_gram,
        light_smoothness_weights_sq=runtime.light_smoothness_weights_sq,
        medium_smoothness_weights_sq=runtime.medium_smoothness_weights_sq,
        smoothness_weights_sq=runtime.smoothness_weights_sq,
        trans_only_smoothness_weights_sq=runtime.trans_only_smoothness_weights_sq,
        gauge_fix=constraint_ctx.gauge_fix,
        gauge_dofs=tuple(constraint_ctx.gauge_dofs),
    )


@dataclass
class _AlignLoopState:
    x: jnp.ndarray
    params5: jnp.ndarray
    motion_coeffs: jnp.ndarray | None
    final_gauge_stats: dict[str, float | str | list[str]]
    l_prev: float | None
    small_impr_streak: int
    loss_hist: list[float]
    outer_stats: list[OuterStat]
    stopped_by_observer: bool
    observer_action: ObserverAction


def _emit_alignment_checkpoint(
    *,
    checkpoint_callback: AlignCheckpointCallback | None,
    state: _AlignLoopState,
    wall_start: float,
) -> None:
    if checkpoint_callback is None:
        return
    checkpoint_callback(
        AlignResumeState(
            x=state.x,
            params5=state.params5,
            motion_coeffs=state.motion_coeffs,
            start_outer_iter=len(state.outer_stats),
            loss=list(state.loss_hist),
            outer_stats=[dict(stat) for stat in state.outer_stats],
            L=(float(state.l_prev) if state.l_prev is not None else None),
            small_impr_streak=int(state.small_impr_streak),
            elapsed_offset=float(time.perf_counter() - wall_start),
        )
    )


def _initial_outer_stat(
    *,
    outer_idx: int,
    active_loss_name: str,
    recon_algo: str,
    objective_provenance: Mapping[str, object],
    backend_provenance: Mapping[str, object],
    cfg: AlignConfig,
    profile_policy: Mapping[str, object],
) -> OuterStat:
    return {
        "outer_idx": outer_idx,
        "loss_kind": active_loss_name,
        "recon_algo": recon_algo,
        "objective_kind": "fixed_volume",
        "objective_provenance": dict(objective_provenance),
        "backend_provenance": dict(backend_provenance),
        "outer_loss_kind": active_loss_name,
        "align_profile": str(cfg.align_profile),
        "quality_tier": str(cfg.quality_tier),
        "fallback_policy": str(cfg.fallback_policy),
        "profile_policy": dict(profile_policy),
    }


def _observer_break_decision(
    *,
    observer_fn: ObserverCallback | None,
    state: _AlignLoopState,
    stat: OuterStat,
) -> bool:
    if observer_fn is None:
        return False
    observer_action = _normalize_observer_action(observer_fn(state.x, state.params5, dict(stat)))
    state.observer_action = observer_action
    stat["observer_action"] = observer_action
    stat["observer_stop"] = observer_action != "continue"
    if observer_action == "continue":
        return False
    state.stopped_by_observer = observer_action == "stop_run"
    return True


def _early_stop_break_decision(
    *,
    cfg: AlignConfig,
    state: _AlignLoopState,
    stat: OuterStat,
    rel_impr: float | None,
    outer_idx: int,
) -> bool:
    if not cfg.early_stop:
        return False
    if rel_impr is None:
        state.small_impr_streak = 0
        return False
    rel_for_patience = rel_impr
    if (not math.isfinite(rel_for_patience)) or (rel_for_patience < 0.0):
        rel_for_patience = 0.0
    if rel_for_patience < float(cfg.early_stop_rel_impr):
        state.small_impr_streak += 1
    else:
        state.small_impr_streak = 0
    if state.small_impr_streak < int(cfg.early_stop_patience):
        return False
    if cfg.log_summary:
        logging.info(
            "Early stop after %d outer iters (%s elapsed): "
            "rel_impr=%.3e < %.3e for %d consecutive outers",
            outer_idx,
            format_duration(stat.get("cumulative_time")),
            float(rel_impr),
            float(cfg.early_stop_rel_impr),
            int(cfg.early_stop_patience),
        )
    return True


def _run_align_outer_iteration(
    *,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    cfg: AlignConfig,
    state: _AlignLoopState,
    step_ctx: AlignmentStepContext,
    outer_idx: int,
    recon_algo: str,
    active_loss_name: str,
    objective_provenance: Mapping[str, object],
    backend_provenance: Mapping[str, object],
    profile_policy: Mapping[str, object],
    wall_start: float,
) -> tuple[OuterStat, float | None]:
    stat = _initial_outer_stat(
        outer_idx=outer_idx,
        active_loss_name=active_loss_name,
        recon_algo=recon_algo,
        objective_provenance=objective_provenance,
        backend_provenance=backend_provenance,
        cfg=cfg,
        profile_policy=profile_policy,
    )
    outer_start = time.perf_counter()
    state.x, state.l_prev, recon_stat = _run_reconstruction_step(
        geometry=geometry,
        grid=grid,
        detector=detector,
        projections=projections,
        det_grid=det_grid,
        params5=state.params5,
        x=state.x,
        cfg=cfg,
        L_prev=state.l_prev,
        outer_idx=outer_idx,
        recon_algo=recon_algo,
    )
    stat.update(recon_stat)
    if bool(stat.get("reconstruction_failed", False)):
        stat["outer_time"] = time.perf_counter() - outer_start
        stat["cumulative_time"] = time.perf_counter() - wall_start
        state.outer_stats.append(stat)
        return stat, None
    (
        state.params5,
        state.motion_coeffs,
        state.final_gauge_stats,
        total_loss,
        rel_impr,
        align_stat,
    ) = _run_alignment_step(
        step_ctx,
        state.params5,
        state.motion_coeffs,
        state.x,
        loss_hist=state.loss_hist,
    )
    state.loss_hist.append(total_loss)
    stat.update(align_stat)
    stat["outer_time"] = time.perf_counter() - outer_start
    stat["cumulative_time"] = time.perf_counter() - wall_start
    state.outer_stats.append(stat)
    return stat, rel_impr


def _log_align_outer_summary(
    *,
    stat: OuterStat,
    cfg: AlignConfig,
    recon_algo: str,
) -> None:
    for line in _format_outer_summary_lines(stat, cfg=cfg, recon_algo=recon_algo):
        logging.info(line)


def _log_alignment_completion(
    *,
    cfg: AlignConfig,
    outer_stats: list[OuterStat],
    wall_start: float,
) -> None:
    if not (cfg.log_summary and outer_stats):
        return
    recon_total = sum(
        float(s.get("recon_time", 0.0)) for s in outer_stats if s.get("recon_time") is not None
    )
    align_total = sum(
        float(s.get("align_time", 0.0)) for s in outer_stats if s.get("align_time") is not None
    )
    wall_total = time.perf_counter() - wall_start
    logging.info(
        "Alignment completed in %s (recon %s, align %s over %d outer iters)",
        format_duration(wall_total),
        format_duration(recon_total),
        format_duration(align_total),
        len(outer_stats),
    )
    first_loss = outer_stats[0].get("loss_before")
    final_loss = outer_stats[-1].get("loss_after")
    if (first_loss is not None) and (final_loss is not None):
        total_delta = final_loss - first_loss
        rel_pct = (total_delta / first_loss) * 100.0 if abs(first_loss) > 1e-12 else None
        rel_str = f", {rel_pct:+.2f}%" if rel_pct is not None else ""
        logging.info(
            "  Loss %s -> %s (Δ %s%s)",
            f"{first_loss:.3e}",
            f"{final_loss:.3e}",
            f"{total_delta:+.3e}",
            rel_str,
        )
    best_loss = min(
        (s.get("loss_after") for s in outer_stats if s.get("loss_after") is not None),
        default=None,
    )
    if best_loss is not None and final_loss is not None and best_loss < final_loss:
        logging.info("  Best loss observed: %.3e", best_loss)


def _final_pose_align_info(
    *,
    state: _AlignLoopState,
    active_loss_name: str,
    recon_algo: str,
    l_prev: float | None,
    wall_total: float,
    profile_policy: Mapping[str, object],
    cfg: AlignConfig,
    motion_model: Any,
    constraint_ctx: PoseConstraintContext,
) -> AlignInfo:
    outer_stats = state.outer_stats
    return {
        "loss": state.loss_hist,
        "loss_kind": active_loss_name,
        "recon_algo": recon_algo,
        "L": (float(l_prev) if l_prev is not None else None),
        "outer_stats": outer_stats,
        "stopped_by_observer": state.stopped_by_observer,
        "observer_action": state.observer_action,
        "wall_time_total": float(wall_total),
        "align_profile": str(cfg.align_profile),
        "profile_policy": dict(profile_policy),
        "quality_tier": str(cfg.quality_tier),
        "fallback_policy": str(cfg.fallback_policy),
        "pose_model": motion_model.name,
        "pose_model_variables": int(motion_model.variable_count),
        "per_view_variables": int(motion_model.per_view_variable_count),
        "pose_model_basis_shape": [
            int(motion_model.basis.shape[0]),
            int(motion_model.basis.shape[1]),
        ],
        "active_dofs": list(motion_model.active_names),
        "active_pose_dofs": list(motion_model.active_names),
        "active_geometry_dofs": [],
        "objective_kind": "fixed_volume",
        "objective_kinds": ["fixed_volume"] if outer_stats else [],
        "objective_provenance": (
            dict(outer_stats[-1].get("objective_provenance", {}))
            if outer_stats and isinstance(outer_stats[-1].get("objective_provenance"), Mapping)
            else None
        ),
        "backend_provenance": (
            dict(outer_stats[-1].get("backend_provenance", {}))
            if outer_stats and isinstance(outer_stats[-1].get("backend_provenance"), Mapping)
            else None
        ),
        "optimizer_kind": str(outer_stats[-1].get("optimizer_kind"))
        if outer_stats and outer_stats[-1].get("optimizer_kind") is not None
        else str(cfg.opt_method),
        "completed_outer_iters": len(outer_stats),
        "small_impr_streak": int(state.small_impr_streak),
        "motion_coeffs": state.motion_coeffs,
        "gauge_fix": constraint_ctx.gauge_fix,
        "gauge_fix_dofs": list(constraint_ctx.gauge_dofs),
        "gauge_fix_final": dict(state.final_gauge_stats),
    }


def align(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,  # (n_views, nv, nu)
    *,
    cfg: AlignConfig | None = None,
    init_x: jnp.ndarray | None = None,
    init_params5: jnp.ndarray | None = None,
    observer: ObserverCallback | None = None,
    resume_state: AlignResumeState | None = None,
    checkpoint_callback: AlignCheckpointCallback | None = None,
    det_grid_override: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, AlignInfo]:
    """Alternating reconstruction + per-view alignment (5-DOF) on small cases.

    Returns (x, params5, info) with loss history and optional metrics.
    """
    setup = _prepare_align_setup(
        geometry,
        grid,
        detector,
        projections,
        cfg=cfg,
        init_x=init_x,
        init_params5=init_params5,
        observer=observer,
        resume_state=resume_state,
    )
    cfg = setup.cfg
    observer_fn = setup.observer_fn
    n_views = setup.n_views
    x = setup.x
    params5 = setup.params5
    active_names = setup.active_names
    active_mask = setup.active_mask
    constraint_ctx = PoseConstraintContext.from_setup(setup)

    params5, initial_gauge_stats = constraint_ctx.apply_full_constraints_with_stats(params5)
    logging.info("Alignment gauge fix: %s", constraint_ctx.description())
    final_gauge_stats = dict(initial_gauge_stats)

    motion_ctx = PoseMotionContext.build(
        geometry=geometry,
        cfg=cfg,
        n_views=n_views,
        active_names=active_names,
        params5=params5,
        resume_state=resume_state,
        constraint_ctx=constraint_ctx,
    )
    motion_model = motion_ctx.motion_model
    params5 = motion_ctx.params5
    motion_coeffs = motion_ctx.motion_coeffs
    if motion_ctx.use_smooth_pose_model:
        _, initial_gauge_stats = constraint_ctx.apply_full_constraints_with_stats(params5)
        final_gauge_stats = dict(initial_gauge_stats)

    start_outer_iter = int(resume_state.start_outer_iter) if resume_state is not None else 0
    if start_outer_iter < 0 or start_outer_iter > int(cfg.outer_iters):
        raise ValueError(
            "align resume_state start_outer_iter must be between 0 and cfg.outer_iters; "
            f"got {start_outer_iter} for outer_iters={int(cfg.outer_iters)}"
        )
    runtime = _build_alignment_runtime_context(
        geometry=geometry,
        grid=grid,
        detector=detector,
        projections=projections,
        cfg=cfg,
        n_views=n_views,
        active_mask=active_mask,
        det_grid_override=det_grid_override,
    )

    objective = _build_pose_objective_bundle(
        geometry=geometry,
        grid=grid,
        detector=detector,
        projections=projections,
        cfg=cfg,
        n_views=n_views,
        active_mask=active_mask,
        runtime=runtime,
    )

    motion_loss_and_grad = motion_ctx.loss_and_grad_for(objective.align_loss)

    opt_mode = str(cfg.opt_method).lower()
    elapsed_offset = float(resume_state.elapsed_offset) if resume_state is not None else 0.0
    wall_start = time.perf_counter() - elapsed_offset
    recon_algo = str(cfg.recon_algo)
    loop_state = _AlignLoopState(
        x=x,
        params5=params5,
        motion_coeffs=motion_coeffs,
        final_gauge_stats=final_gauge_stats,
        l_prev=resume_state.L if resume_state is not None else cfg.recon_L,
        small_impr_streak=int(resume_state.small_impr_streak) if resume_state is not None else 0,
        loss_hist=list(resume_state.loss) if resume_state is not None else [],
        outer_stats=(
            [dict(stat) for stat in resume_state.outer_stats] if resume_state is not None else []
        ),
        stopped_by_observer=False,
        observer_action="continue",
    )

    step_ctx = _build_alignment_step_context(
        cfg=cfg,
        opt_mode=opt_mode,
        runtime=runtime,
        objective=objective,
        setup=setup,
        motion_ctx=motion_ctx,
        motion_loss_and_grad=motion_loss_and_grad,
        motion_model=motion_model,
        constraint_ctx=constraint_ctx,
    )

    backend_provenance = alignment_projector_backend_provenance(
        pose_stack=runtime.pose_stack,
        grid=grid,
        detector=detector,
        volume=loop_state.x,
        det_grid=runtime.det_grid,
        projector_backend=cfg.projector_backend,
        require_differentiable_projector=True,
        api_surface="alignment.pose_objective",
        gather_dtype=cfg.gather_dtype,
    ).to_dict()
    profile_policy = profile_policy_from_config(cfg).to_dict()

    iter_range = range(start_outer_iter, int(cfg.outer_iters))
    for it in progress_iter(
        iter_range,
        total=max(0, int(cfg.outer_iters) - start_outer_iter),
        desc="Align: outer iters",
    ):
        outer_idx = it + 1
        stat, rel_impr = _run_align_outer_iteration(
            geometry=geometry,
            grid=grid,
            detector=detector,
            projections=projections,
            det_grid=runtime.det_grid,
            cfg=cfg,
            state=loop_state,
            step_ctx=step_ctx,
            outer_idx=outer_idx,
            recon_algo=recon_algo,
            active_loss_name=runtime.active_loss_name,
            objective_provenance=runtime.objective_provenance,
            backend_provenance=backend_provenance,
            profile_policy=profile_policy,
            wall_start=wall_start,
        )
        if cfg.log_summary:
            _log_align_outer_summary(stat=stat, cfg=cfg, recon_algo=recon_algo)
        should_break = _observer_break_decision(
            observer_fn=observer_fn,
            state=loop_state,
            stat=stat,
        )
        should_break = should_break or bool(stat.get("reconstruction_failed", False))
        should_break = should_break or _early_stop_break_decision(
            cfg=cfg,
            state=loop_state,
            stat=stat,
            rel_impr=rel_impr,
            outer_idx=outer_idx,
        )
        _emit_alignment_checkpoint(
            checkpoint_callback=checkpoint_callback,
            state=loop_state,
            wall_start=wall_start,
        )
        if should_break:
            break

    _log_alignment_completion(cfg=cfg, outer_stats=loop_state.outer_stats, wall_start=wall_start)
    wall_total = time.perf_counter() - wall_start
    info = _final_pose_align_info(
        state=loop_state,
        active_loss_name=runtime.active_loss_name,
        recon_algo=recon_algo,
        l_prev=loop_state.l_prev,
        wall_total=wall_total,
        profile_policy=profile_policy,
        cfg=cfg,
        motion_model=motion_model,
        constraint_ctx=constraint_ctx,
    )
    return loop_state.x, loop_state.params5, info


__all__ = [
    "_evaluate_align_loss",
    "_is_expected_align_eval_failure",
    "_second_difference_gram",
    "_select_gn_candidate",
    "_should_prefer_gn_candidate",
    "_smooth_gn_candidate",
    "align",
]
