from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
import math
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.align._config import AlignConfig
from tomojax.align._model.motion_models import expand_motion_coefficients, fit_motion_coefficients
from tomojax.align._objectives.loss_specs import loss_is_within_relative_tolerance
from tomojax.align._observer import OuterStat
from tomojax.align._results import _set_float_stat
from tomojax.align.optimizers import PoseLbfgsConfig, PoseOptimizationContext, run_pose_lbfgs

from ._pose_candidates import (
    GNCandidateContext,
    _evaluate_align_loss,
    _is_expected_align_eval_failure,
    _select_gn_candidate,
    _smooth_gn_candidate,
)


@dataclass(frozen=True)
class AlignmentStepOptimizer:
    opt_mode: str


@dataclass(frozen=True)
class AlignmentStepObjective:
    active_loss_name: str
    ls_like: bool
    align_loss: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    align_loss_jit: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    loss_and_grad_manual: Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]
    gn_update_all: Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]


@dataclass(frozen=True)
class AlignmentStepConstraints:
    active_mask: jnp.ndarray
    active_col_indices_np: np.ndarray
    frozen_params5: jnp.ndarray
    bounds_lower: jnp.ndarray
    bounds_upper: jnp.ndarray
    apply_full_constraints: Callable[[jnp.ndarray], jnp.ndarray]
    apply_full_constraints_with_stats: Callable[
        [jnp.ndarray],
        tuple[jnp.ndarray, dict[str, float | str | list[str]]],
    ]


@dataclass(frozen=True)
class AlignmentStepMotion:
    active_coeff_indices: jnp.ndarray
    project_params_to_smooth: Callable[[jnp.ndarray], jnp.ndarray]
    coeffs_to_constrained_params: Callable[[jnp.ndarray], jnp.ndarray]
    use_smooth_pose_model: bool
    motion_loss_and_grad: (
        Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]] | None
    )
    motion_model: Any


@dataclass(frozen=True)
class AlignmentStepSmoothing:
    smoothness_gram: jnp.ndarray
    light_smoothness_weights_sq: jnp.ndarray
    medium_smoothness_weights_sq: jnp.ndarray
    smoothness_weights_sq: jnp.ndarray
    trans_only_smoothness_weights_sq: jnp.ndarray


@dataclass(frozen=True)
class AlignmentStepGauge:
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


def _alignment_step_kind(
    optimizer: AlignmentStepOptimizer,
    objective: AlignmentStepObjective,
) -> tuple[str, OuterStat]:
    if optimizer.opt_mode == "gn" and objective.ls_like:
        return "gn", {}
    if optimizer.opt_mode == "gn":
        logging.warning(
            "Gauss-Newton is incompatible with loss=%s; falling back to GD for this step",
            objective.active_loss_name,
        )
        return "gd", {"optimizer_fallback": "gn->gd"}
    if optimizer.opt_mode == "lbfgs":
        return "lbfgs", {}
    return "gd", {}


def _pre_alignment_step_loss(
    objective: AlignmentStepObjective,
    motion: AlignmentStepMotion,
    *,
    params5: jnp.ndarray,
    vol: jnp.ndarray,
    step_kind: str,
) -> float | None:
    if step_kind == "gn" and not motion.use_smooth_pose_model:
        return None
    return _evaluate_align_loss(
        lambda: objective.align_loss_jit(params5, vol),
        fallback=None,
        context="Skipping pre-step alignment loss evaluation",
    )


def _select_gd_step_candidate(
    objective: AlignmentStepObjective,
    *,
    base_params: jnp.ndarray,
    doubled_params: jnp.ndarray,
    previous_params: jnp.ndarray,
    loss_before_value: float | None,
    vol: jnp.ndarray,
) -> tuple[jnp.ndarray, float | None]:
    base_loss = _evaluate_align_loss(
        lambda: objective.align_loss_jit(base_params, vol),
        fallback=math.inf,
        context="Treating GD base candidate as rejected during alignment loss evaluation",
    )
    doubled_loss = _evaluate_align_loss(
        lambda: objective.align_loss_jit(doubled_params, vol),
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
    cfg: AlignConfig,
    objective: AlignmentStepObjective,
    constraints: AlignmentStepConstraints,
    motion: AlignmentStepMotion,
    params5_in: jnp.ndarray,
    motion_coeffs_in: jnp.ndarray | None,
    vol: jnp.ndarray,
    loss_before_value: float | None,
) -> tuple[jnp.ndarray, jnp.ndarray | None, float | None, jnp.ndarray]:
    scales = jnp.array(
        [cfg.lr_rot, cfg.lr_rot, cfg.lr_rot, cfg.lr_trans, cfg.lr_trans],
        dtype=jnp.float32,
    )
    if motion.use_smooth_pose_model:
        if motion_coeffs_in is None or motion.motion_loss_and_grad is None:
            raise RuntimeError("smooth pose model coefficients were not initialized")
        coeffs_in = motion_coeffs_in
        _, g_coeffs = motion.motion_loss_and_grad(coeffs_in, vol)
        active_scales = scales[motion.active_coeff_indices]
        rms_active = jnp.sqrt(jnp.mean(jnp.square(g_coeffs), axis=0)) + 1e-6
        eff_scales = active_scales / rms_active
        best_coeffs = coeffs_in - g_coeffs * eff_scales[None, :]
        best_params = motion.coeffs_to_constrained_params(best_coeffs)
        cand_coeffs = coeffs_in - 2.0 * g_coeffs * eff_scales[None, :]
        cand_params = motion.coeffs_to_constrained_params(cand_coeffs)
        params5_out, loss_after_value = _select_gd_step_candidate(
            objective,
            base_params=best_params,
            doubled_params=cand_params,
            previous_params=motion.coeffs_to_constrained_params(coeffs_in),
            loss_before_value=loss_before_value,
            vol=vol,
        )
        motion_coeffs_out = fit_motion_coefficients(motion.motion_model, params5_out)
        params5_out = motion.coeffs_to_constrained_params(motion_coeffs_out)
        rms = jnp.zeros((5,), dtype=jnp.float32).at[motion.active_coeff_indices].set(rms_active)
        return params5_out, motion_coeffs_out, loss_after_value, rms

    _, g_params = objective.loss_and_grad_manual(params5_in, vol)
    g_params = g_params * constraints.active_mask
    rms = jnp.sqrt(jnp.mean(jnp.square(g_params), axis=0)) + 1e-6
    eff_scales = scales / rms
    best_params = constraints.apply_full_constraints(params5_in - g_params * eff_scales)
    cand_params = constraints.apply_full_constraints(params5_in - 2.0 * g_params * eff_scales)
    params5_out, loss_after_value = _select_gd_step_candidate(
        objective,
        base_params=best_params,
        doubled_params=cand_params,
        previous_params=params5_in,
        loss_before_value=loss_before_value,
        vol=vol,
    )
    return params5_out, motion_coeffs_in, loss_after_value, rms


def _run_lbfgs_alignment_step(
    cfg: AlignConfig,
    objective: AlignmentStepObjective,
    constraints: AlignmentStepConstraints,
    motion: AlignmentStepMotion,
    params5_in: jnp.ndarray,
    motion_coeffs_in: jnp.ndarray | None,
    vol: jnp.ndarray,
    loss_before_value: float | None,
) -> tuple[jnp.ndarray, jnp.ndarray | None, float | None, OuterStat]:
    result = run_pose_lbfgs(
        params5_in=params5_in,
        motion_coeffs_in=motion_coeffs_in,
        loss_before_value=loss_before_value,
        objective_fn=lambda candidate: objective.align_loss(candidate, vol),
        eval_loss_fn=lambda candidate, label: _evaluate_align_loss(
            lambda: objective.align_loss_jit(candidate, vol),
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
            active_cols=constraints.active_col_indices_np,
            frozen_params5=constraints.frozen_params5,
            bounds_lower=constraints.bounds_lower,
            bounds_upper=constraints.bounds_upper,
            apply_param_constraints=constraints.apply_full_constraints,
            motion_model=motion.motion_model if motion.use_smooth_pose_model else None,
        ),
    )
    if result.stats.get("lbfgs_fallback_to_gd"):
        logging.warning(
            "%s; falling back to GD for this alignment step",
            result.stats.get("lbfgs_message"),
        )
    return result.params5, result.motion_coeffs, result.loss, result.stats


def _gn_smooth_candidate_fn(
    smoothing: AlignmentStepSmoothing,
    constrain_candidate: Callable[[jnp.ndarray], jnp.ndarray],
    params5_in: jnp.ndarray,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None:
    if int(params5_in.shape[0]) < 3:
        return None

    def smooth_candidate(candidate: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        return _smooth_gn_candidate(
            constrain_candidate(candidate),
            smoothing.smoothness_gram,
            weights,
        )

    return smooth_candidate


def _run_gn_alignment_step(
    cfg: AlignConfig,
    objective: AlignmentStepObjective,
    constraints: AlignmentStepConstraints,
    motion: AlignmentStepMotion,
    smoothing: AlignmentStepSmoothing,
    params5_in: jnp.ndarray,
    motion_coeffs_in: jnp.ndarray | None,
    vol: jnp.ndarray,
    loss_before_value: float | None,
) -> tuple[jnp.ndarray, jnp.ndarray | None, float | None, float | None, OuterStat]:
    params5_prev = params5_in
    dp_raw, gn_loss_before = objective.gn_update_all(params5_prev, vol)
    dp_all = dp_raw * constraints.active_mask
    loss_before = loss_before_value
    if not motion.use_smooth_pose_model:
        loss_before = float(gn_loss_before)
    constrain_candidate = (
        motion.project_params_to_smooth
        if motion.use_smooth_pose_model
        else constraints.apply_full_constraints
    )
    if cfg.gn_accept_only_improving and (loss_before is not None):
        params5_out, loss_after = _select_gn_candidate(
            params5_prev,
            dp_all,
            loss_before=loss_before,
            eval_loss=lambda candidate: float(
                _evaluate_align_loss(
                    lambda: objective.align_loss_jit(candidate, vol),
                    fallback=math.inf,
                    context="Treating GN candidate as rejected during alignment loss evaluation",
                )
            ),
            gn_accept_tol=cfg.gn_accept_tol,
            candidate_context=GNCandidateContext(
                constrain_candidate=constrain_candidate,
                smooth_candidate=_gn_smooth_candidate_fn(
                    smoothing,
                    constrain_candidate,
                    params5_in,
                ),
                smoothing_weights=(
                    smoothing.light_smoothness_weights_sq,
                    smoothing.medium_smoothness_weights_sq,
                    smoothing.smoothness_weights_sq,
                    smoothing.trans_only_smoothness_weights_sq,
                ),
            ),
        )
        params5_out = constrain_candidate(params5_out)
    else:
        params5_out = constrain_candidate(params5_prev + dp_all)
        candidate_loss = _evaluate_align_loss(
            lambda: objective.align_loss_jit(params5_out, vol),
            fallback=math.inf,
            context="Treating GN step as rejected during alignment loss evaluation",
        )
        if candidate_loss is not None and math.isfinite(candidate_loss):
            loss_after = candidate_loss
        else:
            params5_out = params5_prev
            loss_after = loss_before
    motion_coeffs_out = motion_coeffs_in
    if motion.use_smooth_pose_model:
        motion_coeffs_out = fit_motion_coefficients(motion.motion_model, params5_out)
        params5_out = motion.coeffs_to_constrained_params(motion_coeffs_out)
    stat: OuterStat = {}
    _set_float_stat(stat, "rot_mean", jnp.mean(jnp.abs(dp_all[:, :3])))
    _set_float_stat(stat, "trans_mean", jnp.mean(jnp.abs(dp_all[:, 3:])))
    return params5_out, motion_coeffs_out, loss_before, loss_after, stat


def _run_alignment_step_core(
    *,
    cfg: AlignConfig,
    optimizer: AlignmentStepOptimizer,
    objective: AlignmentStepObjective,
    constraints: AlignmentStepConstraints,
    motion: AlignmentStepMotion,
    smoothing: AlignmentStepSmoothing,
    params5_in: jnp.ndarray,
    motion_coeffs_in: jnp.ndarray | None,
    vol: jnp.ndarray,
) -> _AlignmentStepCoreResult:
    step_kind, stat = _alignment_step_kind(optimizer, objective)
    loss_before = _pre_alignment_step_loss(
        objective,
        motion,
        params5=params5_in,
        vol=vol,
        step_kind=step_kind,
    )
    if step_kind == "gn":
        params5_out, motion_coeffs_out, loss_before, loss_after, gn_stat = _run_gn_alignment_step(
            cfg,
            objective,
            constraints,
            motion,
            smoothing,
            params5_in,
            motion_coeffs_in,
            vol,
            loss_before,
        )
        stat.update(gn_stat)
    elif step_kind == "lbfgs":
        params5_out, motion_coeffs_out, loss_after, lbfgs_stats = _run_lbfgs_alignment_step(
            cfg,
            objective,
            constraints,
            motion,
            params5_in,
            motion_coeffs_in,
            vol,
            loss_before,
        )
        stat.update(lbfgs_stats)
        if stat.get("lbfgs_fallback_to_gd"):
            step_kind = "gd"
            params5_out, motion_coeffs_out, loss_after, rms = _run_gd_alignment_step(
                cfg,
                objective,
                constraints,
                motion,
                params5_out,
                motion_coeffs_out,
                vol,
                loss_before,
            )
            _set_float_stat(stat, "rot_rms", jnp.mean(rms[:3]))
            _set_float_stat(stat, "trans_rms", jnp.mean(rms[3:]))
    else:
        params5_out, motion_coeffs_out, loss_after, rms = _run_gd_alignment_step(
            cfg,
            objective,
            constraints,
            motion,
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
    constraints: AlignmentStepConstraints,
    motion: AlignmentStepMotion,
    params5: jnp.ndarray,
    motion_coeffs: jnp.ndarray | None,
) -> tuple[jnp.ndarray, jnp.ndarray | None, dict[str, float | str | list[str]]]:
    params5, gauge_stats = constraints.apply_full_constraints_with_stats(params5)
    motion_coeffs_out = motion_coeffs
    if motion.use_smooth_pose_model:
        motion_coeffs_out = fit_motion_coefficients(motion.motion_model, params5)
        params5, gauge_stats = constraints.apply_full_constraints_with_stats(
            expand_motion_coefficients(motion.motion_model, motion_coeffs_out)
        )
        motion_coeffs_out = fit_motion_coefficients(motion.motion_model, params5)
    return params5, motion_coeffs_out, gauge_stats


def _record_gauge_stats(
    stat: OuterStat,
    gauge: AlignmentStepGauge,
    gauge_stats: dict[str, float | str | list[str]],
) -> None:
    stat["gauge_fix"] = gauge.gauge_fix
    stat["gauge_fix_dofs"] = ",".join(gauge.gauge_dofs)
    if gauge.gauge_fix == "mean_translation":
        stat["dx_mean_before_gauge"] = float(gauge_stats["dx_mean_before"])
        stat["dz_mean_before_gauge"] = float(gauge_stats["dz_mean_before"])
        stat["dx_mean_after_gauge"] = float(gauge_stats["dx_mean_after"])
        stat["dz_mean_after_gauge"] = float(gauge_stats["dz_mean_after"])


def _alignment_total_loss(
    objective: AlignmentStepObjective,
    motion: AlignmentStepMotion,
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
        and not motion.use_smooth_pose_model
        and loss_after is not None
        and math.isfinite(float(loss_after))
    )
    if can_reuse_validated_gn_loss:
        stat["loss_after_reused"] = True
        return float(loss_after)
    total_loss_eval = _evaluate_align_loss(
        lambda: objective.align_loss_jit(params5, vol),
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
    *,
    cfg: AlignConfig,
    optimizer: AlignmentStepOptimizer,
    objective: AlignmentStepObjective,
    constraints: AlignmentStepConstraints,
    motion: AlignmentStepMotion,
    smoothing: AlignmentStepSmoothing,
    gauge: AlignmentStepGauge,
    params5_in: jnp.ndarray,
    motion_coeffs_in: jnp.ndarray | None,
    vol: jnp.ndarray,
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
    result = _run_alignment_step_core(
        cfg=cfg,
        optimizer=optimizer,
        objective=objective,
        constraints=constraints,
        motion=motion,
        smoothing=smoothing,
        params5_in=params5_in,
        motion_coeffs_in=motion_coeffs_in,
        vol=vol,
    )
    params5_out, motion_coeffs_out, gauge_stats = _apply_final_alignment_constraints(
        constraints,
        motion,
        result.params5,
        result.motion_coeffs,
    )
    stat = result.stat
    stat["loss_before"] = result.loss_before
    stat["step_kind"] = result.step_kind
    stat["optimizer_kind"] = result.step_kind
    stat["loss_after_step"] = result.loss_after
    _record_gauge_stats(stat, gauge, gauge_stats)
    jax.block_until_ready(params5_out)
    stat["align_time"] = time.perf_counter() - align_start
    total_loss = _alignment_total_loss(
        objective,
        motion,
        params5=params5_out,
        vol=vol,
        loss_before=result.loss_before,
        loss_after=result.loss_after,
        step_kind=result.step_kind,
        loss_hist=loss_hist,
        stat=stat,
    )
    if cfg.gn_accept_only_improving and _should_reject_post_constraint_loss(
        loss_before=result.loss_before,
        total_loss=total_loss,
        rel_tol=float(cfg.gn_accept_tol),
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
