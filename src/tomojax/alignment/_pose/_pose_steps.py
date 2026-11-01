from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
import math
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.alignment._config import AlignConfig
from tomojax.alignment._model.motion_models import (
    expand_motion_coefficients,
    fit_motion_coefficients,
)
from tomojax.alignment._objectives.loss_specs import loss_is_within_relative_tolerance
from tomojax.alignment._observer import OuterStat
from tomojax.alignment._results import _set_float_stat

from ._coupled_objective import CoupledObjective, VolumeHandle
from ._pose_block import smooth_pose_candidate
from ._pose_candidates import (
    GNCandidateContext,
    _evaluate_align_loss,
    _select_gn_candidate,
)


@dataclass(frozen=True)
class AlignmentStepObjective:
    align_loss_jit: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    gn_update_all: Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]

    coupled: CoupledObjective | None = None
    private: bool = False


@dataclass(frozen=True)
class AlignmentStepConstraints:
    active_mask: jnp.ndarray
    active_col_indices_np: np.ndarray
    frozen_pose_params: jnp.ndarray
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
    motion_model: Any


@dataclass(frozen=True)
class AlignmentStepSmoothing:
    light_smoothness_weights_sq: jnp.ndarray
    medium_smoothness_weights_sq: jnp.ndarray
    smoothness_weights_sq: jnp.ndarray
    trans_only_smoothness_weights_sq: jnp.ndarray


@dataclass(frozen=True)
class AlignmentStepGauge:
    gauge_fix: str
    gauge_dofs: tuple[str, ...]


def _run_gn_alignment_step(
    objective: AlignmentStepObjective,
    constraints: AlignmentStepConstraints,
    motion: AlignmentStepMotion,
    smoothing: AlignmentStepSmoothing,
    pose_params_in: jnp.ndarray,
    motion_coeffs_in: jnp.ndarray | None,
    vol: jnp.ndarray,
    loss_before_value: float | None,
) -> tuple[jnp.ndarray, jnp.ndarray | None, float | None, float | None, OuterStat]:
    pose_params_prev = pose_params_in
    dp_raw, gn_loss_before = objective.gn_update_all(pose_params_prev, vol)
    dp_all = dp_raw * constraints.active_mask
    loss_before = loss_before_value
    if not motion.use_smooth_pose_model:
        loss_before = float(gn_loss_before)
    constrain_candidate = (
        motion.project_params_to_smooth
        if motion.use_smooth_pose_model
        else constraints.apply_full_constraints
    )
    if loss_before is not None:
        pose_params_out, loss_after = _select_gn_candidate(
            pose_params_prev,
            dp_all,
            loss_before=loss_before,
            eval_loss=lambda candidate: float(
                _evaluate_align_loss(
                    lambda: objective.align_loss_jit(candidate, vol),
                    fallback=math.inf,
                    context="Treating GN candidate as rejected during alignment loss evaluation",
                )
            ),
            gn_accept_tol=0.0,
            candidate_context=GNCandidateContext(
                constrain_candidate=constrain_candidate,
                # Smoothing needs neighbours on both sides of a view.
                smooth_candidate=None
                if int(pose_params_in.shape[0]) < 3
                else lambda candidate, weights: smooth_pose_candidate(
                    constrain_candidate(candidate), weights
                ),
                smoothing_weights=(
                    smoothing.light_smoothness_weights_sq,
                    smoothing.medium_smoothness_weights_sq,
                    smoothing.smoothness_weights_sq,
                    smoothing.trans_only_smoothness_weights_sq,
                ),
            ),
        )
        pose_params_out = constrain_candidate(pose_params_out)
    else:
        pose_params_out = constrain_candidate(pose_params_prev + dp_all)
        candidate_loss = _evaluate_align_loss(
            lambda: objective.align_loss_jit(pose_params_out, vol),
            fallback=math.inf,
            context="Treating GN step as rejected during alignment loss evaluation",
        )
        if candidate_loss is not None and math.isfinite(candidate_loss):
            loss_after = candidate_loss
        else:
            pose_params_out = pose_params_prev
            loss_after = loss_before
    motion_coeffs_out = motion_coeffs_in
    if motion.use_smooth_pose_model:
        motion_coeffs_out = fit_motion_coefficients(motion.motion_model, pose_params_out)
        pose_params_out = motion.coeffs_to_constrained_params(motion_coeffs_out)
    stat: OuterStat = {}
    _set_float_stat(stat, "rot_mean", jnp.mean(jnp.abs(dp_all[:, :3])))
    _set_float_stat(stat, "trans_mean", jnp.mean(jnp.abs(dp_all[:, 3:])))
    return pose_params_out, motion_coeffs_out, loss_before, loss_after, stat


def _apply_final_alignment_constraints(
    constraints: AlignmentStepConstraints,
    motion: AlignmentStepMotion,
    pose_params: jnp.ndarray,
    motion_coeffs: jnp.ndarray | None,
) -> tuple[jnp.ndarray, jnp.ndarray | None, dict[str, float | str | list[str]]]:
    pose_params, gauge_stats = constraints.apply_full_constraints_with_stats(pose_params)
    motion_coeffs_out = motion_coeffs
    if motion.use_smooth_pose_model:
        motion_coeffs_out = fit_motion_coefficients(motion.motion_model, pose_params)
        pose_params, gauge_stats = constraints.apply_full_constraints_with_stats(
            expand_motion_coefficients(motion.motion_model, motion_coeffs_out)
        )
        motion_coeffs_out = fit_motion_coefficients(motion.motion_model, pose_params)
    return pose_params, motion_coeffs_out, gauge_stats


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


def _run_alignment_step(
    *,
    objective: AlignmentStepObjective,
    constraints: AlignmentStepConstraints,
    motion: AlignmentStepMotion,
    smoothing: AlignmentStepSmoothing,
    gauge: AlignmentStepGauge,
    pose_params_in: jnp.ndarray,
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
    # A smooth pose model scores the start itself; plain GN reports its own.
    loss_before = (
        _evaluate_align_loss(
            lambda: objective.align_loss_jit(pose_params_in, vol),
            fallback=None,
            context="Skipping pre-step alignment loss evaluation",
        )
        if motion.use_smooth_pose_model
        else None
    )
    pose_params_out, motion_coeffs_out, loss_before, loss_after, stat = _run_gn_alignment_step(
        objective,
        constraints,
        motion,
        smoothing,
        pose_params_in,
        motion_coeffs_in,
        vol,
        loss_before,
    )
    pose_params_out, motion_coeffs_out, gauge_stats = _apply_final_alignment_constraints(
        constraints, motion, pose_params_out, motion_coeffs_out
    )
    stat["loss_before"] = loss_before
    stat["step_kind"] = "gn"
    stat["optimizer_kind"] = "gn"
    stat["loss_after_step"] = loss_after
    _record_gauge_stats(stat, gauge, gauge_stats)
    jax.block_until_ready(pose_params_out)
    stat["align_time"] = time.perf_counter() - align_start
    # A plain GN step has already scored its constrained candidate; a smooth pose
    # model's refit is scored here.
    if not motion.use_smooth_pose_model and loss_after is not None and math.isfinite(loss_after):
        total_loss, stat["loss_after_reused"] = float(loss_after), True
    else:
        known = (loss_after, loss_before, loss_hist[-1] if loss_hist else None)
        rescored = _evaluate_align_loss(
            lambda: objective.align_loss_jit(pose_params_out, vol),
            fallback=next((value for value in known if value is not None), None),
            context="Using fallback for final alignment loss bookkeeping",
        )
        total_loss = math.nan if rescored is None else float(rescored)
        stat["loss_after_reused"] = False
    if (
        loss_before is not None
        and math.isfinite(loss_before)
        and math.isfinite(total_loss)
        and total_loss > loss_before
    ):
        pose_params_out = pose_params_in
        motion_coeffs_out = motion_coeffs_in
        total_loss = float(loss_before)
        stat["post_constraint_rejected"] = True
        stat["post_constraint_reject_reason"] = "loss_after_constraints_worse_than_before"
    else:
        stat["post_constraint_rejected"] = False
    stat["loss_after"] = total_loss
    rel_impr = _alignment_relative_improvement(
        stat,
        loss_before=loss_before,
        total_loss=total_loss,
    )
    stat["rel_impr"] = rel_impr
    return pose_params_out, motion_coeffs_out, gauge_stats, total_loss, rel_impr, stat


@partial(jax.jit, static_argnames=("nonnegative",))
def _joint_trial_volume(volume, increment, scale, *, nonnegative):
    """Create one candidate buffer, without full-volume scale or clipping temporaries."""
    candidate = volume + scale * increment
    return jnp.maximum(candidate, 0) if nonnegative else candidate


def _run_coupled_alignment_step(
    *,
    cfg: AlignConfig,
    objective: CoupledObjective,
    constraints: AlignmentStepConstraints,
    gauge: AlignmentStepGauge,
    pose_params_in: jnp.ndarray,
    vol: VolumeHandle,
) -> tuple:
    """Accept a constrained volume/pose pair against the true joint objective.

    Damping affects the local increment model only. Neither the line search nor
    the reported objective adds a prior on the current volume or pose values.
    """
    start = time.perf_counter()
    before = float(objective.loss(pose_params_in, vol.get()))
    result = objective.update(pose_params_in, vol)
    dx, dp = result.increment
    volume_out, params_out, after, scale = None, pose_params_in, before, 0.0
    if bool(result.finite) and math.isfinite(before):
        for trial_scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
            trial_volume = _joint_trial_volume(
                vol.get(), dx, trial_scale, nonnegative=cfg.nonnegative
            )
            trial_params, _ = constraints.apply_full_constraints_with_stats(
                pose_params_in + trial_scale * dp
            )
            value = float(objective.loss(trial_params, trial_volume))
            improves = loss_is_within_relative_tolerance(before, value, 0.0)
            if math.isfinite(value) and improves:
                volume_out, params_out, after, scale = (
                    trial_volume,
                    trial_params,
                    value,
                    trial_scale,
                )
                break
            # A rejected image must be released before allocating the next one.
            del trial_volume
    _, gauge_stats = constraints.apply_full_constraints_with_stats(params_out)
    stat: OuterStat = {
        "objective_kind": "joint_volume_pose",
        "step_kind": "gn_joint",
        "optimizer_kind": "gn_joint",
        "loss_before": before,
        "loss_after_step": after,
        "loss_after": after,
        "joint_step_scale": scale,
        "joint_linear_iterations": int(result.iterations),
        "joint_linear_relative_residual": float(result.relative_residual),
        "joint_linear_finite": bool(result.finite),
        "joint_projector_backend": objective.backend,
        "joint_pose_columns_cached": objective.pose_columns_cached,
        "backend_provenance": {
            "requested_backend": cfg.backend,
            "actual_backend": objective.backend,
            "status": "selected" if cfg.backend == objective.backend else "fallback",
            "fallback_reason": None
            if cfg.backend == objective.backend
            else "joint normal equations use the matched JAX operator for this geometry/integrator",
            "api_surface": "alignment.joint_objective",
            "differentiability": "explicit_volume_transpose",
            "pose_jacobian_backend": "jax",
            "eligible_for_speed_claim": cfg.backend == objective.backend,
        },
        "objective_provenance": {
            "outer_loss_source": "AlignmentLossSpec",
            "outer_loss_kind": "weighted_least_squares_plus_volume_and_pose_regularisation",
            "inner_data_term": "joint_linearised_volume_pose",
            "inner_regulariser": "increment_damping",
            "differentiation_mode": "matrix_free_joint_normal_equations",
            "initialization_policy": "current_level_volume_and_poses",
            "validation_split": "none",
        },
        "post_constraint_rejected": scale == 0,
    }
    _record_gauge_stats(stat, gauge, gauge_stats)
    rel_impr = _alignment_relative_improvement(stat, loss_before=before, total_loss=after)
    stat["rel_impr"] = rel_impr
    volume_out = jax.block_until_ready(vol.get()) if volume_out is None else volume_out
    stat["align_time"] = time.perf_counter() - start
    return volume_out, params_out, None, gauge_stats, after, rel_impr, stat
