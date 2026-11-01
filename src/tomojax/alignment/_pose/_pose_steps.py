from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
import math
import time

import jax
import jax.numpy as jnp

from tomojax.alignment._config import AlignConfig
from tomojax.alignment._model.dofs import DOF_INDEX
from tomojax.alignment._objectives.loss_specs import loss_is_within_relative_tolerance
from tomojax.alignment._observer import OuterStat

from ._coupled_objective import CoupledObjective, VolumeHandle
from ._pose_block import feasible_pose_scale


@dataclass(frozen=True)
class AlignmentStepObjective:
    coupled: CoupledObjective
    private: bool = False


@dataclass(frozen=True)
class AlignmentStepConstraints:
    active_mask: jnp.ndarray
    bounds_lower: jnp.ndarray
    bounds_upper: jnp.ndarray
    apply_full_constraints_with_stats: Callable[
        [jnp.ndarray],
        tuple[jnp.ndarray, dict[str, float | str | list[str]]],
    ]


@dataclass(frozen=True)
class AlignmentStepGauge:
    gauge_fix: str
    gauge_dofs: tuple[str, ...]


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
    maximum_scale = 1.0
    if cfg.pose_model != "per_view":
        dp = jnp.asarray(dp)
        gauge_dofs = (
            (*gauge.gauge_dofs, "dy")
            if constraints.active_mask[5] and "dy" not in gauge.gauge_dofs
            else gauge.gauge_dofs
        )
        for name in gauge_dofs:
            column = DOF_INDEX[name]
            dp = dp.at[:, column].add(-jnp.mean(dp[:, column]))
        maximum_scale = float(
            feasible_pose_scale(
                pose_params_in, dp, constraints.bounds_lower, constraints.bounds_upper
            )
        )
    volume_out, params_out, after, scale = None, pose_params_in, before, 0.0
    if bool(result.finite) and math.isfinite(before):
        for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
            trial_scale = fraction * maximum_scale
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
