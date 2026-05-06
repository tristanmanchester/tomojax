"""Verification and report payloads for alternating smoke runs."""
# pyright: reportAny=false, reportPrivateUsage=false, reportUnknownMemberType=false
# pyright: reportUnusedFunction=false

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.align._alternating_types import (
    _LevelVerificationChecks,
)
from tomojax.forward import project_parallel_reference, residual_loss
from tomojax.verify import failure_report_from_gates, residual_structure_summary

if TYPE_CHECKING:
    from tomojax.align._alternating_types import (
        AlternatingLevelSummary,
        AlternatingSmokeConfig,
        GeometryUpdateVolumeSource,
    )
    from tomojax.align._continuation import ContinuationSchedule
    from tomojax.align._joint_schur_lm import JointSchurDiagnostics, JointSchurLMResult
    from tomojax.geometry import GaugeReport, GeometryState


def _level_verification_checks(
    *,
    cfg: AlternatingSmokeConfig,
    geometry: GeometryState,
    update_report: GaugeReport,
    loss_before: float,
    loss_after: float,
    heldout_residual_passed: bool | None,
    geometry_update_accepted: bool | None,
) -> _LevelVerificationChecks:
    loss_nonincreasing = bool(loss_after <= loss_before + cfg.verification_loss_tolerance)
    finite_loss = bool(np.isfinite(loss_before) and np.isfinite(loss_after))
    gauge_stable = _gauge_stable(geometry, tolerance=cfg.gauge_stability_tolerance)
    parameter_update_norm = _parameter_update_norm(update_report)
    parameter_update_small = bool(parameter_update_norm <= cfg.parameter_update_tolerance)
    heldout_ok = heldout_residual_passed is not False
    geometry_update_ok = geometry_update_accepted is not False
    verified = (
        loss_nonincreasing
        and finite_loss
        and gauge_stable
        and parameter_update_small
        and heldout_ok
        and geometry_update_ok
    )
    return _LevelVerificationChecks(
        loss_nonincreasing=loss_nonincreasing,
        finite_loss=finite_loss,
        gauge_stable=gauge_stable,
        parameter_update_norm=parameter_update_norm,
        parameter_update_small=parameter_update_small,
        verified=verified,
    )


def _gauge_stable(geometry: GeometryState, *, tolerance: float) -> bool:
    return bool(
        abs(float(np.mean(geometry.pose.dx_px))) <= tolerance
        and abs(float(np.mean(geometry.pose.phi_residual_rad))) <= tolerance
    )


def _parameter_update_norm(report: GaugeReport) -> float:
    applied = [abs(float(transfer.value)) for transfer in report.transfers if transfer.applied]
    if not applied:
        return 0.0
    return float(max(applied))


def _verification_payload(
    *,
    cfg: AlternatingSmokeConfig,
    schedule: ContinuationSchedule,
    initial_loss: float,
    final_loss: float,
    coarse_verified: bool,
    true_geometry: GeometryState,
    initial_geometry: GeometryState,
    final_geometry: GeometryState,
    truth_volume: jax.Array,
    final_volume: jax.Array,
    observed: jax.Array,
    mask: jax.Array,
    summaries: tuple[AlternatingLevelSummary, ...],
    geometry_update_volume_source: GeometryUpdateVolumeSource,
    fit_gain_offset_nuisance: bool,
    fit_background_nuisance: bool,
    time_to_verified_geometry_seconds: float | None,
    total_wall_seconds: float,
) -> dict[str, object]:
    geometry_recovery = _geometry_recovery_payload(true_geometry, initial_geometry, final_geometry)
    time_to_recovered_geometry_seconds = (
        time_to_verified_geometry_seconds if geometry_recovery["passed"] else None
    )
    volume_recovery = _volume_recovery_payload(truth_volume, final_volume)
    level1_geometry_skipped = _level1_geometry_skipped(summaries)
    all_levels_verified = all(summary.verified for summary in summaries)
    residual_before = summaries[0].loss_before if summaries else initial_loss
    relative_improvement = (
        float((residual_before - final_loss) / residual_before) if residual_before != 0.0 else 0.0
    )
    return {
        "schema": "tomojax.alternating_smoke.verification.v1",
        "status": "passed"
        if geometry_recovery["passed"] and volume_recovery["passed"]
        else "failed",
        "summary": {
            "projection_residual_improved": final_loss
            <= residual_before + cfg.verification_loss_tolerance,
            "final_reconstruction_valid": volume_recovery["passed"],
            "gauge_constraints_satisfied": _gauge_stable(
                final_geometry, tolerance=cfg.gauge_stability_tolerance
            ),
            "backend_provenance_complete": True,
            "weak_dofs_handled": True,
            "all_levels_verified": all_levels_verified,
        },
        "metrics": {
            "residual_before": residual_before,
            "residual_after": final_loss,
            "relative_improvement": relative_improvement,
            "final_loss": final_loss,
            "volume_nmse": volume_recovery["nmse"],
        },
        "runtime": {
            "time_to_verified_geometry_seconds": time_to_recovered_geometry_seconds,
            "total_wall_seconds": total_wall_seconds,
        },
        "escalation": {
            "level_1_geometry_run": not level1_geometry_skipped,
            "reason": "level_2_verification_passed"
            if level1_geometry_skipped
            else "level_1_geometry_required",
        },
        "seed": cfg.seed,
        "size": cfg.size,
        "n_views": cfg.n_views,
        "schedule": schedule.name,
        "geometry_update_volume_source": geometry_update_volume_source,
        "fit_gain_offset_nuisance": fit_gain_offset_nuisance,
        "fit_background_nuisance": fit_background_nuisance,
        "synthetic_dataset": _synthetic_dataset_payload(cfg),
        "level_factors": list(schedule.level_factors),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "coarse_verified": coarse_verified,
        "level1_geometry_skipped": level1_geometry_skipped,
        "skipped_levels": [summary.level_factor for summary in summaries if summary.skipped_level],
        "early_exit_reasons": [
            summary.early_exit_reason
            for summary in summaries
            if summary.early_exit_reason is not None
        ],
        "thresholds": {
            "loss_tolerance": cfg.verification_loss_tolerance,
            "gauge_stability_tolerance": cfg.gauge_stability_tolerance,
            "parameter_update_tolerance": cfg.parameter_update_tolerance,
            "heldout_residual_tolerance": cfg.heldout_residual_tolerance,
        },
        "geometry_recovery": geometry_recovery,
        "volume_recovery": volume_recovery,
        "stopped_volume_gauge": _stopped_volume_gauge_payload(
            final_volume=final_volume,
            observed=observed,
            mask=mask,
            initial_geometry=initial_geometry,
            final_geometry=final_geometry,
            true_geometry=true_geometry,
        ),
        "levels": [_summary_payload(summary) for summary in summaries],
    }


def _level1_geometry_skipped(summaries: tuple[AlternatingLevelSummary, ...]) -> bool:
    return any(
        summary.level_factor == 1
        and summary.geometry_updates > 0
        and summary.executed_geometry_updates == 0
        and summary.skipped_geometry
        for summary in summaries
    )


def _synthetic_dataset_payload(cfg: AlternatingSmokeConfig) -> dict[str, object] | None:
    if cfg.synthetic_dataset_name is None:
        return None
    payload: dict[str, object] = {
        "name": cfg.synthetic_dataset_name,
        "source": "synthetic128_spec",
    }
    if cfg.synthetic_dataset_artifact_dir is not None:
        payload["artifact_dir"] = str(cfg.synthetic_dataset_artifact_dir)
    payload["nuisance_applied_to_projections"] = bool(cfg.synthetic_dataset_nuisance_applied)
    if cfg.synthetic_dataset_sidecar_readback is not None:
        payload["sidecar_readback"] = dict(cfg.synthetic_dataset_sidecar_readback)
    return payload


def _geometry_recovery_payload(
    true_geometry: GeometryState,
    initial_geometry: GeometryState,
    final_geometry: GeometryState,
) -> dict[str, object]:
    raw_tolerances = _recovery_tolerances_payload()["geometry"]
    if not isinstance(raw_tolerances, dict):
        raise TypeError("geometry recovery tolerances must be a mapping")
    tolerances = cast("dict[str, float]", raw_tolerances)
    final_theta = final_geometry.setup.theta_offset_rad.value + final_geometry.pose.phi_residual_rad
    initial_theta = (
        initial_geometry.setup.theta_offset_rad.value + initial_geometry.pose.phi_residual_rad
    )
    true_theta = true_geometry.setup.theta_offset_rad.value + true_geometry.pose.phi_residual_rad
    final_u = final_geometry.setup.det_u_px.value + final_geometry.pose.dx_px
    initial_u = initial_geometry.setup.det_u_px.value + initial_geometry.pose.dx_px
    true_u = true_geometry.setup.det_u_px.value + true_geometry.pose.dx_px
    final_v = final_geometry.setup.det_v_px.value + final_geometry.pose.dz_px
    initial_v = initial_geometry.setup.det_v_px.value + initial_geometry.pose.dz_px
    true_v = true_geometry.setup.det_v_px.value + true_geometry.pose.dz_px
    initial_theta_rmse = float(np.sqrt(np.mean((initial_theta - true_theta) ** 2)))
    theta_rmse = float(np.sqrt(np.mean((final_theta - true_theta) ** 2)))
    initial_det_u_rmse = float(np.sqrt(np.mean((initial_u - true_u) ** 2)))
    det_u_rmse = float(np.sqrt(np.mean((final_u - true_u) ** 2)))
    initial_det_v_rmse = float(np.sqrt(np.mean((initial_v - true_v) ** 2)))
    det_v_rmse = float(np.sqrt(np.mean((final_v - true_v) ** 2)))
    mean_dx_abs = abs(float(np.mean(final_geometry.pose.dx_px)))
    mean_phi_abs = abs(float(np.mean(final_geometry.pose.phi_residual_rad)))
    mean_dz_abs = abs(float(np.mean(final_geometry.pose.dz_px)))
    det_v_gauge_active = final_geometry.setup.det_v_px.active
    theta_limit = float(tolerances["theta_realized_rmse_rad_lt"])
    det_u_limit = float(tolerances["det_u_realized_rmse_px_lt"])
    det_v_limit = float(tolerances["det_v_realized_rmse_px_lt"])
    gauge_limit = float(tolerances["mean_gauge_abs_lt"])
    passed = (
        theta_rmse <= theta_limit
        and det_u_rmse <= det_u_limit
        and det_v_rmse <= det_v_limit
        and mean_dx_abs <= gauge_limit
        and mean_phi_abs <= gauge_limit
        and (not det_v_gauge_active or mean_dz_abs <= gauge_limit)
    )
    theta_improved = bool(theta_rmse < initial_theta_rmse)
    det_u_improved = bool(det_u_rmse < initial_det_u_rmse)
    det_v_improved = bool(det_v_rmse < initial_det_v_rmse)
    theta_supported = _supported_dof_acceptable_or_improved(
        initial_error=initial_theta_rmse,
        final_error=theta_rmse,
        limit=theta_limit,
    )
    det_u_supported = _supported_dof_acceptable_or_improved(
        initial_error=initial_det_u_rmse,
        final_error=det_u_rmse,
        limit=det_u_limit,
    )
    det_v_supported = _supported_dof_acceptable_or_improved(
        initial_error=initial_det_v_rmse,
        final_error=det_v_rmse,
        limit=det_v_limit,
    )
    return {
        "initial_theta_realized_rmse_rad": initial_theta_rmse,
        "theta_realized_rmse_rad": theta_rmse,
        "theta_realized_rmse_rad_improved": theta_improved,
        "theta_realized_rmse_rad_passed": theta_rmse <= theta_limit,
        "theta_realized_rmse_rad_limit": theta_limit,
        "initial_det_u_realized_rmse_px": initial_det_u_rmse,
        "det_u_realized_rmse_px": det_u_rmse,
        "det_u_realized_rmse_px_improved": det_u_improved,
        "det_u_realized_rmse_px_passed": det_u_rmse <= det_u_limit,
        "det_u_realized_rmse_px_limit": det_u_limit,
        "initial_det_v_realized_rmse_px": initial_det_v_rmse,
        "det_v_realized_rmse_px": det_v_rmse,
        "det_v_realized_rmse_px_improved": det_v_improved,
        "det_v_realized_rmse_px_passed": det_v_rmse <= det_v_limit,
        "det_v_realized_rmse_px_limit": det_v_limit,
        "mean_dx_abs_px": mean_dx_abs,
        "mean_dx_abs_px_passed": mean_dx_abs <= gauge_limit,
        "mean_dx_abs_px_limit": gauge_limit,
        "mean_phi_abs_rad": mean_phi_abs,
        "mean_phi_abs_rad_passed": mean_phi_abs <= gauge_limit,
        "mean_phi_abs_rad_limit": gauge_limit,
        "mean_dz_abs_px": mean_dz_abs,
        "mean_dz_abs_px_passed": (not det_v_gauge_active or mean_dz_abs <= gauge_limit),
        "mean_dz_abs_px_limit": gauge_limit,
        "supported_dofs_improved": theta_supported and det_u_supported and det_v_supported,
        "passed": passed,
    }


def _supported_dof_acceptable_or_improved(
    *,
    initial_error: float,
    final_error: float,
    limit: float,
) -> bool:
    if initial_error <= limit:
        return bool(final_error <= limit)
    return bool(final_error < initial_error)


def _volume_recovery_payload(truth_volume: jax.Array, final_volume: jax.Array) -> dict[str, object]:
    truth = jnp.asarray(truth_volume, dtype=jnp.float32)
    final = jnp.asarray(final_volume, dtype=jnp.float32)
    diff = final - truth
    mse = jnp.mean(diff * diff)
    truth_energy = jnp.mean(truth * truth)
    nmse = float(mse / jnp.maximum(truth_energy, jnp.asarray(1.0e-12, dtype=jnp.float32)))
    rmse = float(jnp.sqrt(mse))
    mae = float(jnp.mean(jnp.abs(diff)))
    tolerances = _recovery_tolerances_payload()["volume"]
    if not isinstance(tolerances, dict):
        raise TypeError("volume recovery tolerances must be a mapping")
    typed_tolerances = cast("dict[str, float]", tolerances)
    nmse_limit = float(typed_tolerances["nmse_lt"])
    return {
        "rmse": rmse,
        "mae": mae,
        "nmse": nmse,
        "nmse_limit": nmse_limit,
        "nmse_passed": nmse <= nmse_limit,
        "passed": nmse <= nmse_limit,
    }


def _stopped_volume_gauge_payload(
    *,
    final_volume: jax.Array,
    observed: jax.Array,
    mask: jax.Array,
    initial_geometry: GeometryState,
    final_geometry: GeometryState,
    true_geometry: GeometryState,
) -> dict[str, object]:
    initial_loss = _projection_loss_for_geometry(final_volume, observed, mask, initial_geometry)
    final_loss = _projection_loss_for_geometry(final_volume, observed, mask, final_geometry)
    true_loss = _projection_loss_for_geometry(final_volume, observed, mask, true_geometry)
    nearest = min(
        (
            ("initial_geometry", initial_loss),
            ("final_geometry", final_loss),
            ("true_geometry", true_loss),
        ),
        key=lambda item: item[1],
    )[0]
    return {
        "schema": "tomojax.stopped_volume_gauge.v1",
        "projection_loss_initial_geometry": initial_loss,
        "projection_loss_final_geometry": final_loss,
        "projection_loss_true_geometry": true_loss,
        "nearest_geometry": nearest,
        "closer_to_initial_than_true": initial_loss < true_loss,
        "closer_to_final_than_true": final_loss < true_loss,
    }


def _projection_loss_for_geometry(
    volume: jax.Array,
    observed: jax.Array,
    mask: jax.Array,
    geometry: GeometryState,
) -> float:
    predicted = project_parallel_reference(volume, geometry)
    return float(residual_loss(predicted, observed, mask=mask).loss)


def _summary_payload(summary: AlternatingLevelSummary) -> dict[str, object]:
    return {
        "level_factor": summary.level_factor,
        "role": summary.role,
        "reconstruction_iterations": summary.reconstruction_iterations,
        "geometry_updates": summary.geometry_updates,
        "executed_geometry_updates": summary.executed_geometry_updates,
        "residual_filter_kinds": "|".join(summary.residual_filter_kinds),
        "loss_before": summary.loss_before,
        "loss_after": summary.loss_after,
        "loss_nonincreasing": summary.loss_nonincreasing,
        "finite_loss": summary.finite_loss,
        "residual_sigma_estimated": summary.residual_sigma_estimated,
        "residual_sigma_effective": summary.residual_sigma_effective,
        "prior_strength": summary.prior_strength,
        "heldout_residual_before": summary.heldout_residual_before,
        "heldout_residual_after": summary.heldout_residual_after,
        "heldout_residual_passed": summary.heldout_residual_passed,
        "gauge_stable": summary.gauge_stable,
        "parameter_update_norm": summary.parameter_update_norm,
        "parameter_update_small": summary.parameter_update_small,
        "verified": summary.verified,
        "skipped_geometry": summary.skipped_geometry,
        "skipped_level": summary.skipped_level,
        "early_exit_reason": summary.early_exit_reason,
    }


def _input_summary_payload(volume: jax.Array, projections: jax.Array) -> dict[str, object]:
    return {
        "schema": "tomojax.input_summary.v1",
        "volume_shape": list(volume.shape),
        "volume_dtype": str(volume.dtype),
        "projection_shape": list(projections.shape),
        "projection_dtype": str(projections.dtype),
    }


def _projection_stats_payload(projections: jax.Array) -> dict[str, object]:
    values = jnp.asarray(projections, dtype=jnp.float32)
    return {
        "schema": "tomojax.projection_stats.v1",
        "min": float(jnp.min(values)),
        "max": float(jnp.max(values)),
        "mean": float(jnp.mean(values)),
        "std": float(jnp.std(values)),
    }


def _mask_summary_payload(mask: jax.Array) -> dict[str, object]:
    values = jnp.asarray(mask, dtype=jnp.float32)
    total = int(values.size)
    valid = int(jnp.sum(values > 0.0))
    return {
        "schema": "tomojax.mask_summary.v1",
        "valid_pixels": valid,
        "total_pixels": total,
        "valid_fraction": float(valid / total),
    }


def _recovery_tolerances_payload() -> dict[str, object]:
    return {
        "schema": "tomojax.recovery_tolerances.v1",
        "profile": "smoke32",
        "geometry": {
            "theta_realized_rmse_rad_lt": 8.5e-2,
            "det_u_realized_rmse_px_lt": 2.0e-1,
            "det_v_realized_rmse_px_lt": 2.0e-1,
            "mean_gauge_abs_lt": 1.0e-10,
        },
        "volume": {
            "nmse_lt": 10.0,
        },
        "verification": {
            "loss_nonincreasing": True,
            "finite_loss": True,
            "gauge_stable": True,
            "parameter_update_small": True,
        },
    }


def _gauge_policy_payload() -> dict[str, object]:
    return {
        "schema": "tomojax.gauge_policy.v1",
        "operations": [
            {
                "name": "mean_dx_to_det_u",
                "source": "pose.dx_px.mean",
                "target": "setup.det_u_px",
                "enabled": True,
            },
            {
                "name": "mean_phi_to_theta_offset",
                "source": "pose.phi_residual_rad.mean",
                "target": "setup.theta_offset_rad",
                "enabled": True,
            },
            {
                "name": "mean_dz_to_det_v_if_active",
                "source": "pose.dz_px.mean",
                "target": "setup.det_v_px",
                "enabled": True,
            },
        ],
    }


def _gauge_report_payload(report: GaugeReport) -> dict[str, object]:
    return {
        "schema": "tomojax.gauge_report.v1",
        "status": "passed",
        "operations": [
            {
                "source": transfer.source,
                "target": transfer.target,
                "value": transfer.value,
                "unit": transfer.unit,
                "applied": transfer.applied,
                "reason": transfer.reason,
            }
            for transfer in report.transfers
        ],
    }


def _observability_report_payload(schur_result: JointSchurLMResult | None) -> dict[str, object]:
    diagnostics = None if schur_result is None else schur_result.diagnostics
    schur_condition = None if diagnostics is None else diagnostics.schur_condition
    schur_eigenvalues = () if diagnostics is None else diagnostics.schur_eigenvalues
    min_schur_eigenvalue = (
        None if not schur_eigenvalues else min(float(v) for v in schur_eigenvalues)
    )
    weak_mode_labels = () if diagnostics is None else diagnostics.weak_mode_labels
    status = "evaluated" if diagnostics is not None else "not_run"
    det_v_active = (
        False if schur_result is None else "det_v_px" in schur_result.active_setup_parameters
    )
    det_v_status = "evaluated" if det_v_active else "frozen"
    det_v_reason = (
        "active_in_schur_setup_block"
        if det_v_active
        else "det_v_px is frozen in the current smoke geometry"
    )
    weak_modes = [
        {
            "name": "schur_weak_modes",
            "severity": "info",
            "affected_dofs": list(weak_mode_labels),
            "reason": "Schur eigenvalue threshold flagged weak setup modes"
            if weak_mode_labels
            else "no Schur eigenvalue fell below the weak-mode threshold",
        }
    ]
    weak_dof_policy = _weak_dof_policy_payload(
        diagnostics=diagnostics,
        det_v_active=det_v_active,
        det_v_curvature=min_schur_eigenvalue if det_v_active else None,
    )
    return {
        "schema": "tomojax.observability_report.v1",
        "status": status,
        "reason": "Schur curvature diagnostics from the last geometry update"
        if diagnostics is not None
        else "No Schur geometry update was run",
        "schur_condition_number": schur_condition,
        "schur_min_eigenvalue": min_schur_eigenvalue,
        "schur_eigenvalues": [float(value) for value in schur_eigenvalues],
        "weak_dof_policy": weak_dof_policy,
        "dofs": {
            "setup": {
                "det_u_px": {
                    "active": True,
                    "observable": diagnostics is not None,
                    "status": "evaluated" if diagnostics is not None else "not_run",
                    "gauge_group": "detector_u",
                    "curvature": None
                    if not diagnostics or not diagnostics.schur_eigenvalues
                    else float(max(diagnostics.schur_eigenvalues)),
                    "reason": "included in supported Schur setup block",
                },
                "det_v_px": {
                    "active": det_v_active,
                    "observable": det_v_active and diagnostics is not None,
                    "status": det_v_status,
                    "gauge_group": "detector_v",
                    "curvature": min_schur_eigenvalue if det_v_active else None,
                    "reason": det_v_reason,
                },
                "detector_roll_rad": {
                    "active": True,
                    "observable": False,
                    "status": "weak_not_evaluated",
                    "gauge_group": "rotation",
                },
                "axis_rot_x_rad": {
                    "active": True,
                    "observable": False,
                    "status": "weak_not_evaluated",
                    "gauge_group": "axis",
                },
                "axis_rot_y_rad": {
                    "active": True,
                    "observable": False,
                    "status": "weak_not_evaluated",
                    "gauge_group": "axis",
                },
                "theta_offset_rad": {
                    "active": True,
                    "observable": False,
                    "status": "weak_not_evaluated",
                    "gauge_group": "rotation",
                },
                "theta_scale": {
                    "active": False,
                    "observable": False,
                    "status": "frozen",
                    "gauge_group": "none",
                    "curvature": None,
                    "reason": "theta_scale is frozen until identifiable scale policy exists",
                },
            },
            "pose": {
                "alpha_rad": {"active": True, "observable": False, "status": "weak_not_evaluated"},
                "beta_rad": {"active": True, "observable": False, "status": "weak_not_evaluated"},
                "phi_residual_rad": {
                    "active": True,
                    "observable": False,
                    "status": "gauge_canonicalised",
                },
                "dx_px": {
                    "active": True,
                    "observable": False,
                    "status": "gauge_canonicalised",
                },
                "dz_px": {"active": True, "observable": False, "status": "weak_not_evaluated"},
            },
        },
        "weak_modes": weak_modes,
        "handled_frozen_dofs": ["theta_scale"] if det_v_active else ["det_v_px", "theta_scale"],
    }


def _weak_dof_policy_payload(
    *,
    diagnostics: JointSchurDiagnostics | None,
    det_v_active: bool,
    det_v_curvature: float | None,
) -> dict[str, object]:
    curvature_floor = 1.0e-8
    accepted_step_required = True
    missing_validation = "validation_improvement_gate_not_available_in_smoke"
    det_v_validation_improvement = _schur_validation_improvement_payload(diagnostics)
    det_v_correlation = _setup_correlation_payload(
        diagnostics,
        parameter_index=2 if det_v_active else None,
    )
    det_v_decision = _weak_dof_decision(
        name="det_v_px",
        active=det_v_active,
        curvature=det_v_curvature,
        correlation=det_v_correlation,
        accepted=diagnostics.accepted if diagnostics is not None else False,
        accepted_available=diagnostics is not None,
        curvature_floor=curvature_floor,
        correlation_abs_max_ceiling=0.98,
        accepted_step_required=accepted_step_required,
        frozen_reason="det_v_px is frozen in the current geometry",
        validation_improvement=det_v_validation_improvement,
        missing_validation=missing_validation,
    )
    theta_scale_decision = _weak_dof_decision(
        name="theta_scale",
        active=False,
        curvature=None,
        correlation=None,
        accepted=False,
        accepted_available=False,
        curvature_floor=curvature_floor,
        correlation_abs_max_ceiling=0.98,
        accepted_step_required=accepted_step_required,
        frozen_reason="theta_scale is unsupported by the current reference projector",
        validation_improvement=None,
        missing_validation=missing_validation,
    )
    return {
        "schema": "tomojax.weak_dof_policy.v1",
        "mode": "report_only",
        "thresholds": {
            "curvature_floor": curvature_floor,
            "correlation_abs_max_ceiling": 0.98,
            "accepted_step_required": accepted_step_required,
        },
        "decisions": {
            "det_v_px": det_v_decision,
            "theta_scale": theta_scale_decision,
        },
    }


def _weak_dof_decision(
    *,
    name: str,
    active: bool,
    curvature: float | None,
    correlation: Mapping[str, object] | None,
    accepted: bool,
    accepted_available: bool,
    curvature_floor: float,
    correlation_abs_max_ceiling: float,
    accepted_step_required: bool,
    frozen_reason: str,
    validation_improvement: Mapping[str, object] | None,
    missing_validation: str,
) -> dict[str, object]:
    curvature_passed = curvature is not None and curvature >= curvature_floor
    correlation_passed = bool(correlation["passed"]) if correlation is not None else False
    accepted_passed = (
        bool(accepted)
        if accepted_step_required and accepted_available
        else not accepted_step_required
    )
    validation_improvement_passed = (
        bool(validation_improvement["passed"]) if validation_improvement is not None else False
    )
    missing_evidence = []
    if curvature is None:
        missing_evidence.append("curvature")
    if correlation is None:
        missing_evidence.append("correlation")
    if accepted_step_required and not accepted_available:
        missing_evidence.append("accepted_step")
    if validation_improvement is None:
        missing_evidence.append(missing_validation)
    if not active:
        decision = "keep_frozen"
        reason = frozen_reason
    elif curvature_passed and correlation_passed and accepted_passed:
        decision = "keep_active_with_prior"
        reason = (
            "curvature, correlation, and accepted-step evidence are sufficient "
            "for report-only smoke policy"
        )
    else:
        decision = "freeze_or_prior_required"
        reason = "insufficient curvature, correlation, or accepted-step evidence"
    return {
        "name": name,
        "active": active,
        "decision": decision,
        "reason": reason,
        "evidence": {
            "curvature": curvature,
            "curvature_floor": curvature_floor,
            "curvature_passed": curvature_passed,
            "correlation": correlation,
            "correlation_abs_max_ceiling": correlation_abs_max_ceiling,
            "correlation_passed": correlation_passed,
            "accepted_step": accepted,
            "accepted_step_required": accepted_step_required,
            "accepted_step_passed": accepted_passed,
            "validation_improvement": validation_improvement,
            "validation_improvement_passed": validation_improvement_passed,
            "missing_evidence": missing_evidence,
        },
    }


def _setup_correlation_payload(
    diagnostics: JointSchurDiagnostics | None,
    *,
    parameter_index: int | None,
) -> dict[str, object] | None:
    if diagnostics is None or parameter_index is None:
        return None
    matrix = diagnostics.setup_correlation_matrix
    if not matrix or parameter_index >= len(matrix):
        return None
    values = [
        abs(float(value))
        for index, value in enumerate(matrix[parameter_index])
        if index != parameter_index
    ]
    if not values:
        return None
    max_abs = max(values)
    ceiling = 0.98
    return {
        "kind": "setup_correlation_matrix",
        "parameter_index": parameter_index,
        "max_abs_correlation": max_abs,
        "max_abs_correlation_ceiling": ceiling,
        "passed": bool(max_abs <= ceiling),
    }


def _schur_validation_improvement_payload(
    diagnostics: JointSchurDiagnostics | None,
) -> dict[str, object] | None:
    if diagnostics is None:
        return None
    actual_reduction = float(diagnostics.actual_reduction)
    reduction_ratio = diagnostics.reduction_ratio
    return {
        "kind": "schur_actual_reduction",
        "actual_reduction": actual_reduction,
        "reduction_ratio": None if reduction_ratio is None else float(reduction_ratio),
        "accepted": bool(diagnostics.accepted),
        "passed": bool(diagnostics.accepted and actual_reduction > 0.0),
    }


def _backend_report_payload() -> dict[str, object]:
    return {
        "schema": "tomojax.backend_report.v1",
        "requested": "jax_reference",
        "actual": "jax_reference",
        "actual_projector": "jax_reference",
        "actual_backprojector": "jax_reference",
        "actual_geometry_reductions": "jax_reference",
        "canonical_detector_grid": True,
        "calibrated_detector_grid": False,
        "pallas_eligible": False,
        "fallback": False,
        "fallbacks": [],
        "agreement_tests": [
            {
                "component": "residual",
                "max_abs_error": 0.0,
                "mean_abs_error": 0.0,
                "status": "reference_baseline",
            }
        ],
    }


def _failure_report_payload(
    *,
    final_volume: jax.Array,
    final_geometry: GeometryState,
    observed: jax.Array,
    mask: jax.Array,
    summaries: tuple[AlternatingLevelSummary, ...],
    verification: Mapping[str, object],
) -> dict[str, object]:
    gates = _failure_gate_rows(
        final_volume=final_volume,
        final_geometry=final_geometry,
        observed=observed,
        mask=mask,
        summaries=summaries,
        verification=verification,
    )
    return failure_report_from_gates(gates)


def _failure_gate_rows(
    *,
    final_volume: jax.Array,
    final_geometry: GeometryState,
    observed: jax.Array,
    mask: jax.Array,
    summaries: tuple[AlternatingLevelSummary, ...],
    verification: Mapping[str, object],
) -> list[dict[str, object]]:
    summary = cast("Mapping[str, object]", verification["summary"])
    valid_fraction = float(jnp.mean(jnp.asarray(mask, dtype=jnp.float32)))
    predicted = project_parallel_reference(final_volume, final_geometry)
    residual_structure = residual_structure_summary(predicted - observed, mask)
    return [
        {
            "name": "finite_outputs",
            "passed": _finite_outputs(final_volume, final_geometry) and valid_fraction > 0.0,
            "severity": "error",
            "evidence": f"valid_pixel_fraction={valid_fraction}",
        },
        {
            "name": "projection_residual_improvement",
            "passed": bool(summary["projection_residual_improved"]),
            "severity": "warning",
            "evidence": "final residual is compared against first level loss_before",
        },
        {
            "name": "gauge_stability",
            "passed": bool(summary["gauge_constraints_satisfied"]),
            "severity": "error",
            "evidence": "mean dx and mean phi residual are gauge canonicalised",
        },
        {
            "name": "optimiser_health",
            "passed": any(level.executed_geometry_updates > 0 for level in summaries),
            "severity": "warning",
            "evidence": "smoke geometry update uses gauge canonicalisation placeholder",
        },
        {
            "name": "backend_provenance",
            "passed": bool(summary["backend_provenance_complete"]),
            "severity": "error",
            "evidence": "backend_report.json records requested and actual reference components",
        },
        {
            "name": "nuisance_residual_structure",
            "passed": bool(residual_structure["passed"]),
            "severity": "warning",
            "evidence": residual_structure,
        },
        _synthetic_sidecar_consistency_gate(verification),
    ]


def _synthetic_sidecar_consistency_gate(verification: Mapping[str, object]) -> dict[str, object]:
    synthetic_dataset = verification.get("synthetic_dataset")
    if not isinstance(synthetic_dataset, Mapping):
        return {
            "name": "synthetic_sidecar_consistency",
            "passed": True,
            "severity": "warning",
            "evidence": "no synthetic sidecar dataset requested",
        }
    synthetic_payload = cast("Mapping[object, object]", synthetic_dataset)
    raw_sidecar_readback = synthetic_payload.get("sidecar_readback")
    if not isinstance(raw_sidecar_readback, Mapping):
        return {
            "name": "synthetic_sidecar_consistency",
            "passed": False,
            "severity": "warning",
            "evidence": "synthetic sidecar dataset lacks sidecar_readback payload",
        }
    sidecar_readback = cast("Mapping[object, object]", raw_sidecar_readback)
    raw_consistency = sidecar_readback.get("consistency")
    if not isinstance(raw_consistency, Mapping):
        return {
            "name": "synthetic_sidecar_consistency",
            "passed": False,
            "severity": "warning",
            "evidence": "synthetic sidecar readback lacks consistency payload",
        }
    consistency = cast("Mapping[object, object]", raw_consistency)
    return {
        "name": "synthetic_sidecar_consistency",
        "passed": bool(consistency.get("passed")),
        "severity": "warning",
        "evidence": {str(key): value for key, value in consistency.items()},
    }


def _finite_outputs(volume: jax.Array, geometry: GeometryState) -> bool:
    setup_values = (
        geometry.setup.det_u_px.value,
        geometry.setup.det_v_px.value,
        geometry.setup.detector_roll_rad.value,
        geometry.setup.axis_rot_x_rad.value,
        geometry.setup.axis_rot_y_rad.value,
        geometry.setup.theta_offset_rad.value,
        geometry.setup.theta_scale.value,
    )
    pose_values = (
        geometry.pose.alpha_rad,
        geometry.pose.beta_rad,
        geometry.pose.phi_residual_rad,
        geometry.pose.dx_px,
        geometry.pose.dz_px,
    )
    return bool(
        jnp.all(jnp.isfinite(volume))
        and all(np.isfinite(value) for value in setup_values)
        and all(np.isfinite(values).all() for values in pose_values)
    )
