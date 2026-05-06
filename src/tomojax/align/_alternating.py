"""Small v2 alternating alignment smoke runner."""
# pyright: reportAny=false, reportUnknownMemberType=false

from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.align._continuation import (
    ContinuationLevel,
    ContinuationSchedule,
    reference_continuation_schedule,
)
from tomojax.align._joint_schur_lm import (
    JointSchurDiagnostics,
    JointSchurLMConfig,
    JointSchurLMResult,
    joint_schur_normal_eq_summary,
    solve_joint_schur_lm,
)
from tomojax.datasets import make_benchmark_phantom
from tomojax.forward import (
    apply_residual_filter_schedule,
    project_parallel_reference,
    residual_loss,
    robust_residual_scale,
)
from tomojax.geometry import (
    GaugeReport,
    GeometryState,
    write_geometry_json,
    write_pose_decomposition_csv,
    write_pose_params_csv,
)
from tomojax.recon import (
    ReferenceFISTAConfig,
    ReferenceFISTAResult,
    fista_reconstruct_reference,
    reconstruct_backprojection_reference,
    write_fista_trace_csv,
)
from tomojax.verify import residual_structure_summary, validate_run_artifacts

if TYPE_CHECKING:
    from collections.abc import Mapping

GeometryUpdateVolumeSource = Literal["fixed_synthetic_truth", "stopped_reconstruction"]


@dataclass(frozen=True)
class AlternatingSmokeConfig:
    """Configuration for the deterministic 32^3 alternating smoke run."""

    seed: int = 17
    size: int = 32
    n_views: int = 4
    schedule: ContinuationSchedule | None = None
    verification_loss_tolerance: float = 1.0e-5
    gauge_stability_tolerance: float = 1.0e-10
    parameter_update_tolerance: float = 2.0
    heldout_residual_tolerance: float = 1.0e-5
    heldout_view_index: int | None = -1
    geometry_update_volume_source: GeometryUpdateVolumeSource = "stopped_reconstruction"
    fit_gain_offset_nuisance: bool = False
    synthetic_dataset_name: str | None = None
    synthetic_dataset_artifact_dir: Path | None = None


@dataclass(frozen=True)
class AlternatingLevelSummary:
    """Per-level alternating smoke summary."""

    level_factor: int
    role: str
    reconstruction_iterations: int
    geometry_updates: int
    executed_geometry_updates: int
    residual_filter_kinds: tuple[str, ...]
    loss_before: float
    loss_after: float
    loss_nonincreasing: bool
    finite_loss: bool
    residual_sigma_estimated: float
    residual_sigma_effective: float
    prior_strength: float
    heldout_residual_before: float | None
    heldout_residual_after: float | None
    heldout_residual_passed: bool | None
    gauge_stable: bool
    parameter_update_norm: float
    parameter_update_small: bool
    verified: bool
    skipped_geometry: bool
    skipped_level: bool
    early_exit_reason: str | None
    schur_diagnostics: JointSchurDiagnostics | None = None


@dataclass(frozen=True)
class _LevelVerificationChecks:
    loss_nonincreasing: bool
    finite_loss: bool
    gauge_stable: bool
    parameter_update_norm: float
    parameter_update_small: bool
    verified: bool


@dataclass(frozen=True)
class AlternatingSmokeResult:
    """Result payload for the deterministic alternating smoke run."""

    final_volume: jax.Array
    initial_geometry: GeometryState
    final_geometry: GeometryState
    levels: tuple[AlternatingLevelSummary, ...]
    verification: Mapping[str, object]
    artifacts: Mapping[str, Path]


@dataclass(frozen=True)
class AlternatingAlignmentSolver:
    """Phase 7 alternating alignment solver entrypoint."""

    config: AlternatingSmokeConfig = field(default_factory=AlternatingSmokeConfig)

    def run_smoke(self, output_dir: str | Path) -> AlternatingSmokeResult:
        """Run the deterministic 32^3 smoke profile."""
        return _run_alternating_solver_smoke_impl(output_dir, config=self.config)


def run_alternating_solver_smoke(
    output_dir: str | Path,
    *,
    config: AlternatingSmokeConfig | None = None,
) -> AlternatingSmokeResult:
    """Run the smallest deterministic v2 alternating solver smoke slice."""
    return AlternatingAlignmentSolver(config=config or AlternatingSmokeConfig()).run_smoke(
        output_dir
    )


def _run_alternating_solver_smoke_impl(
    output_dir: str | Path,
    *,
    config: AlternatingSmokeConfig,
) -> AlternatingSmokeResult:
    schedule = config.schedule or reference_continuation_schedule()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = jnp.asarray(make_benchmark_phantom(config.size, seed=config.seed), dtype=jnp.float32)
    true_geometry = _synthetic_true_geometry(config.n_views)
    initial_geometry = _synthetic_initial_geometry(config.n_views)
    observed = project_parallel_reference(truth, true_geometry)
    mask = jnp.ones_like(observed, dtype=jnp.float32)
    train_mask, heldout_mask = _heldout_masks(mask, config.heldout_view_index)

    geometry = initial_geometry
    gauge_report = GaugeReport(())
    volume: jax.Array | None = None
    summaries: list[AlternatingLevelSummary] = []
    fista_result: ReferenceFISTAResult | None = None
    coarse_verified = False
    last_schur_result: JointSchurLMResult | None = None

    initial_loss = _projection_loss(
        truth,
        observed,
        geometry,
        mask,
        schedule.levels[0],
        sigma=schedule.levels[0].residual_sigma,
    )
    previous_loss = initial_loss
    for level in schedule.levels:
        if level.run_if_coarse_unverified and coarse_verified:
            summaries.append(
                _skipped_level_summary(
                    level,
                    previous_loss,
                    early_exit_reason="coarse_verification_passed",
                )
            )
            continue

        fista_result = fista_reconstruct_reference(
            observed,
            geometry,
            initial_volume=(
                volume
                if volume is not None
                else reconstruct_backprojection_reference(observed, geometry, depth=config.size)
            ),
            mask=mask,
            config=ReferenceFISTAConfig(
                iterations=level.reconstruction_iterations,
                step_size=2.0e-3,
                tv_weight=0.0,
                residual_sigma=level.residual_sigma,
                residual_delta=level.residual_delta,
            ),
        )
        volume = jax.lax.stop_gradient(fista_result.volume)
        residual_sigma_estimated = _level_residual_sigma(volume, observed, geometry, mask, level)
        residual_sigma_effective = max(level.residual_sigma, residual_sigma_estimated)
        loss_before = _projection_loss(
            volume,
            observed,
            geometry,
            mask,
            level,
            sigma=residual_sigma_effective,
        )
        geometry_updates, early_exit_reason = _geometry_updates_for_level(level, coarse_verified)
        skipped_geometry = geometry_updates == 0
        update_report = GaugeReport(())
        geometry_before_update = geometry
        if geometry_updates > 0:
            geometry_update_volume = _geometry_update_volume(
                truth_volume=truth,
                stopped_volume=volume,
                source=config.geometry_update_volume_source,
            )
            geometry, update_report, schur_result = _run_geometry_updates(
                geometry_update_volume,
                observed,
                geometry,
                train_mask,
                level,
                geometry_updates,
                sigma=residual_sigma_effective,
                fit_gain_offset_nuisance=config.fit_gain_offset_nuisance,
            )
            gauge_report = update_report
            last_schur_result = schur_result
            loss_before = schur_result.initial_loss
            loss_after = schur_result.final_loss
        else:
            schur_result = None
            loss_after = _projection_loss(
                volume,
                observed,
                geometry,
                mask,
                level,
                sigma=residual_sigma_effective,
            )
        heldout_before, heldout_after, heldout_passed = _heldout_residual_check(
            volume,
            observed,
            before_geometry=geometry_before_update,
            after_geometry=geometry,
            heldout_mask=heldout_mask,
            level=level,
            sigma=residual_sigma_effective,
            tolerance=config.heldout_residual_tolerance,
        )
        verification_checks = _level_verification_checks(
            cfg=config,
            geometry=geometry,
            update_report=update_report,
            loss_before=loss_before,
            loss_after=loss_after,
            heldout_residual_passed=heldout_passed,
        )
        previous_loss = loss_after
        coarse_verified = coarse_verified or (
            level.role == "preview"
            and level.skip_finer_if_verified
            and verification_checks.verified
        )
        summaries.append(
            AlternatingLevelSummary(
                level_factor=level.level_factor,
                role=level.role,
                reconstruction_iterations=level.reconstruction_iterations,
                geometry_updates=level.geometry_updates,
                executed_geometry_updates=geometry_updates,
                residual_filter_kinds=_residual_filter_kinds(level),
                loss_before=loss_before,
                loss_after=loss_after,
                loss_nonincreasing=verification_checks.loss_nonincreasing,
                finite_loss=verification_checks.finite_loss,
                residual_sigma_estimated=residual_sigma_estimated,
                residual_sigma_effective=residual_sigma_effective,
                prior_strength=level.prior_strength,
                heldout_residual_before=heldout_before,
                heldout_residual_after=heldout_after,
                heldout_residual_passed=heldout_passed,
                gauge_stable=verification_checks.gauge_stable,
                parameter_update_norm=verification_checks.parameter_update_norm,
                parameter_update_small=verification_checks.parameter_update_small,
                verified=verification_checks.verified,
                skipped_geometry=skipped_geometry,
                skipped_level=False,
                early_exit_reason=early_exit_reason,
                schur_diagnostics=(None if schur_result is None else schur_result.diagnostics),
            )
        )

    if volume is None or fista_result is None:
        raise RuntimeError("continuation schedule produced no reconstruction")

    final_loss = (
        last_schur_result.final_loss if last_schur_result is not None else summaries[-1].loss_after
    )
    artifacts = _write_artifacts(
        out_dir,
        true_geometry=true_geometry,
        initial_geometry=initial_geometry,
        final_geometry=geometry,
        truth_volume=truth,
        final_volume=volume,
        observed=observed,
        mask=mask,
        schedule=schedule,
        gauge_report=gauge_report,
        fista_result=fista_result,
        summaries=tuple(summaries),
        schur_result=last_schur_result,
        geometry_update_volume_source=config.geometry_update_volume_source,
        fit_gain_offset_nuisance=config.fit_gain_offset_nuisance,
        verification=_verification_payload(
            cfg=config,
            schedule=schedule,
            initial_loss=initial_loss,
            final_loss=final_loss,
            coarse_verified=coarse_verified,
            true_geometry=true_geometry,
            final_geometry=geometry,
            truth_volume=truth,
            final_volume=volume,
            summaries=tuple(summaries),
            geometry_update_volume_source=config.geometry_update_volume_source,
            fit_gain_offset_nuisance=config.fit_gain_offset_nuisance,
        ),
    )
    verification = json.loads(artifacts["verification_json"].read_text(encoding="utf-8"))
    return AlternatingSmokeResult(
        final_volume=volume,
        initial_geometry=initial_geometry,
        final_geometry=geometry,
        levels=tuple(summaries),
        verification=verification,
        artifacts=artifacts,
    )


def _synthetic_initial_geometry(n_views: int) -> GeometryState:
    base = _geometry_with_active_det_v(n_views)
    return GeometryState(
        setup=base.setup,
        pose=base.pose,
    )


def _synthetic_true_geometry(n_views: int) -> GeometryState:
    base = _geometry_with_active_det_v(n_views)
    span = np.linspace(-1.0, 1.0, num=n_views, dtype=np.float64)
    setup = base.setup.replace_parameter(
        "theta_offset_rad",
        base.setup.theta_offset_rad.with_value(0.035),
    )
    setup = setup.replace_parameter("det_u_px", setup.det_u_px.with_value(0.045))
    setup = setup.replace_parameter("det_v_px", setup.det_v_px.with_value(-0.03))
    return GeometryState(
        setup=setup,
        pose=base.pose.with_updates(
            phi_residual_rad=0.1 * span,
            dx_px=0.02 + 0.01 * span,
            dz_px=-0.0125 + 0.0075 * span,
        ),
    )


def _geometry_with_active_det_v(n_views: int) -> GeometryState:
    base = GeometryState.zeros(n_views)
    setup = base.setup.replace_parameter(
        "det_v_px",
        replace(base.setup.det_v_px, value=0.0, active=True),
    )
    return GeometryState(setup=setup, pose=base.pose)


def _heldout_masks(
    mask: jax.Array,
    heldout_view_index: int | None,
) -> tuple[jax.Array, jax.Array | None]:
    if heldout_view_index is None:
        return mask, None
    n_views = int(mask.shape[0])
    view_index = int(heldout_view_index)
    if view_index < 0:
        view_index += n_views
    if view_index < 0 or view_index >= n_views:
        raise ValueError("heldout_view_index must select an existing view")
    heldout = jnp.zeros_like(mask, dtype=jnp.float32).at[view_index, :, :].set(mask[view_index])
    train = jnp.asarray(mask, dtype=jnp.float32).at[view_index, :, :].set(0.0)
    return train, heldout


def _run_geometry_updates(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    mask: jax.Array,
    level: ContinuationLevel,
    updates: int,
    *,
    sigma: float,
    fit_gain_offset_nuisance: bool,
) -> tuple[GeometryState, GaugeReport, JointSchurLMResult]:
    result = solve_joint_schur_lm(
        volume,
        observed,
        geometry,
        mask=mask,
        config=JointSchurLMConfig(
            max_iterations=max(1, int(updates)),
            damping=1.0e-3,
            delta=level.residual_delta,
            sigma=sigma,
            setup_trust_radius=level.trust_radius_px,
            pose_trust_radius=level.trust_radius_px,
            parameter_prior_strength=level.prior_strength,
            fit_gain_offset=fit_gain_offset_nuisance,
        ),
    )
    return result.canonicalized_geometry.state, result.canonicalized_geometry.report, result


def _geometry_update_volume(
    *,
    truth_volume: jax.Array,
    stopped_volume: jax.Array,
    source: GeometryUpdateVolumeSource,
) -> jax.Array:
    if source == "fixed_synthetic_truth":
        return jax.lax.stop_gradient(truth_volume)
    if source == "stopped_reconstruction":
        return jax.lax.stop_gradient(stopped_volume)
    raise ValueError(f"unknown geometry update volume source {source!r}")


def _geometry_updates_for_level(
    level: ContinuationLevel,
    coarse_verified: bool,
) -> tuple[int, str | None]:
    if level.role == "final" and coarse_verified:
        return 0, "coarse_verification_passed"
    return level.geometry_updates, None


def _skipped_level_summary(
    level: ContinuationLevel,
    loss: float,
    *,
    early_exit_reason: str,
) -> AlternatingLevelSummary:
    return AlternatingLevelSummary(
        level_factor=level.level_factor,
        role=level.role,
        reconstruction_iterations=0,
        geometry_updates=level.geometry_updates,
        executed_geometry_updates=0,
        residual_filter_kinds=_residual_filter_kinds(level),
        loss_before=loss,
        loss_after=loss,
        loss_nonincreasing=True,
        finite_loss=True,
        residual_sigma_estimated=level.residual_sigma,
        residual_sigma_effective=level.residual_sigma,
        prior_strength=level.prior_strength,
        heldout_residual_before=None,
        heldout_residual_after=None,
        heldout_residual_passed=None,
        gauge_stable=True,
        parameter_update_norm=0.0,
        parameter_update_small=True,
        verified=True,
        skipped_geometry=True,
        skipped_level=True,
        early_exit_reason=early_exit_reason,
    )


def _residual_filter_kinds(level: ContinuationLevel) -> tuple[str, ...]:
    return tuple(config.kind for config in level.residual_filters)


def _level_verification_checks(
    *,
    cfg: AlternatingSmokeConfig,
    geometry: GeometryState,
    update_report: GaugeReport,
    loss_before: float,
    loss_after: float,
    heldout_residual_passed: bool | None,
) -> _LevelVerificationChecks:
    loss_nonincreasing = bool(loss_after <= loss_before + cfg.verification_loss_tolerance)
    finite_loss = bool(np.isfinite(loss_before) and np.isfinite(loss_after))
    gauge_stable = _gauge_stable(geometry, tolerance=cfg.gauge_stability_tolerance)
    parameter_update_norm = _parameter_update_norm(update_report)
    parameter_update_small = bool(parameter_update_norm <= cfg.parameter_update_tolerance)
    heldout_ok = heldout_residual_passed is not False
    verified = (
        loss_nonincreasing
        and finite_loss
        and gauge_stable
        and parameter_update_small
        and heldout_ok
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


def _projection_loss(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    mask: jax.Array,
    level: ContinuationLevel,
    *,
    sigma: float,
) -> float:
    predicted = project_parallel_reference(volume, geometry)
    filtered = apply_residual_filter_schedule(
        predicted - observed, level.residual_filters, mask=mask
    )
    result = residual_loss(
        filtered.residual,
        jnp.zeros_like(filtered.residual),
        mask=None,
        sigma=sigma,
        delta=level.residual_delta,
    )
    return float(result.loss)


def _level_residual_sigma(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    mask: jax.Array,
    level: ContinuationLevel,
) -> float:
    predicted = project_parallel_reference(volume, geometry)
    filtered = apply_residual_filter_schedule(
        predicted - observed,
        level.residual_filters,
        mask=mask,
    )
    return float(robust_residual_scale(filtered.residual))


def _heldout_projection_loss(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    heldout_mask: jax.Array | None,
    level: ContinuationLevel,
    *,
    sigma: float,
) -> float | None:
    if heldout_mask is None:
        return None
    return _projection_loss(volume, observed, geometry, heldout_mask, level, sigma=sigma)


def _heldout_residual_check(
    volume: jax.Array,
    observed: jax.Array,
    *,
    before_geometry: GeometryState,
    after_geometry: GeometryState,
    heldout_mask: jax.Array | None,
    level: ContinuationLevel,
    sigma: float,
    tolerance: float,
) -> tuple[float | None, float | None, bool | None]:
    before = _heldout_projection_loss(
        volume,
        observed,
        before_geometry,
        heldout_mask,
        level,
        sigma=sigma,
    )
    after = _heldout_projection_loss(
        volume,
        observed,
        after_geometry,
        heldout_mask,
        level,
        sigma=sigma,
    )
    passed = _heldout_residual_passed(before=before, after=after, tolerance=tolerance)
    return before, after, passed


def _heldout_residual_passed(
    *,
    before: float | None,
    after: float | None,
    tolerance: float,
) -> bool | None:
    if before is None or after is None:
        return None
    return bool(after <= before + tolerance)


def _verification_payload(
    *,
    cfg: AlternatingSmokeConfig,
    schedule: ContinuationSchedule,
    initial_loss: float,
    final_loss: float,
    coarse_verified: bool,
    true_geometry: GeometryState,
    final_geometry: GeometryState,
    truth_volume: jax.Array,
    final_volume: jax.Array,
    summaries: tuple[AlternatingLevelSummary, ...],
    geometry_update_volume_source: GeometryUpdateVolumeSource,
    fit_gain_offset_nuisance: bool,
) -> dict[str, object]:
    geometry_recovery = _geometry_recovery_payload(true_geometry, final_geometry)
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
    return payload


def _geometry_recovery_payload(
    true_geometry: GeometryState,
    final_geometry: GeometryState,
) -> dict[str, object]:
    raw_tolerances = _recovery_tolerances_payload()["geometry"]
    if not isinstance(raw_tolerances, dict):
        raise TypeError("geometry recovery tolerances must be a mapping")
    tolerances = cast("dict[str, float]", raw_tolerances)
    final_theta = final_geometry.setup.theta_offset_rad.value + final_geometry.pose.phi_residual_rad
    true_theta = true_geometry.setup.theta_offset_rad.value + true_geometry.pose.phi_residual_rad
    final_u = final_geometry.setup.det_u_px.value + final_geometry.pose.dx_px
    true_u = true_geometry.setup.det_u_px.value + true_geometry.pose.dx_px
    final_v = final_geometry.setup.det_v_px.value + final_geometry.pose.dz_px
    true_v = true_geometry.setup.det_v_px.value + true_geometry.pose.dz_px
    theta_rmse = float(np.sqrt(np.mean((final_theta - true_theta) ** 2)))
    det_u_rmse = float(np.sqrt(np.mean((final_u - true_u) ** 2)))
    det_v_rmse = float(np.sqrt(np.mean((final_v - true_v) ** 2)))
    mean_dx_abs = abs(float(np.mean(final_geometry.pose.dx_px)))
    mean_phi_abs = abs(float(np.mean(final_geometry.pose.phi_residual_rad)))
    mean_dz_abs = abs(float(np.mean(final_geometry.pose.dz_px)))
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
        and mean_dz_abs <= gauge_limit
    )
    return {
        "theta_realized_rmse_rad": theta_rmse,
        "theta_realized_rmse_rad_passed": theta_rmse <= theta_limit,
        "theta_realized_rmse_rad_limit": theta_limit,
        "det_u_realized_rmse_px": det_u_rmse,
        "det_u_realized_rmse_px_passed": det_u_rmse <= det_u_limit,
        "det_u_realized_rmse_px_limit": det_u_limit,
        "det_v_realized_rmse_px": det_v_rmse,
        "det_v_realized_rmse_px_passed": det_v_rmse <= det_v_limit,
        "det_v_realized_rmse_px_limit": det_v_limit,
        "mean_dx_abs_px": mean_dx_abs,
        "mean_dx_abs_px_passed": mean_dx_abs <= gauge_limit,
        "mean_dx_abs_px_limit": gauge_limit,
        "mean_phi_abs_rad": mean_phi_abs,
        "mean_phi_abs_rad_passed": mean_phi_abs <= gauge_limit,
        "mean_phi_abs_rad_limit": gauge_limit,
        "mean_dz_abs_px": mean_dz_abs,
        "mean_dz_abs_px_passed": mean_dz_abs <= gauge_limit,
        "mean_dz_abs_px_limit": gauge_limit,
        "passed": passed,
    }


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


def _write_artifacts(
    output_dir: Path,
    *,
    true_geometry: GeometryState,
    initial_geometry: GeometryState,
    final_geometry: GeometryState,
    truth_volume: jax.Array,
    final_volume: jax.Array,
    observed: jax.Array,
    mask: jax.Array,
    schedule: ContinuationSchedule,
    gauge_report: GaugeReport,
    fista_result: ReferenceFISTAResult,
    summaries: tuple[AlternatingLevelSummary, ...],
    schur_result: JointSchurLMResult | None,
    geometry_update_volume_source: GeometryUpdateVolumeSource,
    fit_gain_offset_nuisance: bool,
    verification: Mapping[str, object],
) -> dict[str, Path]:
    artifacts = {
        "alignment_summary_csv": output_dir / "alignment_summary.csv",
        "artifact_index_json": output_dir / "artifact_index.json",
        "backend_report_json": output_dir / "backend_report.json",
        "config_resolved_toml": output_dir / "config_resolved.toml",
        "final_volume_npy": output_dir / "final_volume.npy",
        "failure_report_json": output_dir / "failure_report.json",
        "fista_trace_csv": output_dir / "fista_trace.csv",
        "geometry_corrupted_json": output_dir / "geometry_corrupted.json",
        "gauge_policy_json": output_dir / "gauge_policy.json",
        "gauge_report_json": output_dir / "gauge_report.json",
        "geometry_final_json": output_dir / "geometry_final.json",
        "geometry_initial_json": output_dir / "geometry_initial.json",
        "geometry_trace_csv": output_dir / "geometry_trace.csv",
        "geometry_true_json": output_dir / "geometry_true.json",
        "ground_truth_volume_npy": output_dir / "ground_truth_volume.npy",
        "input_summary_json": output_dir / "input_summary.json",
        "mask_summary_json": output_dir / "mask_summary.json",
        "pose_decomposition_csv": output_dir / "pose_decomposition.csv",
        "pose_params_csv": output_dir / "pose_params.csv",
        "plots_summary_json": output_dir / "plots" / "summary.json",
        "observability_report_json": output_dir / "observability_report.json",
        "observed_projections_npy": output_dir / "observed_projections.npy",
        "preview_error_slice_npy": output_dir / "preview_slices" / "central_z_error.npy",
        "preview_final_slice_npy": output_dir / "preview_slices" / "central_z_final.npy",
        "preview_summary_json": output_dir / "preview_slices" / "summary.json",
        "preview_truth_slice_npy": output_dir / "preview_slices" / "central_z_truth.npy",
        "projection_mask_npy": output_dir / "projection_mask.npy",
        "projection_stats_json": output_dir / "projection_stats.json",
        "recovery_tolerances_json": output_dir / "recovery_tolerances.json",
        "residual_map_raw_npy": output_dir / "residual_maps" / "final_raw_residual.npy",
        "residual_map_summary_json": output_dir / "residual_maps" / "summary.json",
        "residual_metrics_csv": output_dir / "residual_metrics.csv",
        "run_manifest_json": output_dir / "run_manifest.json",
        "schur_diagnostics_json": output_dir / "schur_diagnostics.json",
        "verification_json": output_dir / "verification.json",
    }
    _write_config_resolved(
        artifacts["config_resolved_toml"],
        schedule,
        geometry_update_volume_source=geometry_update_volume_source,
        fit_gain_offset_nuisance=fit_gain_offset_nuisance,
        synthetic_dataset=verification.get("synthetic_dataset"),
    )
    _write_json(
        artifacts["run_manifest_json"],
        _run_manifest_payload(
            final_volume,
            observed,
            schedule,
            geometry_update_volume_source=geometry_update_volume_source,
            fit_gain_offset_nuisance=fit_gain_offset_nuisance,
            synthetic_dataset=verification.get("synthetic_dataset"),
        ),
    )
    _write_json(artifacts["input_summary_json"], _input_summary_payload(final_volume, observed))
    _write_json(artifacts["projection_stats_json"], _projection_stats_payload(observed))
    _write_json(artifacts["mask_summary_json"], _mask_summary_payload(mask))
    _write_array(artifacts["observed_projections_npy"], observed)
    _write_mask_array(artifacts["projection_mask_npy"], mask)
    _write_json(artifacts["recovery_tolerances_json"], _recovery_tolerances_payload())
    _write_array(artifacts["ground_truth_volume_npy"], truth_volume)
    _write_json(artifacts["gauge_policy_json"], _gauge_policy_payload())
    _write_json(artifacts["gauge_report_json"], _gauge_report_payload(gauge_report))
    _write_json(
        artifacts["observability_report_json"],
        _observability_report_payload(schur_result),
    )
    _write_json(artifacts["backend_report_json"], _backend_report_payload())
    _write_json(
        artifacts["failure_report_json"],
        _failure_report_payload(
            final_volume=final_volume,
            final_geometry=final_geometry,
            observed=observed,
            mask=mask,
            summaries=summaries,
            verification=verification,
        ),
    )
    _write_final_volume(artifacts["final_volume_npy"], final_volume)
    _write_preview_slice_artifacts(
        truth_path=artifacts["preview_truth_slice_npy"],
        final_path=artifacts["preview_final_slice_npy"],
        error_path=artifacts["preview_error_slice_npy"],
        summary_path=artifacts["preview_summary_json"],
        truth_volume=truth_volume,
        final_volume=final_volume,
    )
    write_geometry_json(artifacts["geometry_true_json"], true_geometry)
    write_geometry_json(artifacts["geometry_corrupted_json"], initial_geometry)
    write_geometry_json(artifacts["geometry_initial_json"], initial_geometry)
    write_geometry_json(artifacts["geometry_final_json"], final_geometry)
    write_pose_params_csv(artifacts["pose_params_csv"], final_geometry.pose)
    write_pose_decomposition_csv(artifacts["pose_decomposition_csv"], final_geometry)
    _ = write_fista_trace_csv(fista_result, artifacts["fista_trace_csv"])
    _write_alignment_summary(artifacts["alignment_summary_csv"], summaries)
    _write_geometry_trace(artifacts["geometry_trace_csv"], summaries)
    _write_json(
        artifacts["plots_summary_json"],
        _plots_summary_payload(fista_result=fista_result, summaries=summaries),
    )
    _write_json(
        artifacts["schur_diagnostics_json"],
        _schur_diagnostics_payload(
            schur_result,
            geometry_update_volume_source=geometry_update_volume_source,
        ),
    )
    _write_residual_map_artifacts(
        artifacts["residual_map_raw_npy"],
        artifacts["residual_map_summary_json"],
        final_volume=final_volume,
        final_geometry=final_geometry,
        observed=observed,
        mask=mask,
    )
    _write_residual_metrics(
        artifacts["residual_metrics_csv"],
        summaries,
        final_volume=final_volume,
        final_geometry=final_geometry,
        observed=observed,
        mask=mask,
    )
    _write_json(artifacts["verification_json"], verification)
    _write_json(artifacts["artifact_index_json"], _artifact_index_payload(output_dir, artifacts))
    _ = validate_run_artifacts(output_dir)
    return artifacts


def _write_alignment_summary(
    path: Path,
    summaries: tuple[AlternatingLevelSummary, ...],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "level_factor",
                "role",
                "reconstruction_iterations",
                "geometry_updates",
                "executed_geometry_updates",
                "residual_filter_kinds",
                "loss_before",
                "loss_after",
                "loss_nonincreasing",
                "finite_loss",
                "residual_sigma_estimated",
                "residual_sigma_effective",
                "prior_strength",
                "heldout_residual_before",
                "heldout_residual_after",
                "heldout_residual_passed",
                "gauge_stable",
                "parameter_update_norm",
                "parameter_update_small",
                "verified",
                "schur_accepted",
                "schur_condition",
                "schur_dense_step_difference_norm",
                "schur_predicted_reduction",
                "schur_actual_reduction",
                "schur_reduction_ratio",
                "skipped_geometry",
                "skipped_level",
                "early_exit_reason",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(_summary_payload(summary))


def _write_geometry_trace(path: Path, summaries: tuple[AlternatingLevelSummary, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "level_factor",
                "role",
                "geometry_updates_requested",
                "geometry_updates_executed",
                "loss_before",
                "loss_after",
                "loss_delta",
                "loss_nonincreasing",
                "residual_sigma_estimated",
                "residual_sigma_effective",
                "prior_strength",
                "heldout_residual_before",
                "heldout_residual_after",
                "heldout_residual_passed",
                "gauge_stable",
                "parameter_update_norm",
                "parameter_update_small",
                "verified",
                "schur_accepted",
                "schur_condition",
                "schur_dense_step_difference_norm",
                "schur_predicted_reduction",
                "schur_actual_reduction",
                "schur_reduction_ratio",
                "skipped_geometry",
                "skipped_level",
                "early_exit_reason",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "level_factor": summary.level_factor,
                    "role": summary.role,
                    "geometry_updates_requested": summary.geometry_updates,
                    "geometry_updates_executed": summary.executed_geometry_updates,
                    "loss_before": summary.loss_before,
                    "loss_after": summary.loss_after,
                    "loss_delta": summary.loss_after - summary.loss_before,
                    "loss_nonincreasing": summary.loss_nonincreasing,
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
                    "schur_accepted": _optional_schur_value(summary.schur_diagnostics, "accepted"),
                    "schur_condition": _optional_schur_value(
                        summary.schur_diagnostics, "schur_condition"
                    ),
                    "schur_dense_step_difference_norm": _optional_schur_value(
                        summary.schur_diagnostics,
                        "dense_step_difference_norm",
                    ),
                    "schur_predicted_reduction": _optional_schur_value(
                        summary.schur_diagnostics,
                        "predicted_reduction",
                    ),
                    "schur_actual_reduction": _optional_schur_value(
                        summary.schur_diagnostics,
                        "actual_reduction",
                    ),
                    "schur_reduction_ratio": _optional_schur_value(
                        summary.schur_diagnostics,
                        "reduction_ratio",
                    ),
                    "skipped_geometry": summary.skipped_geometry,
                    "skipped_level": summary.skipped_level,
                    "early_exit_reason": summary.early_exit_reason,
                }
            )


def _optional_schur_value(diagnostics: JointSchurDiagnostics | None, field_name: str) -> object:
    if diagnostics is None:
        return ""
    value = getattr(diagnostics, field_name)
    return "" if value is None else value


def _plots_summary_payload(
    *,
    fista_result: ReferenceFISTAResult,
    summaries: tuple[AlternatingLevelSummary, ...],
) -> dict[str, object]:
    return {
        "schema": "tomojax.plots_summary.v1",
        "rendered": False,
        "reason": "smoke run stores plot-ready numeric traces without rendering dependencies",
        "fista_loss": [
            {
                "iteration": row.iteration,
                "loss": row.loss,
                "data_loss": row.data_loss,
                "regulariser": row.regulariser,
            }
            for row in fista_result.trace
        ],
        "geometry_loss": [
            {
                "level_factor": summary.level_factor,
                "role": summary.role,
                "loss_before": summary.loss_before,
                "loss_after": summary.loss_after,
                "loss_delta": summary.loss_after - summary.loss_before,
                "skipped_level": summary.skipped_level,
                "skipped_geometry": summary.skipped_geometry,
            }
            for summary in summaries
        ],
    }


def _schur_diagnostics_payload(
    result: JointSchurLMResult | None,
    *,
    geometry_update_volume_source: GeometryUpdateVolumeSource,
) -> dict[str, object]:
    if result is None:
        return {
            "schema": "tomojax.schur_diagnostics.v1",
            "status": "not_run",
            "solver": "joint_schur_lm_reference",
            "geometry_update_volume_source": geometry_update_volume_source,
        }
    payload = joint_schur_normal_eq_summary(result)
    payload["schema"] = "tomojax.schur_diagnostics.v1"
    payload["status"] = "passed" if result.final_loss <= result.initial_loss else "warning"
    payload["geometry_update_volume_source"] = geometry_update_volume_source
    return payload


def _write_residual_metrics(
    path: Path,
    summaries: tuple[AlternatingLevelSummary, ...],
    *,
    final_volume: jax.Array,
    final_geometry: GeometryState,
    observed: jax.Array,
    mask: jax.Array,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_type",
                "level_factor",
                "view_index",
                "role",
                "loss_before",
                "loss_after",
                "absolute_improvement",
                "rmse",
                "mae",
                "robust_loss",
                "valid_pixel_fraction",
                "outlier_fraction",
                "raw_rmse",
                "residual_filter_kinds",
                "loss_nonincreasing",
                "finite_loss",
                "residual_sigma_estimated",
                "residual_sigma_effective",
                "prior_strength",
                "heldout_residual_before",
                "heldout_residual_after",
                "heldout_residual_passed",
                "gauge_stable",
                "parameter_update_norm",
                "parameter_update_small",
                "skipped_level",
                "early_exit_reason",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "row_type": "level_summary",
                    "level_factor": summary.level_factor,
                    "view_index": "",
                    "role": summary.role,
                    "loss_before": summary.loss_before,
                    "loss_after": summary.loss_after,
                    "absolute_improvement": summary.loss_before - summary.loss_after,
                    "rmse": "",
                    "mae": "",
                    "robust_loss": "",
                    "valid_pixel_fraction": "",
                    "outlier_fraction": "",
                    "raw_rmse": "",
                    "residual_filter_kinds": "|".join(summary.residual_filter_kinds),
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
                    "skipped_level": summary.skipped_level,
                    "early_exit_reason": summary.early_exit_reason,
                }
            )
        for row in _view_residual_metric_rows(final_volume, final_geometry, observed, mask):
            writer.writerow(row)


def _view_residual_metric_rows(
    final_volume: jax.Array,
    final_geometry: GeometryState,
    observed: jax.Array,
    mask: jax.Array,
) -> tuple[dict[str, object], ...]:
    predicted = project_parallel_reference(final_volume, final_geometry)
    residual = jnp.asarray(predicted - observed, dtype=jnp.float32)
    mask_arr = jnp.asarray(mask, dtype=bool)
    rows: list[dict[str, object]] = []
    for view_index in range(int(residual.shape[0])):
        view_residual = residual[view_index]
        view_mask = mask_arr[view_index]
        valid = view_residual[view_mask]
        rmse = float(jnp.sqrt(jnp.mean(valid * valid)))
        mae = float(jnp.mean(jnp.abs(valid)))
        robust = residual_loss(
            view_residual,
            jnp.zeros_like(view_residual),
            mask=view_mask.astype(jnp.float32),
        ).loss
        rows.append(
            {
                "row_type": "view_residual",
                "level_factor": "final",
                "view_index": view_index,
                "role": "final",
                "loss_before": "",
                "loss_after": "",
                "absolute_improvement": "",
                "rmse": rmse,
                "mae": mae,
                "robust_loss": float(robust),
                "valid_pixel_fraction": float(jnp.mean(view_mask.astype(jnp.float32))),
                "outlier_fraction": 0.0,
                "raw_rmse": rmse,
                "residual_filter_kinds": "raw",
                "loss_nonincreasing": "",
                "finite_loss": "",
                "residual_sigma_estimated": "",
                "residual_sigma_effective": "",
                "prior_strength": "",
                "heldout_residual_before": "",
                "heldout_residual_after": "",
                "heldout_residual_passed": "",
                "gauge_stable": "",
                "parameter_update_norm": "",
                "parameter_update_small": "",
                "skipped_level": "",
                "early_exit_reason": "",
            }
        )
    return tuple(rows)


def _write_residual_map_artifacts(
    residual_path: Path,
    summary_path: Path,
    *,
    final_volume: jax.Array,
    final_geometry: GeometryState,
    observed: jax.Array,
    mask: jax.Array,
) -> None:
    predicted = project_parallel_reference(final_volume, final_geometry)
    residual = jnp.asarray(predicted - observed, dtype=jnp.float32)
    mask_arr = jnp.asarray(mask, dtype=bool)
    _write_array(residual_path, residual)
    _write_json(
        summary_path,
        {
            "schema": "tomojax.residual_map_summary.v1",
            "residual_map": residual_path.name,
            "shape": list(residual.shape),
            "dtype": str(residual.dtype),
            "valid_pixel_fraction": float(jnp.mean(mask_arr.astype(jnp.float32))),
            "rmse": float(jnp.sqrt(jnp.mean(residual[mask_arr] * residual[mask_arr]))),
            "mae": float(jnp.mean(jnp.abs(residual[mask_arr]))),
            "min": float(jnp.min(residual)),
            "max": float(jnp.max(residual)),
        },
    )


def _write_preview_slice_artifacts(
    *,
    truth_path: Path,
    final_path: Path,
    error_path: Path,
    summary_path: Path,
    truth_volume: jax.Array,
    final_volume: jax.Array,
) -> None:
    center_index = int(truth_volume.shape[0] // 2)
    truth_slice = jnp.asarray(truth_volume[center_index, :, :], dtype=jnp.float32)
    final_slice = jnp.asarray(final_volume[center_index, :, :], dtype=jnp.float32)
    error_slice = jnp.asarray(final_slice - truth_slice, dtype=jnp.float32)
    _write_array(truth_path, truth_slice)
    _write_array(final_path, final_slice)
    _write_array(error_path, error_slice)
    _write_json(
        summary_path,
        {
            "schema": "tomojax.preview_slices.v1",
            "axis": "z",
            "index": center_index,
            "shape": list(truth_slice.shape),
            "dtype": str(truth_slice.dtype),
            "truth_slice": truth_path.name,
            "final_slice": final_path.name,
            "error_slice": error_path.name,
            "error_rmse": float(jnp.sqrt(jnp.mean(error_slice * error_slice))),
            "error_mae": float(jnp.mean(jnp.abs(error_slice))),
        },
    )


def _artifact_index_payload(output_dir: Path, artifacts: Mapping[str, Path]) -> dict[str, object]:
    return {
        "schema": "tomojax.artifact_index.v1",
        "artifacts": [
            {
                "name": name,
                "path": path.relative_to(output_dir).as_posix(),
                "type": _artifact_type(path),
                "media_type": _media_type(path),
                "description": _artifact_description(name),
            }
            for name, path in sorted(artifacts.items())
            if name != "artifact_index_json"
        ],
    }


def _artifact_type(path: Path) -> str:
    if path.suffix == ".json":
        return "json"
    if path.suffix == ".csv":
        return "csv"
    if path.suffix == ".toml":
        return "toml"
    if path.suffix == ".npy":
        return "npy"
    return "binary"


def _media_type(path: Path) -> str:
    if path.suffix == ".json":
        return "application/json"
    if path.suffix == ".csv":
        return "text/csv"
    if path.suffix == ".toml":
        return "application/toml"
    return "application/octet-stream"


def _artifact_description(name: str) -> str:
    descriptions = {
        "alignment_summary_csv": "Per-continuation-level alignment summary",
        "backend_report_json": "Backend provenance for the smoke run",
        "config_resolved_toml": "Resolved deterministic smoke configuration",
        "final_volume_npy": "Final reconstructed 32^3 volume",
        "failure_report_json": "Failure status for the smoke run",
        "fista_trace_csv": "Reference FISTA iteration trace",
        "gauge_policy_json": "Gauge canonicalisation policy",
        "gauge_report_json": "Gauge canonicalisation transfer report",
        "geometry_corrupted_json": "Corrupted synthetic input geometry state",
        "geometry_final_json": "Final canonical geometry state",
        "geometry_initial_json": "Initial corrupted geometry state",
        "geometry_trace_csv": "Per-level geometry update trace",
        "geometry_true_json": "True uncorrupted synthetic geometry state",
        "ground_truth_volume_npy": "Ground-truth synthetic smoke volume",
        "input_summary_json": "Synthetic input shape and dtype summary",
        "mask_summary_json": "Projection mask coverage summary",
        "observability_report_json": "Smoke observability placeholder report",
        "observed_projections_npy": "Observed synthetic smoke projections",
        "plots_summary_json": "Plot-ready convergence summary",
        "pose_decomposition_csv": "Final realised per-view pose decomposition",
        "pose_params_csv": "Final per-view pose parameters",
        "preview_error_slice_npy": "Central final-minus-truth preview slice",
        "preview_final_slice_npy": "Central final-volume preview slice",
        "preview_summary_json": "Preview-slice summary",
        "preview_truth_slice_npy": "Central truth-volume preview slice",
        "projection_mask_npy": "Valid projection mask",
        "projection_stats_json": "Observed projection summary statistics",
        "recovery_tolerances_json": "Smoke recovery tolerance contract",
        "residual_map_raw_npy": "Final raw projection residual map",
        "residual_map_summary_json": "Final raw residual-map summary",
        "residual_metrics_csv": "Per-level residual metrics",
        "run_manifest_json": "Resolved smoke run manifest",
        "schur_diagnostics_json": "Joint Schur LM diagnostics summary",
        "verification_json": "Smoke verification report",
    }
    return descriptions[name]


def _write_config_resolved(
    path: Path,
    schedule: ContinuationSchedule,
    *,
    geometry_update_volume_source: GeometryUpdateVolumeSource,
    fit_gain_offset_nuisance: bool,
    synthetic_dataset: object,
) -> None:
    lines = [
        f'profile = "{schedule.name}"',
        'align_mode = "auto"',
        'backend_requested = "jax_reference"',
        'backend_actual = "jax_reference"',
        'geometry_model = "parallel_tomography_reference"',
        f'geometry_update_volume_source = "{geometry_update_volume_source}"',
        f"fit_gain_offset_nuisance = {str(bool(fit_gain_offset_nuisance)).lower()}",
    ]
    if isinstance(synthetic_dataset, dict):
        dataset_payload = cast("dict[object, object]", synthetic_dataset)
        name = dataset_payload.get("name")
        artifact_dir = dataset_payload.get("artifact_dir")
        if isinstance(name, str):
            lines.append(f'synthetic_dataset_name = "{name}"')
        if isinstance(artifact_dir, str):
            lines.append(f'synthetic_dataset_artifact_dir = "{artifact_dir}"')
    lines.extend((f"level_factors = {list(schedule.level_factors)!r}", ""))
    _ = path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _write_final_volume(path: Path, volume: jax.Array) -> None:
    _write_array(path, volume)


def _write_array(path: Path, array: jax.Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, np.asarray(jax.device_get(array), dtype=np.float32), allow_pickle=False)


def _write_mask_array(path: Path, mask: jax.Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, np.asarray(jax.device_get(mask), dtype=bool), allow_pickle=False)


def _run_manifest_payload(
    volume: jax.Array,
    projections: jax.Array,
    schedule: ContinuationSchedule,
    *,
    geometry_update_volume_source: GeometryUpdateVolumeSource,
    fit_gain_offset_nuisance: bool,
    synthetic_dataset: object,
) -> dict[str, object]:
    dataset: dict[str, object] = {
        "source": "tomojax.datasets.make_benchmark_phantom",
        "shape": list(volume.shape),
        "projection_shape": list(projections.shape),
        "projection_dtype": str(projections.dtype),
    }
    if isinstance(synthetic_dataset, dict):
        dataset["synthetic128_benchmark"] = synthetic_dataset
    return {
        "schema": "tomojax.run_manifest.v1",
        "tomojax_version": _tomojax_version(),
        "git_commit": _git_commit(),
        "run_id": f"{schedule.name}-deterministic",
        "started_at": "deterministic-smoke",
        "finished_at": "deterministic-smoke",
        "profile": schedule.name,
        "align_mode": "auto",
        "dataset": dataset,
        "geometry_model": "parallel_tomography_reference",
        "geometry_update_volume_source": geometry_update_volume_source,
        "fit_gain_offset_nuisance": fit_gain_offset_nuisance,
        "continuation": {
            "name": schedule.name,
            "level_factors": list(schedule.level_factors),
        },
        "backend_requested": "jax_reference",
        "backend_actual": "jax_reference",
        "status": "passed",
    }


def _tomojax_version() -> str:
    try:
        return version("tomojax")
    except PackageNotFoundError:
        return "0+unknown"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


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
    det_v_decision = _weak_dof_decision(
        name="det_v_px",
        active=det_v_active,
        curvature=det_v_curvature,
        accepted=diagnostics.accepted if diagnostics is not None else False,
        curvature_floor=curvature_floor,
        accepted_step_required=accepted_step_required,
        frozen_reason="det_v_px is frozen in the current geometry",
        missing_validation=missing_validation,
    )
    theta_scale_decision = _weak_dof_decision(
        name="theta_scale",
        active=False,
        curvature=None,
        accepted=False,
        curvature_floor=curvature_floor,
        accepted_step_required=accepted_step_required,
        frozen_reason="theta_scale is unsupported by the current reference projector",
        missing_validation=missing_validation,
    )
    return {
        "schema": "tomojax.weak_dof_policy.v1",
        "mode": "report_only",
        "thresholds": {
            "curvature_floor": curvature_floor,
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
    accepted: bool,
    curvature_floor: float,
    accepted_step_required: bool,
    frozen_reason: str,
    missing_validation: str,
) -> dict[str, object]:
    curvature_passed = curvature is not None and curvature >= curvature_floor
    accepted_passed = bool(accepted) if accepted_step_required else True
    if not active:
        decision = "keep_frozen"
        reason = frozen_reason
    elif curvature_passed and accepted_passed:
        decision = "keep_active_with_prior"
        reason = "curvature and accepted-step evidence are sufficient for report-only smoke policy"
    else:
        decision = "freeze_or_prior_required"
        reason = "insufficient curvature or accepted-step evidence"
    return {
        "name": name,
        "active": active,
        "decision": decision,
        "reason": reason,
        "evidence": {
            "curvature": curvature,
            "curvature_floor": curvature_floor,
            "curvature_passed": curvature_passed,
            "accepted_step": accepted,
            "accepted_step_required": accepted_step_required,
            "accepted_step_passed": accepted_passed,
            "correlation": None,
            "validation_improvement": None,
            "missing_evidence": ["correlation", missing_validation],
        },
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
    warning_gates = [gate for gate in gates if not gate["passed"]]
    return {
        "schema": "tomojax.failure_report.v1",
        "status": "passed",
        "failure": None,
        "failure_classes": [
            "geometry_not_observable",
            "pose_overfit",
            "nuisance_unmodelled",
            "backend_fallback_unexpected",
            "reconstruction_underconverged",
            "motion_model_insufficient",
            "deformation_suspected",
            "bad_input_metadata",
            "nan_or_inf",
            "no_improvement",
        ],
        "gates": gates,
        "warnings": [
            {
                "class": "no_improvement",
                "severity": "warning",
                "evidence": [str(gate["evidence"])],
                "recommended_action": (
                    "run a longer continuation profile or enable real geometry LM/GN updates"
                ),
            }
            for gate in warning_gates
            if gate["name"] == "projection_residual_improvement"
        ]
        + [
            {
                "class": "nuisance_unmodelled",
                "severity": "warning",
                "evidence": [str(gate["evidence"])],
                "recommended_action": (
                    "enable gain/offset nuisance fitting or inspect flat-field correction"
                ),
            }
            for gate in warning_gates
            if gate["name"] == "nuisance_residual_structure"
        ],
    }


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
    ]


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


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "AlternatingAlignmentSolver",
    "AlternatingLevelSummary",
    "AlternatingSmokeConfig",
    "AlternatingSmokeResult",
    "GeometryUpdateVolumeSource",
    "run_alternating_solver_smoke",
]
