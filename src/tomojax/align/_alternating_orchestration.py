"""Orchestration for the v2 alternating alignment smoke runner."""
# pyright: reportAny=false, reportPrivateUsage=false, reportUnknownMemberType=false
# pyright: reportUnusedFunction=false

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.align._alternating_artifacts import _write_artifacts
from tomojax.align._alternating_geometry_update import (
    _geometry_update_volume_for_level,
    _geometry_updates_for_level,
    _run_geometry_updates,
)
from tomojax.align._alternating_heldout import (
    _heldout_masks,
    _heldout_residual_check,
    _level_residual_sigma,
    _projection_loss,
)
from tomojax.align._alternating_inputs import build_smoke_inputs
from tomojax.align._alternating_level_helpers import (
    _coarse_verification_state,
    _residual_filter_kinds,
    _skipped_level_summary,
)
from tomojax.align._alternating_types import (
    AlternatingBootstrapSummary,
    AlternatingLevelSummary,
    AlternatingSmokeConfig,
    AlternatingSmokeResult,
    PreviewInitialization,
    PreviewResidualFilterMode,
    PreviewVolumeSupport,
)
from tomojax.align._alternating_verification import (
    _level_verification_checks,
    _verification_payload,
)
from tomojax.align._continuation import reference_continuation_schedule
from tomojax.forward import ResidualFilterConfig
from tomojax.geometry import GaugeReport, canonicalize_geometry_gauges
from tomojax.recon import (
    ReferenceFISTAConfig,
    ReferenceFISTAResult,
    centered_volume_support,
    fista_reconstruct_reference,
    reconstruct_average_reference,
    reconstruct_backprojection_reference,
)

if TYPE_CHECKING:
    from tomojax.align._continuation import ContinuationLevel
    from tomojax.align._joint_schur_lm import JointSchurLMResult
    from tomojax.geometry import GeometryState


@dataclass(frozen=True)
class _GeometryFirstBootstrapResult:
    geometry: GeometryState
    gauge_report: GaugeReport
    volume: jax.Array
    schur_result: JointSchurLMResult
    summary: AlternatingBootstrapSummary


def _run_alternating_solver_smoke_impl(  # noqa: PLR0915 - orchestrates level state
    output_dir: str | Path,
    *,
    config: AlternatingSmokeConfig,
) -> AlternatingSmokeResult:
    run_start = time.perf_counter()
    schedule = config.schedule or reference_continuation_schedule()
    out_dir = _prepare_output_dir(output_dir)

    inputs = build_smoke_inputs(config)
    truth, observed, mask = inputs.truth_volume, inputs.observed_projections, inputs.mask
    true_geometry, initial_geometry = inputs.true_geometry, inputs.initial_geometry
    train_mask, heldout_mask = _heldout_masks(mask, config.heldout_view_index)

    geometry, gauge_report = initial_geometry, GaugeReport(())
    volume: jax.Array | None = None
    summaries: list[AlternatingLevelSummary] = []
    bootstrap_summary: AlternatingBootstrapSummary | None = None
    fista_result: ReferenceFISTAResult | None = None
    last_schur_result: JointSchurLMResult | None = None
    constrained_first_preview_volume: jax.Array | None = None
    coarse_verified = False
    time_to_verified_geometry_seconds = None

    initial_loss = previous_loss = _projection_loss(
        truth,
        observed,
        geometry,
        mask,
        schedule.levels[0],
        sigma=schedule.levels[0].residual_sigma,
    )
    (
        geometry,
        gauge_report,
        volume,
        last_schur_result,
        bootstrap_summary,
    ) = _apply_initial_geometry_first_bootstrap(
        config,
        observed,
        geometry,
        mask=train_mask,
        level=schedule.levels[0],
        gauge_report=gauge_report,
        volume=volume,
        last_schur_result=last_schur_result,
    )
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
            initial_volume=_preview_initial_volume(
                config,
                observed,
                geometry,
                level=level,
                previous_volume=volume,
            ),
            volume_support=_preview_volume_support(
                config,
                level=level,
                shape=(config.size, config.size, config.size),
            ),
            mask=_preview_reconstruction_mask(config, mask=mask, train_mask=train_mask),
            config=ReferenceFISTAConfig(
                iterations=_preview_reconstruction_iterations(config, level),
                step_size=_preview_fista_step_size(config),
                tv_weight=level.reconstruction_tv_weight * max(config.preview_tv_scale, 0.0),
                residual_sigma=level.residual_sigma,
                residual_delta=level.residual_delta,
                residual_filters=_preview_residual_filters(
                    config,
                    level,
                    level.residual_filters,
                ),
                center_l2_weight=max(float(config.preview_center_l2_weight), 0.0),
            ),
        )
        volume = jax.lax.stop_gradient(fista_result.volume)
        if _captures_constrained_first_preview_volume(config, level):
            constrained_first_preview_volume = volume
        residual_sigma_estimated = _level_residual_sigma(volume, observed, geometry, mask, level)
        residual_sigma_effective = _effective_residual_sigma(
            config,
            level=level,
            estimated=residual_sigma_estimated,
        )
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
            geometry_update_volume = _geometry_update_volume_for_level(
                truth_volume=truth,
                stopped_volume=_stopped_geometry_update_volume(
                    config,
                    level,
                    current_volume=volume,
                    constrained_first_preview_volume=constrained_first_preview_volume,
                ),
                observed=observed,
                mask=train_mask,
                level=level,
                source=config.geometry_update_volume_source,
                active_setup_parameters=config.geometry_update_active_setup_parameters,
            )
            geometry, update_report, schur_result = _run_geometry_updates(
                geometry_update_volume,
                observed,
                geometry,
                train_mask,
                level,
                geometry_updates,
                sigma=residual_sigma_effective,
                setup_prior_strength=config.geometry_update_setup_prior_strength,
                pose_prior_strength=config.geometry_update_pose_prior_strength,
                pose_trust_radius=config.geometry_update_pose_trust_radius,
                active_setup_parameters=_active_setup_parameters_for_level(
                    config,
                    level.level_factor,
                ),
                solver=config.geometry_update_solver,
                pose_frozen=_pose_frozen_for_level(config, level.level_factor),
                active_pose_dofs=_active_pose_dofs_for_level(config, level.level_factor),
                fit_gain_offset_nuisance=config.fit_gain_offset_nuisance,
                fit_background_nuisance=config.fit_background_nuisance,
                residual_filters=_geometry_residual_filters(config, level.residual_filters),
                parameter_prior_strength=_geometry_parameter_prior_strength(config, level),
            )
            geometry, update_report, schur_result, volume = _apply_geometry_acceptance(
                config,
                volume,
                geometry_update_volume,
                observed,
                before_geometry=geometry_before_update,
                candidate_geometry=geometry,
                train_mask=train_mask,
                heldout_mask=heldout_mask,
                level=level,
                sigma=residual_sigma_effective,
                update_report=update_report,
                schur_result=schur_result,
            )
            gauge_report = update_report
            last_schur_result = schur_result
            loss_before, loss_after = _schur_loss_pair(schur_result)
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
            geometry_update_accepted=(
                None if schur_result is None else schur_result.diagnostics.accepted
            ),
        )
        previous_loss = loss_after
        coarse_verified, time_to_verified_geometry_seconds = _coarse_verification_state(
            level=level,
            checks_verified=(
                _allows_coarse_early_exit(config) and verification_checks.verified
            ),
            already_verified=coarse_verified,
            previous_time_to_verified=time_to_verified_geometry_seconds,
            run_start=run_start,
        )
        summaries.append(
            AlternatingLevelSummary(
                level_factor=level.level_factor,
                role=level.role,
                reconstruction_iterations=_preview_reconstruction_iterations(config, level),
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

    geometry, gauge_report, last_schur_result = _run_final_polishes(
        config,
        schedule.levels[-1],
        summaries=summaries,
        truth_volume=truth,
        stopped_volume=volume,
        observed=observed,
        train_mask=train_mask,
        full_mask=mask,
        heldout_mask=heldout_mask,
        geometry=geometry,
        gauge_report=gauge_report,
        last_schur_result=last_schur_result,
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
        bootstrap_summary=bootstrap_summary,
        schur_result=last_schur_result,
        geometry_update_volume_source=config.geometry_update_volume_source,
        geometry_update_solver=config.geometry_update_solver,
        geometry_update_setup_prior_strength=config.geometry_update_setup_prior_strength,
        geometry_update_pose_prior_strength=config.geometry_update_pose_prior_strength,
        geometry_update_pose_trust_radius=config.geometry_update_pose_trust_radius,
        geometry_update_pose_frozen=config.geometry_update_pose_frozen,
        geometry_update_pose_activate_at_level_factor=(
            config.geometry_update_pose_activate_at_level_factor
        ),
        geometry_update_alpha_beta_activate_at_level_factor=(
            config.geometry_update_alpha_beta_activate_at_level_factor
        ),
        geometry_update_theta_activate_at_level_factor=(
            config.geometry_update_theta_activate_at_level_factor
        ),
        geometry_update_phi_polish_updates=config.geometry_update_phi_polish_updates,
        geometry_update_final_pose_polish_updates=(
            config.geometry_update_final_pose_polish_updates
        ),
        geometry_update_active_setup_parameters=config.geometry_update_active_setup_parameters,
        geometry_update_active_pose_dofs=config.geometry_update_active_pose_dofs,
        preview_volume_support=config.preview_volume_support,
        preview_initialization=config.preview_initialization,
        preview_reconstruction_mask_source=config.preview_reconstruction_mask_source,
        preview_tv_scale=config.preview_tv_scale,
        preview_residual_filter_mode=config.preview_residual_filter_mode,
        preview_center_l2_weight=config.preview_center_l2_weight,
        stopped_preview_policy=config.stopped_preview_policy,
        fit_gain_offset_nuisance=config.fit_gain_offset_nuisance,
        fit_background_nuisance=config.fit_background_nuisance,
        verification=_verification_payload(
            cfg=config,
            schedule=schedule,
            initial_loss=initial_loss,
            final_loss=_final_loss(last_schur_result, summaries),
            coarse_verified=coarse_verified,
            true_geometry=true_geometry,
            initial_geometry=initial_geometry,
            final_geometry=geometry,
            truth_volume=truth,
            final_volume=volume,
            observed=observed,
            mask=mask,
            summaries=tuple(summaries),
            geometry_update_volume_source=config.geometry_update_volume_source,
            fit_gain_offset_nuisance=config.fit_gain_offset_nuisance,
            fit_background_nuisance=config.fit_background_nuisance,
            time_to_verified_geometry_seconds=time_to_verified_geometry_seconds,
            total_wall_seconds=float(time.perf_counter() - run_start),
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


def _prepare_output_dir(output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _schur_loss_pair(result: JointSchurLMResult) -> tuple[float, float]:
    return result.initial_loss, result.final_loss


def _final_loss(
    last_schur_result: JointSchurLMResult | None,
    summaries: list[AlternatingLevelSummary],
) -> float:
    if last_schur_result is not None:
        return last_schur_result.final_loss
    return summaries[-1].loss_after


def _run_final_polishes(
    config: AlternatingSmokeConfig,
    level: ContinuationLevel,
    *,
    summaries: list[AlternatingLevelSummary],
    truth_volume: jax.Array,
    stopped_volume: jax.Array,
    observed: jax.Array,
    train_mask: jax.Array,
    full_mask: jax.Array,
    heldout_mask: jax.Array | None,
    geometry: GeometryState,
    gauge_report: GaugeReport,
    last_schur_result: JointSchurLMResult | None,
) -> tuple[GeometryState, GaugeReport, JointSchurLMResult | None]:
    geometry, gauge_report, last_schur_result = _maybe_run_polish_stage(
        config,
        level,
        summaries=summaries,
        truth_volume=truth_volume,
        stopped_volume=stopped_volume,
        observed=observed,
        train_mask=train_mask,
        full_mask=full_mask,
        heldout_mask=heldout_mask,
        geometry=geometry,
        gauge_report=gauge_report,
        last_schur_result=last_schur_result,
        role="polish",
        updates=config.geometry_update_phi_polish_updates,
        active_setup_parameters=(),
        active_pose_dofs=("phi_residual_rad",),
    )
    return _run_final_pose_polishes(
        config,
        level,
        summaries=summaries,
        truth_volume=truth_volume,
        stopped_volume=stopped_volume,
        observed=observed,
        train_mask=train_mask,
        full_mask=full_mask,
        heldout_mask=heldout_mask,
        geometry=geometry,
        gauge_report=gauge_report,
        last_schur_result=last_schur_result,
    )


def _run_final_pose_polishes(
    config: AlternatingSmokeConfig,
    level: ContinuationLevel,
    *,
    summaries: list[AlternatingLevelSummary],
    truth_volume: jax.Array,
    stopped_volume: jax.Array,
    observed: jax.Array,
    train_mask: jax.Array,
    full_mask: jax.Array,
    heldout_mask: jax.Array | None,
    geometry: GeometryState,
    gauge_report: GaugeReport,
    last_schur_result: JointSchurLMResult | None,
) -> tuple[GeometryState, GaugeReport, JointSchurLMResult | None]:
    updates = int(config.geometry_update_final_pose_polish_updates)
    first_updates = min(updates, 32)
    geometry, gauge_report, last_schur_result = _maybe_run_final_pose_polish(
        config,
        level,
        summaries=summaries,
        truth_volume=truth_volume,
        stopped_volume=stopped_volume,
        observed=observed,
        train_mask=train_mask,
        full_mask=full_mask,
        heldout_mask=heldout_mask,
        geometry=geometry,
        gauge_report=gauge_report,
        last_schur_result=last_schur_result,
        role="final_pose_polish",
        updates=first_updates,
    )
    return _maybe_run_final_pose_polish(
        config,
        level,
        summaries=summaries,
        truth_volume=truth_volume,
        stopped_volume=stopped_volume,
        observed=observed,
        train_mask=train_mask,
        full_mask=full_mask,
        heldout_mask=heldout_mask,
        geometry=geometry,
        gauge_report=gauge_report,
        last_schur_result=last_schur_result,
        role="final_pose_repolish",
        updates=updates - first_updates,
    )


def _maybe_run_final_pose_polish(
    config: AlternatingSmokeConfig,
    level: ContinuationLevel,
    *,
    summaries: list[AlternatingLevelSummary],
    truth_volume: jax.Array,
    stopped_volume: jax.Array,
    observed: jax.Array,
    train_mask: jax.Array,
    full_mask: jax.Array,
    heldout_mask: jax.Array | None,
    geometry: GeometryState,
    gauge_report: GaugeReport,
    last_schur_result: JointSchurLMResult | None,
    role: str,
    updates: int,
) -> tuple[GeometryState, GaugeReport, JointSchurLMResult | None]:
    return _maybe_run_polish_stage(
        config,
        level,
        summaries=summaries,
        truth_volume=truth_volume,
        stopped_volume=stopped_volume,
        observed=observed,
        train_mask=train_mask,
        full_mask=full_mask,
        heldout_mask=heldout_mask,
        geometry=geometry,
        gauge_report=gauge_report,
        last_schur_result=last_schur_result,
        role=role,
        updates=updates,
        active_setup_parameters=("det_u_px",),
        active_pose_dofs=(
            "alpha_rad",
            "beta_rad",
            "phi_residual_rad",
            "dx_px",
            "dz_px",
        ),
    )


def _maybe_run_polish_stage(
    config: AlternatingSmokeConfig,
    level: ContinuationLevel,
    *,
    summaries: list[AlternatingLevelSummary],
    truth_volume: jax.Array,
    stopped_volume: jax.Array,
    observed: jax.Array,
    train_mask: jax.Array,
    full_mask: jax.Array,
    heldout_mask: jax.Array | None,
    geometry: GeometryState,
    gauge_report: GaugeReport,
    last_schur_result: JointSchurLMResult | None,
    role: str,
    updates: int,
    active_setup_parameters: tuple[str, ...],
    active_pose_dofs: tuple[str, ...],
) -> tuple[GeometryState, GaugeReport, JointSchurLMResult | None]:
    if int(updates) <= 0:
        return geometry, gauge_report, last_schur_result
    summary, geometry, gauge_report, result = _run_polish_stage(
        config,
        level,
        truth_volume=truth_volume,
        stopped_volume=stopped_volume,
        observed=observed,
        train_mask=train_mask,
        full_mask=full_mask,
        heldout_mask=heldout_mask,
        geometry=geometry,
        role=role,
        updates=updates,
        active_setup_parameters=active_setup_parameters,
        active_pose_dofs=active_pose_dofs,
    )
    summaries.append(summary)
    return geometry, gauge_report, result


def _run_phi_polish(
    config: AlternatingSmokeConfig,
    level: ContinuationLevel,
    *,
    truth_volume: jax.Array,
    stopped_volume: jax.Array,
    observed: jax.Array,
    train_mask: jax.Array,
    full_mask: jax.Array,
    heldout_mask: jax.Array | None,
    geometry: GeometryState,
) -> tuple[AlternatingLevelSummary, GeometryState, GaugeReport, JointSchurLMResult]:
    return _run_polish_stage(
        config,
        level,
        truth_volume=truth_volume,
        stopped_volume=stopped_volume,
        observed=observed,
        train_mask=train_mask,
        full_mask=full_mask,
        heldout_mask=heldout_mask,
        geometry=geometry,
        role="polish",
        updates=config.geometry_update_phi_polish_updates,
        active_setup_parameters=(),
        active_pose_dofs=("phi_residual_rad",),
    )


def _run_polish_stage(
    config: AlternatingSmokeConfig,
    level: ContinuationLevel,
    *,
    truth_volume: jax.Array,
    stopped_volume: jax.Array,
    observed: jax.Array,
    train_mask: jax.Array,
    full_mask: jax.Array,
    heldout_mask: jax.Array | None,
    geometry: GeometryState,
    role: str,
    updates: int,
    active_setup_parameters: tuple[str, ...],
    active_pose_dofs: tuple[str, ...],
) -> tuple[AlternatingLevelSummary, GeometryState, GaugeReport, JointSchurLMResult]:
    update_count = int(updates)
    volume = _geometry_update_volume_for_level(
        truth_volume=truth_volume,
        stopped_volume=stopped_volume,
        observed=observed,
        mask=train_mask,
        level=level,
        source=config.geometry_update_volume_source,
        active_setup_parameters=active_setup_parameters,
    )
    residual_sigma_estimated = _level_residual_sigma(volume, observed, geometry, full_mask, level)
    residual_sigma_effective = _effective_residual_sigma(
        config,
        level=level,
        estimated=residual_sigma_estimated,
    )
    before_geometry = geometry
    geometry, report, result = _run_geometry_updates(
        volume,
        observed,
        geometry,
        train_mask,
        level,
        update_count,
        sigma=residual_sigma_effective,
        setup_prior_strength=config.geometry_update_setup_prior_strength,
        pose_prior_strength=config.geometry_update_pose_prior_strength,
        pose_trust_radius=config.geometry_update_pose_trust_radius,
        active_setup_parameters=active_setup_parameters,
        solver=config.geometry_update_solver,
        pose_frozen=False,
        active_pose_dofs=active_pose_dofs,
        fit_gain_offset_nuisance=config.fit_gain_offset_nuisance,
        fit_background_nuisance=config.fit_background_nuisance,
        residual_filters=_geometry_residual_filters(config, level.residual_filters),
        parameter_prior_strength=_geometry_parameter_prior_strength(config, level),
    )
    geometry, report, result = _apply_heldout_geometry_acceptance(
        config,
        volume,
        observed,
        before_geometry=before_geometry,
        candidate_geometry=geometry,
        heldout_mask=heldout_mask,
        level=level,
        sigma=residual_sigma_effective,
        update_report=report,
        schur_result=result,
    )
    loss_before, loss_after = _schur_loss_pair(result)
    heldout_before, heldout_after, heldout_passed = _heldout_residual_check(
        volume,
        observed,
        before_geometry=before_geometry,
        after_geometry=geometry,
        heldout_mask=heldout_mask,
        level=level,
        sigma=residual_sigma_effective,
        tolerance=config.heldout_residual_tolerance,
    )
    checks = _level_verification_checks(
        cfg=config,
        geometry=geometry,
        update_report=report,
        loss_before=loss_before,
        loss_after=loss_after,
        heldout_residual_passed=heldout_passed,
        geometry_update_accepted=result.diagnostics.accepted,
    )
    return (
        AlternatingLevelSummary(
            level_factor=level.level_factor,
            role=role,
            reconstruction_iterations=0,
            geometry_updates=update_count,
            executed_geometry_updates=update_count,
            residual_filter_kinds=_residual_filter_kinds(level),
            loss_before=loss_before,
            loss_after=loss_after,
            loss_nonincreasing=checks.loss_nonincreasing,
            finite_loss=checks.finite_loss,
            residual_sigma_estimated=residual_sigma_estimated,
            residual_sigma_effective=residual_sigma_effective,
            prior_strength=level.prior_strength,
            heldout_residual_before=heldout_before,
            heldout_residual_after=heldout_after,
            heldout_residual_passed=heldout_passed,
            gauge_stable=checks.gauge_stable,
            parameter_update_norm=checks.parameter_update_norm,
            parameter_update_small=checks.parameter_update_small,
            verified=checks.verified,
            skipped_geometry=False,
            skipped_level=False,
            early_exit_reason=None,
            schur_diagnostics=result.diagnostics,
        ),
        geometry,
        report,
        result,
    )


def _pose_frozen_for_level(config: AlternatingSmokeConfig, level_factor: int) -> bool:
    if config.geometry_update_pose_frozen:
        return True
    activate_at = config.geometry_update_pose_activate_at_level_factor
    return activate_at is not None and int(level_factor) > int(activate_at)


def _active_setup_parameters_for_level(
    config: AlternatingSmokeConfig,
    level_factor: int,
) -> tuple[str, ...]:
    activate_at = config.geometry_update_theta_activate_at_level_factor
    if activate_at is None or int(level_factor) <= int(activate_at):
        return config.geometry_update_active_setup_parameters
    return tuple(
        name
        for name in config.geometry_update_active_setup_parameters
        if name != "theta_offset_rad"
    )


def _active_pose_dofs_for_level(
    config: AlternatingSmokeConfig,
    level_factor: int,
) -> tuple[str, ...]:
    activate_at = config.geometry_update_alpha_beta_activate_at_level_factor
    if activate_at is None or int(level_factor) <= int(activate_at):
        return config.geometry_update_active_pose_dofs
    return tuple(
        name
        for name in config.geometry_update_active_pose_dofs
        if name not in {"alpha_rad", "beta_rad"}
    )


def _preview_volume_support(
    config: AlternatingSmokeConfig,
    level: ContinuationLevel,
    *,
    shape: tuple[int, int, int],
) -> jax.Array | None:
    support = _effective_preview_volume_support(config, level)
    if support == "none":
        return None
    return centered_volume_support(shape, kind=support)


def _preview_fista_step_size(config: AlternatingSmokeConfig) -> float:
    if int(config.size) < 64:
        return 2.0e-3
    return 100.0 * max(float(config.size), 1.0) / 128.0


def _preview_reconstruction_iterations(
    config: AlternatingSmokeConfig,
    level: ContinuationLevel,
) -> int:
    if _uses_no_fista_first_level(config, level):
        return 0
    return level.reconstruction_iterations


def _preview_reconstruction_mask(
    config: AlternatingSmokeConfig,
    *,
    mask: jax.Array,
    train_mask: jax.Array,
) -> jax.Array:
    if config.preview_reconstruction_mask_source == "all_views":
        return mask
    if config.preview_reconstruction_mask_source == "train_views":
        return train_mask
    raise ValueError(
        "unknown preview reconstruction mask source "
        f"{config.preview_reconstruction_mask_source!r}"
    )


def _effective_residual_sigma(
    config: AlternatingSmokeConfig,
    *,
    level: ContinuationLevel,
    estimated: float,
) -> float:
    if config.geometry_update_volume_source == "fixed_synthetic_truth":
        return float(level.residual_sigma)
    return max(float(level.residual_sigma), float(estimated))


def _allows_coarse_early_exit(config: AlternatingSmokeConfig) -> bool:
    return (
        config.geometry_update_volume_source != "fixed_synthetic_truth"
        and config.preview_reconstruction_mask_source != "train_views"
    )


def _apply_initial_geometry_first_bootstrap(
    config: AlternatingSmokeConfig,
    observed: jax.Array,
    geometry: GeometryState,
    *,
    mask: jax.Array,
    level: ContinuationLevel,
    gauge_report: GaugeReport,
    volume: jax.Array | None,
    last_schur_result: JointSchurLMResult | None,
) -> tuple[
    GeometryState,
    GaugeReport,
    jax.Array | None,
    JointSchurLMResult | None,
    AlternatingBootstrapSummary | None,
]:
    bootstrap = _maybe_run_geometry_first_det_u_bootstrap(
        config,
        observed,
        geometry,
        mask=mask,
        level=level,
    )
    if bootstrap is None:
        return geometry, gauge_report, volume, last_schur_result, None
    return (
        bootstrap.geometry,
        bootstrap.gauge_report,
        jax.lax.stop_gradient(bootstrap.volume),
        bootstrap.schur_result,
        bootstrap.summary,
    )


def _maybe_run_geometry_first_det_u_bootstrap(
    config: AlternatingSmokeConfig,
    observed: jax.Array,
    geometry: GeometryState,
    *,
    mask: jax.Array,
    level: ContinuationLevel,
) -> _GeometryFirstBootstrapResult | None:
    if not _uses_geometry_first_det_u_bootstrap(config, level):
        return None
    shape = (config.size, config.size, config.size)
    support = _preview_volume_support(config, level=level, shape=shape)
    current_volume = jnp.zeros(shape, dtype=jnp.float32)
    neutral = _candidate_refresh_initial_volume(
        config,
        observed,
        current_volume=current_volume,
        level=level,
    )
    loss_before_first_schur = _projection_loss(
        neutral,
        observed,
        geometry,
        mask,
        level,
        sigma=level.residual_sigma,
    )
    first_geometry, _first_report, _first_schur = _run_geometry_updates(
        neutral,
        observed,
        geometry,
        mask,
        level,
        level.geometry_updates,
        sigma=level.residual_sigma,
        setup_prior_strength=config.geometry_update_setup_prior_strength,
        pose_prior_strength=config.geometry_update_pose_prior_strength,
        pose_trust_radius=config.geometry_update_pose_trust_radius,
        active_setup_parameters=("det_u_px",),
        solver=config.geometry_update_solver,
        pose_frozen=True,
        active_pose_dofs=(),
        fit_gain_offset_nuisance=False,
        fit_background_nuisance=False,
        residual_filters=_geometry_residual_filters(config, level.residual_filters),
        parameter_prior_strength=_geometry_parameter_prior_strength(config, level),
    )
    loss_after_first_schur = _projection_loss(
        neutral,
        observed,
        first_geometry,
        mask,
        level,
        sigma=level.residual_sigma,
    )
    refreshed = fista_reconstruct_reference(
        observed,
        first_geometry,
        initial_volume=neutral,
        volume_support=support,
        mask=mask,
        config=ReferenceFISTAConfig(
            iterations=_preview_reconstruction_iterations(config, level),
            step_size=_preview_fista_step_size(config),
            tv_weight=level.reconstruction_tv_weight * max(config.preview_tv_scale, 0.0),
            residual_sigma=level.residual_sigma,
            residual_delta=level.residual_delta,
            residual_filters=_preview_residual_filters(config, level, level.residual_filters),
            center_l2_weight=max(float(config.preview_center_l2_weight), 0.0),
        ),
    )
    loss_after_fista_refresh = _projection_loss(
        refreshed.volume,
        observed,
        first_geometry,
        mask,
        level,
        sigma=level.residual_sigma,
    )
    final_geometry, final_report, final_schur = _run_geometry_updates(
        refreshed.volume,
        observed,
        first_geometry,
        mask,
        level,
        level.geometry_updates,
        sigma=level.residual_sigma,
        setup_prior_strength=config.geometry_update_setup_prior_strength,
        pose_prior_strength=config.geometry_update_pose_prior_strength,
        pose_trust_radius=config.geometry_update_pose_trust_radius,
        active_setup_parameters=("det_u_px",),
        solver=config.geometry_update_solver,
        pose_frozen=True,
        active_pose_dofs=(),
        fit_gain_offset_nuisance=False,
        fit_background_nuisance=False,
        residual_filters=_geometry_residual_filters(config, level.residual_filters),
        parameter_prior_strength=_geometry_parameter_prior_strength(config, level),
    )
    final_loss_before, final_loss_after = _schur_loss_pair(final_schur)
    return _GeometryFirstBootstrapResult(
        geometry=final_geometry,
        gauge_report=final_report,
        volume=jax.lax.stop_gradient(refreshed.volume),
        schur_result=final_schur,
        summary=AlternatingBootstrapSummary(
            level_factor=level.level_factor,
            role="geometry_first_bootstrap",
            schur_updates_per_pass=level.geometry_updates,
            schur_passes=2,
            executed_geometry_updates=2 * level.geometry_updates,
            fista_refresh_iterations=_preview_reconstruction_iterations(config, level),
            residual_filter_kinds=_residual_filter_kinds(level),
            loss_before_first_schur=loss_before_first_schur,
            loss_after_first_schur=loss_after_first_schur,
            loss_before_fista_refresh=loss_after_first_schur,
            loss_after_fista_refresh=loss_after_fista_refresh,
            loss_before_final_schur=final_loss_before,
            loss_after_final_schur=final_loss_after,
            accepted=bool(final_schur.diagnostics.accepted),
            final_det_u_px=float(final_geometry.setup.det_u_px.value),
            parameter_update_norm=max(
                float(final_schur.diagnostics.setup_update_norm),
                float(final_schur.diagnostics.pose_update_norm),
            ),
            first_schur_diagnostics=_first_schur.diagnostics,
            final_schur_diagnostics=final_schur.diagnostics,
        ),
    )


def _uses_geometry_first_det_u_bootstrap(
    config: AlternatingSmokeConfig,
    level: ContinuationLevel,
) -> bool:
    return (
        _uses_production_stopped_det_u_gate(config)
        and level.role == "preview"
        and int(level.level_factor) == 4
    )


def _uses_production_stopped_det_u_gate(config: AlternatingSmokeConfig) -> bool:
    return (
        config.geometry_update_volume_source == "stopped_reconstruction"
        and config.geometry_update_solver == "joint_schur"
        and config.geometry_update_pose_frozen
        and tuple(config.geometry_update_active_setup_parameters) == ("det_u_px",)
        and not config.fit_gain_offset_nuisance
        and not config.fit_background_nuisance
    )


def _preview_initial_volume(
    config: AlternatingSmokeConfig,
    observed: jax.Array,
    geometry: GeometryState,
    *,
    level: ContinuationLevel,
    previous_volume: jax.Array | None,
) -> jax.Array:
    if previous_volume is not None:
        return previous_volume
    initialization = _effective_preview_initialization(config, level)
    if initialization == "backprojection":
        if int(config.size) < 64:
            return reconstruct_average_reference(observed, depth=config.size)
        return reconstruct_backprojection_reference(observed, geometry, depth=config.size)
    if initialization == "zero":
        return jnp.zeros((config.size, config.size, config.size), dtype=jnp.float32)
    if initialization == "constant":
        fill_value = jnp.mean(jnp.asarray(observed, dtype=jnp.float32)) / jnp.asarray(
            max(config.size, 1), dtype=jnp.float32
        )
        return jnp.full((config.size, config.size, config.size), fill_value, dtype=jnp.float32)
    if initialization == "average_projection":
        return reconstruct_average_reference(observed, depth=config.size)
    raise ValueError(f"unknown preview initialization {initialization!r}")


def _preview_residual_filters(
    config: AlternatingSmokeConfig,
    level: ContinuationLevel,
    residual_filters: tuple[ResidualFilterConfig, ...],
) -> tuple[ResidualFilterConfig, ...]:
    mode = _effective_preview_residual_filter_mode(config, level)
    if mode == "continuation":
        return residual_filters
    if mode == "raw":
        return (ResidualFilterConfig(kind="raw"),)
    raise ValueError(f"unknown preview residual filter mode {mode!r}")


def _effective_preview_initialization(
    config: AlternatingSmokeConfig,
    level: ContinuationLevel,
) -> PreviewInitialization:
    if _uses_constant_cylindrical_first_level(config, level):
        return "constant"
    return config.preview_initialization


def _effective_preview_volume_support(
    config: AlternatingSmokeConfig,
    level: ContinuationLevel,
) -> PreviewVolumeSupport:
    if _uses_constant_cylindrical_first_level(config, level):
        return "cylindrical"
    return config.preview_volume_support


def _effective_preview_residual_filter_mode(
    config: AlternatingSmokeConfig,
    level: ContinuationLevel,
) -> PreviewResidualFilterMode:
    if _uses_constant_cylindrical_first_level(config, level):
        return "raw"
    return config.preview_residual_filter_mode


def _uses_constant_cylindrical_first_level(
    config: AlternatingSmokeConfig,
    level: ContinuationLevel,
) -> bool:
    return (
        config.stopped_preview_policy
        in {
            "constant_cylindrical_first_level",
            "constant_cylindrical_first_level_no_fista",
        }
        and config.geometry_update_volume_source == "stopped_reconstruction"
        and level.role == "preview"
        and int(level.level_factor) == 4
    )


def _uses_no_fista_first_level(
    config: AlternatingSmokeConfig,
    level: ContinuationLevel,
) -> bool:
    return (
        config.stopped_preview_policy == "constant_cylindrical_first_level_no_fista"
        and _uses_constant_cylindrical_first_level(config, level)
    )


def _captures_constrained_first_preview_volume(
    config: AlternatingSmokeConfig,
    level: ContinuationLevel,
) -> bool:
    return _uses_constant_cylindrical_first_level(config, level)


def _stopped_geometry_update_volume(
    config: AlternatingSmokeConfig,
    level: ContinuationLevel,
    *,
    current_volume: jax.Array,
    constrained_first_preview_volume: jax.Array | None,
) -> jax.Array:
    if (
        config.stopped_preview_policy
        in {
            "constant_cylindrical_first_level",
            "constant_cylindrical_first_level_no_fista",
        }
        and config.geometry_update_volume_source == "stopped_reconstruction"
        and constrained_first_preview_volume is not None
        and not _uses_constant_cylindrical_first_level(config, level)
    ):
        return constrained_first_preview_volume
    return current_volume


def _apply_geometry_acceptance(
    config: AlternatingSmokeConfig,
    current_volume: jax.Array,
    geometry_update_volume: jax.Array,
    observed: jax.Array,
    *,
    before_geometry: GeometryState,
    candidate_geometry: GeometryState,
    train_mask: jax.Array,
    heldout_mask: jax.Array | None,
    level: ContinuationLevel,
    sigma: float,
    update_report: GaugeReport,
    schur_result: JointSchurLMResult,
) -> tuple[GeometryState, GaugeReport, JointSchurLMResult, jax.Array]:
    if config.geometry_update_volume_source != "stopped_reconstruction":
        geometry, report, result = _apply_heldout_geometry_acceptance(
            config,
            geometry_update_volume,
            observed,
            before_geometry=before_geometry,
            candidate_geometry=candidate_geometry,
            heldout_mask=heldout_mask,
            level=level,
            sigma=sigma,
            update_report=update_report,
            schur_result=schur_result,
        )
        return geometry, report, result, current_volume
    if _uses_production_stopped_det_u_gate(config):
        canonicalized = canonicalize_geometry_gauges(candidate_geometry)
        return (
            canonicalized.state,
            update_report,
            replace(
                schur_result,
                geometry=canonicalized.state,
                canonicalized_geometry=canonicalized,
            ),
            current_volume,
        )
    return _apply_candidate_refresh_acceptance(
        config,
        current_volume,
        observed,
        before_geometry=before_geometry,
        candidate_geometry=candidate_geometry,
        train_mask=train_mask,
        heldout_mask=heldout_mask,
        level=level,
        sigma=sigma,
        update_report=update_report,
        schur_result=schur_result,
    )


def _apply_candidate_refresh_acceptance(
    config: AlternatingSmokeConfig,
    current_volume: jax.Array,
    observed: jax.Array,
    *,
    before_geometry: GeometryState,
    candidate_geometry: GeometryState,
    train_mask: jax.Array,
    heldout_mask: jax.Array | None,
    level: ContinuationLevel,
    sigma: float,
    update_report: GaugeReport,
    schur_result: JointSchurLMResult,
) -> tuple[GeometryState, GaugeReport, JointSchurLMResult, jax.Array]:
    initial_volume = _candidate_refresh_initial_volume(
        config,
        observed,
        current_volume=current_volume,
        level=level,
    )
    before_refresh = _candidate_refresh_volume(
        config,
        observed,
        before_geometry,
        initial_volume=initial_volume,
        train_mask=train_mask,
        level=level,
    )
    candidate_refresh = _candidate_refresh_volume(
        config,
        observed,
        candidate_geometry,
        initial_volume=initial_volume,
        train_mask=train_mask,
        level=level,
    )
    validation_mask = heldout_mask if heldout_mask is not None else train_mask
    before_loss = _projection_loss(
        before_refresh,
        observed,
        before_geometry,
        validation_mask,
        level,
        sigma=sigma,
    )
    candidate_loss = _projection_loss(
        candidate_refresh,
        observed,
        candidate_geometry,
        validation_mask,
        level,
        sigma=sigma,
    )
    accepted = bool(candidate_loss <= before_loss + config.heldout_residual_tolerance)
    selected_geometry = candidate_geometry if accepted else before_geometry
    selected_volume = candidate_refresh if accepted else before_refresh
    canonicalized = canonicalize_geometry_gauges(selected_geometry)
    actual_reduction = float(before_loss - candidate_loss)
    refreshed_diagnostics = (
        replace(schur_result.diagnostics, accepted=True)
        if accepted
        else replace(
            schur_result.diagnostics,
            accepted=False,
            actual_reduction=actual_reduction,
            reduction_ratio=_candidate_refresh_reduction_ratio(
                actual_reduction=actual_reduction,
                predicted_reduction=schur_result.diagnostics.predicted_reduction,
            ),
        )
    )
    refreshed_result = replace(
        schur_result,
        geometry=canonicalized.state,
        canonicalized_geometry=canonicalized,
        final_loss=(schur_result.final_loss if accepted else schur_result.initial_loss),
        diagnostics=refreshed_diagnostics,
    )
    return (
        canonicalized.state,
        canonicalized.report if not accepted else update_report,
        refreshed_result,
        jax.lax.stop_gradient(selected_volume),
    )


def _candidate_refresh_initial_volume(
    config: AlternatingSmokeConfig,
    observed: jax.Array,
    *,
    current_volume: jax.Array,
    level: ContinuationLevel,
) -> jax.Array:
    if len(current_volume.shape) != 3:
        raise ValueError("candidate refresh volume must be 3-D")
    shape = (
        int(current_volume.shape[0]),
        int(current_volume.shape[1]),
        int(current_volume.shape[2]),
    )
    initial = reconstruct_average_reference(observed, depth=shape[1]) / jnp.asarray(
        max(shape[1], 1), dtype=jnp.float32
    )
    support = _preview_volume_support(config, level=level, shape=shape)
    if support is None:
        return initial
    return initial * support


def _candidate_refresh_volume(
    config: AlternatingSmokeConfig,
    observed: jax.Array,
    geometry: GeometryState,
    *,
    initial_volume: jax.Array,
    train_mask: jax.Array,
    level: ContinuationLevel,
) -> jax.Array:
    result = fista_reconstruct_reference(
        observed,
        geometry,
        initial_volume=initial_volume,
        volume_support=_preview_volume_support(
            config,
            level=level,
            shape=(config.size, config.size, config.size),
        ),
        mask=train_mask,
        config=ReferenceFISTAConfig(
            iterations=_candidate_refresh_iterations(level),
            step_size=_preview_fista_step_size(config),
            tv_weight=level.reconstruction_tv_weight * max(config.preview_tv_scale, 0.0),
            residual_sigma=level.residual_sigma,
            residual_delta=level.residual_delta,
            residual_filters=_preview_residual_filters(
                config,
                level,
                level.residual_filters,
            ),
            center_l2_weight=max(float(config.preview_center_l2_weight), 0.0),
        ),
    )
    return jax.lax.stop_gradient(result.volume)


def _candidate_refresh_iterations(level: ContinuationLevel) -> int:
    return max(1, min(4, int(level.reconstruction_iterations)))


def _candidate_refresh_reduction_ratio(
    *,
    actual_reduction: float,
    predicted_reduction: float,
) -> float | None:
    predicted = float(predicted_reduction)
    if not np.isfinite(predicted) or abs(predicted) <= 1.0e-12:
        return None
    return float(actual_reduction / predicted)


def _apply_heldout_geometry_acceptance(
    config: AlternatingSmokeConfig,
    volume: jax.Array,
    observed: jax.Array,
    *,
    before_geometry: GeometryState,
    candidate_geometry: GeometryState,
    heldout_mask: jax.Array | None,
    level: ContinuationLevel,
    sigma: float,
    update_report: GaugeReport,
    schur_result: JointSchurLMResult,
) -> tuple[GeometryState, GaugeReport, JointSchurLMResult]:
    if config.geometry_update_volume_source != "stopped_reconstruction":
        return candidate_geometry, update_report, schur_result
    before = _projection_loss_or_none(
        volume,
        observed,
        before_geometry,
        heldout_mask,
        level,
        sigma=sigma,
    )
    after = _projection_loss_or_none(
        volume,
        observed,
        candidate_geometry,
        heldout_mask,
        level,
        sigma=sigma,
    )
    if before is None or after is None or after <= before + config.heldout_residual_tolerance:
        return candidate_geometry, update_report, schur_result
    canonicalized = canonicalize_geometry_gauges(before_geometry)
    rejected_diagnostics = replace(
        schur_result.diagnostics,
        accepted=False,
        actual_reduction=float(before - after),
        reduction_ratio=None,
    )
    rejected_result = replace(
        schur_result,
        geometry=before_geometry,
        canonicalized_geometry=canonicalized,
        final_loss=schur_result.initial_loss,
        diagnostics=rejected_diagnostics,
    )
    return canonicalized.state, canonicalized.report, rejected_result


def _projection_loss_or_none(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    mask: jax.Array | None,
    level: ContinuationLevel,
    *,
    sigma: float,
) -> float | None:
    if mask is None:
        return None
    return _projection_loss(volume, observed, geometry, mask, level, sigma=sigma)


def _geometry_residual_filters(
    config: AlternatingSmokeConfig,
    residual_filters: tuple[ResidualFilterConfig, ...],
) -> tuple[ResidualFilterConfig, ...]:
    if config.geometry_update_volume_source == "fixed_synthetic_truth":
        return (ResidualFilterConfig(kind="raw"),)
    return residual_filters


def _geometry_parameter_prior_strength(
    config: AlternatingSmokeConfig,
    level: ContinuationLevel,
) -> float:
    if config.geometry_update_volume_source == "fixed_synthetic_truth":
        return 0.0
    return float(level.prior_strength)
