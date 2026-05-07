"""Orchestration for the v2 alternating alignment smoke runner."""
# pyright: reportAny=false, reportPrivateUsage=false, reportUnknownMemberType=false
# pyright: reportUnusedFunction=false

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

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


def _run_alternating_solver_smoke_impl(
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
    fista_result: ReferenceFISTAResult | None
    last_schur_result: JointSchurLMResult | None
    constrained_first_preview_volume: jax.Array | None = None
    fista_result, coarse_verified = None, False
    time_to_verified_geometry_seconds, last_schur_result = None, None

    initial_loss = previous_loss = _projection_loss(
        truth,
        observed,
        geometry,
        mask,
        schedule.levels[0],
        sigma=schedule.levels[0].residual_sigma,
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
            mask=mask,
            config=ReferenceFISTAConfig(
                iterations=level.reconstruction_iterations,
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
                active_setup_parameters=_active_setup_parameters_for_level(
                    config,
                    level.level_factor,
                ),
                solver=config.geometry_update_solver,
                pose_frozen=_pose_frozen_for_level(config, level.level_factor),
                active_pose_dofs=config.geometry_update_active_pose_dofs,
                fit_gain_offset_nuisance=config.fit_gain_offset_nuisance,
                fit_background_nuisance=config.fit_background_nuisance,
                residual_filters=_geometry_residual_filters(config, level.residual_filters),
                parameter_prior_strength=_geometry_parameter_prior_strength(config, level),
            )
            geometry, update_report, schur_result = _apply_heldout_geometry_acceptance(
                config,
                geometry_update_volume,
                observed,
                before_geometry=geometry_before_update,
                candidate_geometry=geometry,
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
                verification_checks.verified
                and config.geometry_update_volume_source != "fixed_synthetic_truth"
            ),
            already_verified=coarse_verified,
            previous_time_to_verified=time_to_verified_geometry_seconds,
            run_start=run_start,
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
        geometry_update_solver=config.geometry_update_solver,
        geometry_update_setup_prior_strength=config.geometry_update_setup_prior_strength,
        geometry_update_pose_prior_strength=config.geometry_update_pose_prior_strength,
        geometry_update_pose_frozen=config.geometry_update_pose_frozen,
        geometry_update_pose_activate_at_level_factor=(
            config.geometry_update_pose_activate_at_level_factor
        ),
        geometry_update_theta_activate_at_level_factor=(
            config.geometry_update_theta_activate_at_level_factor
        ),
        geometry_update_active_setup_parameters=config.geometry_update_active_setup_parameters,
        geometry_update_active_pose_dofs=config.geometry_update_active_pose_dofs,
        preview_volume_support=config.preview_volume_support,
        preview_initialization=config.preview_initialization,
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
            final_loss=final_loss,
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
        config.stopped_preview_policy == "constant_cylindrical_first_level"
        and config.geometry_update_volume_source == "stopped_reconstruction"
        and level.role == "preview"
        and int(level.level_factor) == 4
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
        config.stopped_preview_policy == "constant_cylindrical_first_level"
        and config.geometry_update_volume_source == "stopped_reconstruction"
        and constrained_first_preview_volume is not None
        and not _uses_constant_cylindrical_first_level(config, level)
    ):
        return constrained_first_preview_volume
    return current_volume


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
