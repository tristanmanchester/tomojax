"""Orchestration for the v2 alternating alignment smoke runner."""
# pyright: reportAny=false, reportPrivateUsage=false, reportUnknownMemberType=false
# pyright: reportUnusedFunction=false

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import TYPE_CHECKING

import jax

from tomojax.align._alternating_artifacts import _write_artifacts
from tomojax.align._alternating_geometry_update import (
    _geometry_update_volume,
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
)
from tomojax.align._alternating_verification import (
    _level_verification_checks,
    _verification_payload,
)
from tomojax.align._continuation import reference_continuation_schedule
from tomojax.geometry import GaugeReport
from tomojax.recon import (
    ReferenceFISTAConfig,
    ReferenceFISTAResult,
    fista_reconstruct_reference,
    reconstruct_backprojection_reference,
)

if TYPE_CHECKING:
    from tomojax.align._joint_schur_lm import JointSchurLMResult


def _run_alternating_solver_smoke_impl(
    output_dir: str | Path,
    *,
    config: AlternatingSmokeConfig,
) -> AlternatingSmokeResult:
    run_start = time.perf_counter()
    schedule = config.schedule or reference_continuation_schedule()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = build_smoke_inputs(config)
    truth, observed, mask = inputs.truth_volume, inputs.observed_projections, inputs.mask
    true_geometry, initial_geometry = inputs.true_geometry, inputs.initial_geometry
    train_mask, heldout_mask = _heldout_masks(mask, config.heldout_view_index)

    geometry = initial_geometry
    gauge_report = GaugeReport(())
    volume: jax.Array | None = None
    summaries: list[AlternatingLevelSummary] = []
    fista_result: ReferenceFISTAResult | None = None
    coarse_verified = False
    time_to_verified_geometry_seconds: float | None = None
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
                fit_background_nuisance=config.fit_background_nuisance,
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
            geometry_update_accepted=(
                None if schur_result is None else schur_result.diagnostics.accepted
            ),
        )
        previous_loss = loss_after
        coarse_verified, time_to_verified_geometry_seconds = _coarse_verification_state(
            level=level,
            checks_verified=verification_checks.verified,
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
