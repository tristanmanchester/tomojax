"""Small v2 alternating alignment smoke runner."""
# pyright: reportAny=false, reportPrivateUsage=false, reportUnknownMemberType=false

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import jax

from tomojax.align._alternating_artifacts import _write_artifacts
from tomojax.align._alternating_heldout import (
    _heldout_masks,
    _heldout_residual_check,
    _level_residual_sigma,
    _projection_loss,
)
from tomojax.align._alternating_inputs import build_smoke_inputs
from tomojax.align._alternating_types import (
    AlternatingLevelSummary,
    AlternatingSmokeConfig,
    AlternatingSmokeResult,
    GeometryUpdateVolumeSource,
)
from tomojax.align._alternating_verification import (
    _level_verification_checks,
    _verification_payload,
)
from tomojax.align._continuation import ContinuationLevel, reference_continuation_schedule
from tomojax.align._joint_schur_lm import (
    JointSchurLMConfig,
    JointSchurLMResult,
    solve_joint_schur_lm,
)
from tomojax.geometry import GaugeReport, GeometryState
from tomojax.recon import (
    ReferenceFISTAConfig,
    ReferenceFISTAResult,
    fista_reconstruct_reference,
    reconstruct_backprojection_reference,
)


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
    fit_background_nuisance: bool,
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
            fit_background_offset=fit_background_nuisance,
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


__all__ = [
    "AlternatingAlignmentSolver",
    "AlternatingLevelSummary",
    "AlternatingSmokeConfig",
    "AlternatingSmokeResult",
    "GeometryUpdateVolumeSource",
    "run_alternating_solver_smoke",
]
