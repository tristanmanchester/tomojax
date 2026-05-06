"""Per-level smoke-run helpers for alternating orchestration."""
# pyright: reportUnusedFunction=false

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from tomojax.align._alternating_types import AlternatingLevelSummary

if TYPE_CHECKING:
    from tomojax.align._continuation import ContinuationLevel


def _coarse_verification_state(
    *,
    level: ContinuationLevel,
    checks_verified: bool,
    already_verified: bool,
    previous_time_to_verified: float | None,
    run_start: float,
) -> tuple[bool, float | None]:
    level_verified = level.role == "preview" and level.skip_finer_if_verified and checks_verified
    if level_verified and not already_verified:
        return True, float(time.perf_counter() - run_start)
    return already_verified or level_verified, previous_time_to_verified


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
