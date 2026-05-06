"""Continuation schedules for v2 alternating alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tomojax.forward import ResidualFilterConfig

ContinuationScheduleName = Literal["smoke32", "lightning", "balanced", "reference"]
ContinuationLevelRole = Literal["preview", "final"]


@dataclass(frozen=True)
class ContinuationLevel:
    """One reconstruction/geometry continuation level."""

    level_factor: int
    reconstruction_iterations: int
    geometry_updates: int
    residual_sigma: float
    residual_delta: float
    trust_radius_px: float
    prior_strength: float
    residual_filters: tuple[ResidualFilterConfig, ...]
    role: ContinuationLevelRole = "preview"
    skip_finer_if_verified: bool = True
    run_if_coarse_unverified: bool = False


@dataclass(frozen=True)
class ContinuationSchedule:
    """Ordered continuation policy for alternating alignment."""

    name: ContinuationScheduleName
    levels: tuple[ContinuationLevel, ...]

    @property
    def level_factors(self) -> tuple[int, ...]:
        """Return the ordered level factors."""
        return tuple(level.level_factor for level in self.levels)


def reference_continuation_schedule(
    name: ContinuationScheduleName = "smoke32",
) -> ContinuationSchedule:
    """Return the deterministic reference continuation schedule."""
    if name == "smoke32":
        return _make_schedule(
            name=name,
            preview_iterations=1,
            preview_updates=8,
            final_iterations=1,
            final_updates=1,
        )
    if name == "lightning":
        return _make_schedule(
            name=name,
            preview_iterations=2,
            preview_updates=1,
            final_iterations=2,
            final_updates=1,
        )
    if name == "balanced":
        return _make_schedule(
            name=name,
            preview_iterations=4,
            preview_updates=2,
            final_iterations=4,
            final_updates=1,
        )
    if name == "reference":
        return _make_schedule(
            name=name,
            preview_iterations=8,
            preview_updates=4,
            final_iterations=8,
            final_updates=2,
        )
    raise ValueError(f"unknown continuation schedule {name!r}")


def _make_schedule(
    *,
    name: ContinuationScheduleName,
    preview_iterations: int,
    preview_updates: int,
    final_iterations: int,
    final_updates: int,
) -> ContinuationSchedule:
    return ContinuationSchedule(
        name=name,
        levels=(
            ContinuationLevel(
                level_factor=4,
                reconstruction_iterations=preview_iterations,
                geometry_updates=preview_updates,
                residual_sigma=1.0,
                residual_delta=1.0,
                trust_radius_px=2.0,
                prior_strength=1.0e-3,
                residual_filters=(
                    ResidualFilterConfig(kind="lowpass_gaussian", weight=1.0, sigma_px=1.0),
                ),
                role="preview",
                skip_finer_if_verified=True,
            ),
            ContinuationLevel(
                level_factor=2,
                reconstruction_iterations=preview_iterations,
                geometry_updates=preview_updates,
                residual_sigma=0.85,
                residual_delta=0.85,
                trust_radius_px=1.0,
                prior_strength=5.0e-4,
                residual_filters=(
                    ResidualFilterConfig(kind="lowpass_gaussian", weight=0.7, sigma_px=1.0),
                    ResidualFilterConfig(
                        kind="bandpass_difference_of_gaussians",
                        weight=0.3,
                        sigma_px=0.8,
                        outer_sigma_px=1.6,
                    ),
                ),
                role="preview",
                skip_finer_if_verified=True,
                run_if_coarse_unverified=True,
            ),
            ContinuationLevel(
                level_factor=1,
                reconstruction_iterations=final_iterations,
                geometry_updates=final_updates,
                residual_sigma=0.75,
                residual_delta=0.75,
                trust_radius_px=0.5,
                prior_strength=1.0e-4,
                residual_filters=(ResidualFilterConfig(kind="raw"),),
                role="final",
                skip_finer_if_verified=False,
            ),
        ),
    )


__all__ = [
    "ContinuationLevel",
    "ContinuationSchedule",
    "ContinuationScheduleName",
    "reference_continuation_schedule",
]
