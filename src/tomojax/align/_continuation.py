"""Continuation schedules for v2 alternating alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ContinuationScheduleName = Literal["smoke32"]
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
    role: ContinuationLevelRole = "preview"
    skip_finer_if_verified: bool = True


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
    if name != "smoke32":
        raise ValueError(f"unknown continuation schedule {name!r}")
    return ContinuationSchedule(
        name=name,
        levels=(
            ContinuationLevel(
                level_factor=4,
                reconstruction_iterations=1,
                geometry_updates=1,
                residual_sigma=1.0,
                residual_delta=1.0,
                trust_radius_px=2.0,
                prior_strength=1.0e-3,
                role="preview",
                skip_finer_if_verified=True,
            ),
            ContinuationLevel(
                level_factor=1,
                reconstruction_iterations=1,
                geometry_updates=0,
                residual_sigma=0.75,
                residual_delta=0.75,
                trust_radius_px=0.5,
                prior_strength=1.0e-4,
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
