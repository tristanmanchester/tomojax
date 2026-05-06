"""Reference reconstruction schedules for v2 FISTA previews."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tomojax.forward import ResidualFilterConfig

from ._fista_reference import ReferenceFISTAConfig

ReferenceReconstructionScheduleName = Literal["preview", "final"]


@dataclass(frozen=True)
class ReferenceFISTAScheduleEntry:
    level_factor: int
    role: Literal["preview", "final"]
    fista: ReferenceFISTAConfig
    residual_filters: tuple[ResidualFilterConfig, ...]


@dataclass(frozen=True)
class ReferenceFISTASchedule:
    name: ReferenceReconstructionScheduleName
    entries: tuple[ReferenceFISTAScheduleEntry, ...]

    @property
    def level_factors(self) -> tuple[int, ...]:
        return tuple(entry.level_factor for entry in self.entries)


def reference_fista_schedule(
    name: ReferenceReconstructionScheduleName = "preview",
) -> ReferenceFISTASchedule:
    """Resolve a v2 reference reconstruction schedule by name."""
    if name == "preview":
        return ReferenceFISTASchedule(
            name=name,
            entries=(
                ReferenceFISTAScheduleEntry(
                    level_factor=4,
                    role="preview",
                    fista=ReferenceFISTAConfig(iterations=4, step_size=5e-3, tv_weight=1e-3),
                    residual_filters=(
                        ResidualFilterConfig(kind="lowpass_gaussian", weight=1.0, sigma_px=1.0),
                    ),
                ),
                ReferenceFISTAScheduleEntry(
                    level_factor=2,
                    role="preview",
                    fista=ReferenceFISTAConfig(iterations=6, step_size=5e-3, tv_weight=1e-3),
                    residual_filters=(
                        ResidualFilterConfig(kind="lowpass_gaussian", weight=0.7, sigma_px=1.0),
                        ResidualFilterConfig(
                            kind="bandpass_difference_of_gaussians",
                            weight=0.3,
                            sigma_px=0.8,
                            outer_sigma_px=1.6,
                        ),
                    ),
                ),
            ),
        )
    if name == "final":
        return ReferenceFISTASchedule(
            name=name,
            entries=(
                ReferenceFISTAScheduleEntry(
                    level_factor=1,
                    role="final",
                    fista=ReferenceFISTAConfig(iterations=12, step_size=5e-3, tv_weight=5e-4),
                    residual_filters=(ResidualFilterConfig(kind="raw"),),
                ),
            ),
        )
    raise ValueError(f"unknown reference FISTA schedule: {name}")
