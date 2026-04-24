from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ._json import JsonValue, drop_none, normalize_json


@dataclass(frozen=True)
class MetricSpec:
    """A named metric and whether it is minimized or maximized."""

    name: str
    direction: str = "minimize"
    domain: str = "projection"
    description: str | None = None

    def __post_init__(self) -> None:
        if self.direction not in {"minimize", "maximize"}:
            raise ValueError("metric direction must be 'minimize' or 'maximize'")

    def to_dict(self) -> dict[str, JsonValue]:
        return drop_none(
            {
                "name": self.name,
                "direction": self.direction,
                "domain": self.domain,
                "description": self.description,
            }
        )


@dataclass(frozen=True)
class CandidateScore:
    """Score and diagnostics for one future calibration candidate."""

    parameters: Mapping[str, object]
    score: float
    rank: int | None = None
    uncertainty: Mapping[str, object] | None = None
    artifacts: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        return drop_none(
            {
                "parameters": normalize_json(self.parameters),
                "score": self.score,
                "rank": self.rank,
                "uncertainty": normalize_json(self.uncertainty),
                "artifacts": normalize_json(self.artifacts),
            }
        )


@dataclass(frozen=True)
class ObjectiveCard:
    """Serializable objective contract for future geometry-calibration runs."""

    primary_metric: MetricSpec
    secondary_metrics: tuple[MetricSpec, ...] = field(default_factory=tuple)
    validation_split: Mapping[str, object] | None = None
    top_candidates: tuple[CandidateScore, ...] = field(default_factory=tuple)
    curvature: Mapping[str, object] | None = None
    contact_sheet: str | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        return drop_none(
            {
                "primary_metric": self.primary_metric.to_dict(),
                "secondary_metrics": [metric.to_dict() for metric in self.secondary_metrics],
                "validation_split": normalize_json(self.validation_split),
                "top_candidates": [candidate.to_dict() for candidate in self.top_candidates],
                "curvature": normalize_json(self.curvature),
                "contact_sheet": self.contact_sheet,
            }
        )
