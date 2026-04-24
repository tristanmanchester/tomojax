from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ._json import JsonValue, drop_none


ThetaSign = Literal[-1, 1]


@dataclass(frozen=True)
class ConventionEvidence:
    """One piece of evidence used to judge a detector or angle convention."""

    source: str
    description: str
    score: float | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        return drop_none(
            {
                "source": self.source,
                "description": self.description,
                "score": self.score,
            }
        )


@dataclass(frozen=True)
class ConventionAudit:
    """Serializable record of detector and angle convention assumptions."""

    flip_u: bool | None = None
    flip_v: bool | None = None
    transpose_detector: bool | None = None
    theta_sign: ThetaSign | None = None
    theta_zero_deg: float | None = None
    confidence: float | None = None
    correction_applied: bool = False
    evidence: tuple[ConventionEvidence, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.theta_sign is not None and self.theta_sign not in {-1, 1}:
            raise ValueError("theta_sign must be -1, 1, or None")

    @property
    def is_ambiguous(self) -> bool:
        return self.confidence is None or float(self.confidence) < 0.5

    def to_dict(self) -> dict[str, JsonValue]:
        return drop_none(
            {
                "flip_u": self.flip_u,
                "flip_v": self.flip_v,
                "transpose_detector": self.transpose_detector,
                "theta_sign": self.theta_sign,
                "theta_zero_deg": self.theta_zero_deg,
                "confidence": self.confidence,
                "correction_applied": self.correction_applied,
                "ambiguous": self.is_ambiguous,
                "evidence": [item.to_dict() for item in self.evidence],
            }
        )
