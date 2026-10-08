from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, cast

type AlignmentQualityTier = Literal[
    "proposal",
    "fast",
    "refine",
    "verify",
    "final",
    "reference",
]


@dataclass(frozen=True, slots=True)
class ReconstructionQualityPolicy:
    """Stage-level reconstruction quality and diagnostic policy."""

    tier: AlignmentQualityTier
    iterations_multiplier: float
    compute_iteration_loss: bool
    compute_final_data_loss: bool
    compute_final_regulariser_value: bool
    prefer_mixed_precision: bool
    final_quality: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_POLICIES: dict[AlignmentQualityTier, ReconstructionQualityPolicy] = {
    "proposal": ReconstructionQualityPolicy("proposal", 0.25, False, False, False, True),
    "fast": ReconstructionQualityPolicy("fast", 1.0, False, False, False, True),
    "refine": ReconstructionQualityPolicy("refine", 1.5, False, True, False, True),
    "verify": ReconstructionQualityPolicy("verify", 2.0, True, True, True, False),
    "final": ReconstructionQualityPolicy("final", 4.0, True, True, True, False, True),
    "reference": ReconstructionQualityPolicy("reference", 2.0, True, True, True, False),
}


def normalize_quality_tier(value: str) -> AlignmentQualityTier:
    if value in _POLICIES:
        return cast("AlignmentQualityTier", value)
    raise ValueError(
        "quality tier must be one of 'proposal', 'fast', 'refine', "
        "'verify', 'final', or 'reference'"
    )


def reconstruction_quality_policy(value: str) -> ReconstructionQualityPolicy:
    return _POLICIES[normalize_quality_tier(value)]


def scaled_reconstruction_iterations(
    iterations: int | float,
    policy: ReconstructionQualityPolicy,
) -> int:
    return max(1, int(float(iterations) * float(policy.iterations_multiplier)))


__all__ = [
    "AlignmentQualityTier",
    "ReconstructionQualityPolicy",
    "normalize_quality_tier",
    "reconstruction_quality_policy",
    "scaled_reconstruction_iterations",
]
