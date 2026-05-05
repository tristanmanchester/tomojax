from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, TypeAlias, cast


AlignmentQualityTier: TypeAlias = Literal[
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
    recon_iters_multiplier: float
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
    tier = str(value).strip().lower().replace("-", "_")
    if tier in _POLICIES:
        return cast(AlignmentQualityTier, tier)
    if tier == "tortoise":
        return "reference"
    if tier == "lightning":
        return "fast"
    raise ValueError(
        "quality tier must be one of 'proposal', 'fast', 'refine', "
        "'verify', 'final', or 'reference'"
    )


def reconstruction_quality_policy(value: str) -> ReconstructionQualityPolicy:
    return _POLICIES[normalize_quality_tier(value)]


__all__ = [
    "AlignmentQualityTier",
    "ReconstructionQualityPolicy",
    "normalize_quality_tier",
    "reconstruction_quality_policy",
]
