from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomojax.core.backend_policy import ProjectorBackend
    from tomojax.recon.types import Regulariser


type AlignmentProfile = Literal["lightning", "tortoise"]
type AlignmentProfileInput = AlignmentProfile | str
type FallbackPolicy = Literal["fallback", "strict"]
type QualityTier = Literal["fast", "reference"]


@dataclass(frozen=True, slots=True)
class AlignmentProfilePolicy:
    """Effective defaults for a high-level alignment posture."""

    align_profile: AlignmentProfile
    projector_backend: ProjectorBackend
    gather_dtype: str
    regulariser: Regulariser
    recon_algo: str
    views_per_batch: int
    checkpoint_projector: bool
    pose_model: str
    quality_tier: QualityTier
    fallback_policy: FallbackPolicy

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_alignment_profile(value: AlignmentProfileInput) -> AlignmentProfile:
    """Normalize a public alignment profile value."""
    profile = str(value).strip().lower().replace("-", "_")
    if profile in {"", "default", "fast", "lightning"}:
        return "lightning"
    if profile in {"slow", "safe", "reference", "tortoise"}:
        return "tortoise"
    raise ValueError("align_profile must be one of 'lightning' or 'tortoise'")


def alignment_profile_policy(value: AlignmentProfileInput) -> AlignmentProfilePolicy:
    """Return the baseline policy for a normalized alignment profile."""
    profile = normalize_alignment_profile(value)
    if profile == "tortoise":
        return AlignmentProfilePolicy(
            align_profile="tortoise",
            projector_backend="jax",
            gather_dtype="fp32",
            regulariser="tv",
            recon_algo="fista",
            views_per_batch=1,
            checkpoint_projector=True,
            pose_model="per_view",
            quality_tier="reference",
            fallback_policy="fallback",
        )
    return AlignmentProfilePolicy(
        align_profile="lightning",
        projector_backend="pallas",
        gather_dtype="auto",
        regulariser="huber_tv",
        recon_algo="fista",
        views_per_batch=0,
        checkpoint_projector=True,
        pose_model="spline",
        quality_tier="fast",
        fallback_policy="fallback",
    )


def profile_policy_from_config(cfg: object) -> AlignmentProfilePolicy:
    """Build a policy snapshot from an already-normalized alignment config."""
    return AlignmentProfilePolicy(
        align_profile=normalize_alignment_profile(getattr(cfg, "align_profile", "lightning")),
        projector_backend=cast("ProjectorBackend", getattr(cfg, "projector_backend", "jax")),
        gather_dtype=str(getattr(cfg, "gather_dtype", "fp32")),
        regulariser=cast("Regulariser", getattr(cfg, "regulariser", "tv")),
        recon_algo=str(getattr(cfg, "recon_algo", "fista")),
        views_per_batch=int(getattr(cfg, "views_per_batch", 1)),
        checkpoint_projector=bool(getattr(cfg, "checkpoint_projector", True)),
        pose_model=str(getattr(cfg, "pose_model", "per_view")),
        quality_tier=cast("QualityTier", getattr(cfg, "quality_tier", "fast")),
        fallback_policy=cast("FallbackPolicy", getattr(cfg, "fallback_policy", "fallback")),
    )


def resolve_profiled_cli_defaults(
    *,
    align_profile: AlignmentProfileInput,
    current: Mapping[str, object],
    configured_keys: set[str],
) -> dict[str, object]:
    """Apply profile defaults only for options the user/config did not specify."""
    policy = alignment_profile_policy(align_profile)
    defaults = policy.to_dict()
    resolved = dict(current)
    for key in (
        "projector_backend",
        "gather_dtype",
        "regulariser",
        "recon_algo",
        "views_per_batch",
        "checkpoint_projector",
        "pose_model",
    ):
        if key not in configured_keys:
            resolved[key] = defaults[key]
    resolved["align_profile"] = policy.align_profile
    resolved["quality_tier"] = policy.quality_tier
    resolved["fallback_policy"] = policy.fallback_policy
    resolved["profile_defaults"] = defaults
    resolved["profile_configured_keys"] = sorted(configured_keys)
    return resolved


__all__ = [
    "AlignmentProfile",
    "AlignmentProfileInput",
    "AlignmentProfilePolicy",
    "FallbackPolicy",
    "QualityTier",
    "alignment_profile_policy",
    "normalize_alignment_profile",
    "profile_policy_from_config",
    "resolve_profiled_cli_defaults",
]
