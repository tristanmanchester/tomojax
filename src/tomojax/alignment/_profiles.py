from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomojax.core.backend_policy import ProjectorBackend
    from tomojax.recon.types import Regulariser


type QualityTier = Literal["fast", "reference"]


@dataclass(frozen=True, slots=True)
class AlignmentProfilePolicy:
    """The solver defaults an alignment ``quality`` stands for."""

    quality: QualityTier
    projector_backend: ProjectorBackend
    gather_dtype: str
    regulariser: Regulariser
    reconstruction: str
    views_per_batch: int
    checkpoint_projector: bool
    pose_model: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_quality(value: object) -> QualityTier:
    """Return ``value`` if it is an alignment quality, else raise ValueError."""
    if value not in ("fast", "reference"):
        raise ValueError(f"quality must be 'fast' or 'reference', not {value!r}")
    return cast("QualityTier", value)


def alignment_profile_policy(quality: QualityTier) -> AlignmentProfilePolicy:
    """Return the solver defaults for ``quality``."""
    if normalize_quality(quality) == "reference":
        return AlignmentProfilePolicy(
            quality="reference",
            projector_backend="jax",
            gather_dtype="fp32",
            regulariser="tv",
            reconstruction="fista",
            views_per_batch=0,
            checkpoint_projector=True,
            pose_model="per_view",
        )
    return AlignmentProfilePolicy(
        quality="fast",
        projector_backend="pallas",
        gather_dtype="auto",
        regulariser="huber_tv",
        reconstruction="fista",
        views_per_batch=0,
        checkpoint_projector=True,
        pose_model="per_view",
    )


def profile_policy_from_config(cfg: object) -> AlignmentProfilePolicy:
    """Build a policy snapshot from an already-normalized alignment config."""
    return AlignmentProfilePolicy(
        quality=normalize_quality(getattr(cfg, "quality", "fast")),
        projector_backend=cast("ProjectorBackend", getattr(cfg, "projector_backend", "jax")),
        gather_dtype=str(getattr(cfg, "gather_dtype", "fp32")),
        regulariser=cast("Regulariser", getattr(cfg, "regulariser", "tv")),
        reconstruction=str(getattr(cfg, "reconstruction", "fista")),
        views_per_batch=int(getattr(cfg, "views_per_batch", 0)),
        checkpoint_projector=bool(getattr(cfg, "checkpoint_projector", True)),
        pose_model=str(getattr(cfg, "pose_model", "per_view")),
    )


def resolve_profiled_cli_defaults(
    *,
    quality: QualityTier,
    current: Mapping[str, object],
    configured_keys: set[str],
) -> dict[str, object]:
    """Apply profile defaults only for options the user/config did not specify."""
    policy = alignment_profile_policy(quality)
    defaults = policy.to_dict()
    resolved = dict(current)
    for key in (
        "projector_backend",
        "gather_dtype",
        "regulariser",
        "reconstruction",
        "views_per_batch",
        "checkpoint_projector",
        "pose_model",
    ):
        if key not in configured_keys:
            resolved[key] = defaults[key]
    resolved["quality"] = policy.quality
    resolved["profile_defaults"] = defaults
    resolved["profile_configured_keys"] = sorted(configured_keys)
    return resolved


__all__ = [
    "AlignmentProfilePolicy",
    "QualityTier",
    "alignment_profile_policy",
    "normalize_quality",
    "profile_policy_from_config",
    "resolve_profiled_cli_defaults",
]
