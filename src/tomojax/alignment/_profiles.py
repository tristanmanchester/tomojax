from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from tomojax.core.backend_policy import ProjectorBackend
    from tomojax.recon.types import Regulariser


type QualityTier = Literal["fast", "reference"]
type StageQualityTier = Literal["fast", "reference", "verify"]


@dataclass(frozen=True, slots=True)
class AlignmentProfilePolicy:
    """The solver defaults an alignment ``quality`` stands for.

    Parameters
    ----------
    quality : QualityTier
        Alignment quality tier, "fast" or "reference".
    backend : ProjectorBackend
        Projector implementation, "jax" or "cuda".
    gather_dtype : str
        Precision used for projection gathers.
    regulariser : Regulariser
        Volume regularizer, "tv" or "huber_tv".
    reconstruction : str
        Inner reconstruction method, "fista" or "spdhg".
    views_per_batch : int
        View batch size; zero means automatic memory-based sizing.
    checkpoint_projector : bool
        Whether projector intermediates are recomputed during differentiation.
    pose_model : str
        Motion representation, "per_view", "polynomial" or "spline".
    """

    quality: QualityTier
    backend: ProjectorBackend
    gather_dtype: str
    regulariser: Regulariser
    reconstruction: str
    views_per_batch: int
    checkpoint_projector: bool
    pose_model: str

    def to_dict(self) -> dict[str, object]:
        """Return the solver policy as a dictionary.

        Returns
        -------
        dict
            Solver-policy fields in JSON-compatible form.
        """
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
            backend="jax",
            gather_dtype="fp32",
            regulariser="tv",
            reconstruction="fista",
            views_per_batch=0,
            checkpoint_projector=True,
            pose_model="per_view",
        )
    return AlignmentProfilePolicy(
        quality="fast",
        backend="cuda",
        gather_dtype="auto",
        regulariser="huber_tv",
        reconstruction="fista",
        views_per_batch=0,
        checkpoint_projector=True,
        pose_model="per_view",
    )


def profile_policy_from_config(cfg: object) -> AlignmentProfilePolicy:
    """Build a policy snapshot from an already-normalized alignment config.

    Parameters
    ----------
    cfg : AlignConfig
        Normalized configuration whose effective policy is recorded.

    Returns
    -------
    AlignmentProfilePolicy
        Snapshot of the configuration effective solver policy.

    Raises
    ------
    ValueError
        If the configuration names an unsupported quality policy.
    """
    return AlignmentProfilePolicy(
        quality=normalize_quality(getattr(cfg, "quality", "fast")),
        backend=cast("ProjectorBackend", getattr(cfg, "backend", "jax")),
        gather_dtype=str(getattr(cfg, "gather_dtype", "fp32")),
        regulariser=cast("Regulariser", getattr(cfg, "regulariser", "tv")),
        reconstruction=str(getattr(cfg, "reconstruction", "fista")),
        views_per_batch=int(getattr(cfg, "views_per_batch", 0)),
        checkpoint_projector=bool(getattr(cfg, "checkpoint_projector", True)),
        pose_model=str(getattr(cfg, "pose_model", "per_view")),
    )


@dataclass(frozen=True, slots=True)
class ReconstructionQualityPolicy:
    """Stage-level reconstruction quality and diagnostic policy."""

    tier: StageQualityTier
    iterations_multiplier: float
    compute_iteration_loss: bool
    compute_final_data_loss: bool
    compute_final_regulariser_value: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_POLICIES: dict[StageQualityTier, ReconstructionQualityPolicy] = {
    "fast": ReconstructionQualityPolicy("fast", 1.0, False, False, False),
    "verify": ReconstructionQualityPolicy("verify", 2.0, True, True, True),
    "reference": ReconstructionQualityPolicy("reference", 2.0, True, True, True),
}


def normalize_quality_tier(value: str) -> StageQualityTier:
    if value in _POLICIES:
        return cast("StageQualityTier", value)
    raise ValueError("quality tier must be 'fast', 'verify', or 'reference'")


def reconstruction_quality_policy(value: str) -> ReconstructionQualityPolicy:
    return _POLICIES[normalize_quality_tier(value)]


def scaled_reconstruction_iterations(
    iterations: int | float,
    policy: ReconstructionQualityPolicy,
) -> int:
    return max(1, int(float(iterations) * float(policy.iterations_multiplier)))


__all__ = [
    "AlignmentProfilePolicy",
    "QualityTier",
    "alignment_profile_policy",
    "normalize_quality",
    "profile_policy_from_config",
]
