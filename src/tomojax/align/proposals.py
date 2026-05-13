from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..core.backend_policy import ProjectorBackendInput
from ..core.geometry import Detector, Grid
from .objectives.fixed_volume import (
    alignment_projector_backend_provenance,
    project_and_score_stack,
)
from .objectives.loss_adapters import LossAdapter


@dataclass(frozen=True, slots=True)
class ProposalCandidate:
    """A candidate pose-stack update scored by a proposal stage."""

    name: str
    pose_stack: jnp.ndarray
    metadata: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ProposalScoringResult:
    """Result from performance-oriented proposal scoring."""

    best_index: int
    best_name: str
    best_value: float
    values: tuple[float, ...]
    backend_provenance: dict[str, object]
    candidate_metadata: tuple[dict[str, object], ...]

    @property
    def improved(self) -> bool:
        return self.best_index > 0 and self.best_value < self.values[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "best_index": int(self.best_index),
            "best_name": self.best_name,
            "best_value": float(self.best_value),
            "values": [float(value) for value in self.values],
            "improved": bool(self.improved),
            "backend_provenance": dict(self.backend_provenance),
            "candidate_metadata": [dict(item) for item in self.candidate_metadata],
        }


def score_pose_stack_candidates(
    *,
    candidates: Sequence[ProposalCandidate],
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    targets: jnp.ndarray,
    loss_adapter: LossAdapter,
    projector_backend: ProjectorBackendInput = "pallas",
    gather_dtype: str = "auto",
    views_per_batch: int = 0,
    projector_unroll: int = 1,
    checkpoint_projector: bool = True,
) -> ProposalScoringResult:
    """Score candidate pose stacks without requiring differentiability.

    The first candidate is treated as the baseline/current state. This helper is
    intentionally performance-oriented and records provenance so downstream
    verification can decide whether the candidate is acceptable.
    """
    if not candidates:
        raise ValueError("proposal scoring requires at least one candidate")
    values: list[float] = []
    metadata: list[dict[str, object]] = []
    for candidate in candidates:
        score = project_and_score_stack(
            pose_stack=candidate.pose_stack,
            grid=grid,
            detector=detector,
            volume=volume,
            det_grid=det_grid,
            targets=targets,
            loss_adapter=loss_adapter,
            views_per_batch=views_per_batch,
            projector_unroll=projector_unroll,
            checkpoint_projector=checkpoint_projector,
            gather_dtype=gather_dtype,
            projector_backend=projector_backend,
            require_differentiable_projector=False,
        )
        jax.block_until_ready(score)
        values.append(float(score))
        metadata.append(dict(candidate.metadata or {}))

    best_index = min(range(len(values)), key=values.__getitem__)
    best_candidate = candidates[best_index]
    provenance = alignment_projector_backend_provenance(
        pose_stack=best_candidate.pose_stack,
        grid=grid,
        detector=detector,
        volume=volume,
        det_grid=det_grid,
        projector_backend=projector_backend,
        require_differentiable_projector=False,
        gather_dtype=gather_dtype,
        api_surface="alignment.proposal_scoring",
    ).to_dict()
    return ProposalScoringResult(
        best_index=int(best_index),
        best_name=best_candidate.name,
        best_value=float(values[best_index]),
        values=tuple(float(value) for value in values),
        backend_provenance=provenance,
        candidate_metadata=tuple(metadata),
    )


__all__ = [
    "ProposalCandidate",
    "ProposalScoringResult",
    "score_pose_stack_candidates",
]
