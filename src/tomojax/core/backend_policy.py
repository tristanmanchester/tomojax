"""Backend selection and provenance helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

type ProjectorBackend = Literal["jax", "pallas"]
type ProjectorBackendInput = ProjectorBackend | str


@dataclass(frozen=True, slots=True)
class BackendProvenance:
    """Requested/actual backend metadata for supported accelerator boundaries."""

    requested_backend: ProjectorBackend
    actual_backend: ProjectorBackend
    status: Literal["supported", "fallback", "unsupported"]
    fallback_reason: str | None = None
    api_surface: str = "unknown"
    differentiability: Literal["gradient_safe", "performance_only"] = "gradient_safe"

    @property
    def eligible_for_speed_claim(self) -> bool:
        """Return whether timings can be described as using the requested backend."""
        return self.status == "supported" and self.requested_backend == self.actual_backend

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible backend provenance."""
        data = asdict(self)
        data["eligible_for_speed_claim"] = self.eligible_for_speed_claim
        return data


def normalize_projector_backend(value: ProjectorBackendInput) -> ProjectorBackend:
    """Normalize a public projector backend value."""
    backend = str(value).strip().lower().replace("-", "_")
    if backend in {"jax", "default"}:
        return "jax"
    if backend == "pallas":
        return "pallas"
    raise ValueError("projector_backend must be one of 'jax' or 'pallas'")


def backend_provenance(
    *,
    requested_backend: ProjectorBackendInput,
    actual_backend: ProjectorBackendInput,
    api_surface: str,
    fallback_reason: str | None = None,
    differentiability: Literal["gradient_safe", "performance_only"] = "gradient_safe",
) -> BackendProvenance:
    """Build normalized backend provenance metadata."""
    requested = normalize_projector_backend(requested_backend)
    actual = normalize_projector_backend(actual_backend)
    status: Literal["supported", "fallback", "unsupported"]
    if requested == actual and fallback_reason is None:
        status = "supported"
    elif actual == "jax":
        status = "fallback"
    else:
        status = "unsupported"
    return BackendProvenance(
        requested_backend=requested,
        actual_backend=actual,
        status=status,
        fallback_reason=fallback_reason,
        api_surface=str(api_surface),
        differentiability=differentiability,
    )
