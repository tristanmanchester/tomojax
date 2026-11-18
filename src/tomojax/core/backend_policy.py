"""Backend selection and provenance helpers."""

from __future__ import annotations

from typing import Literal

from .validation import option_name

type ProjectorBackend = Literal["jax", "cuda"]
type ProjectorBackendInput = ProjectorBackend | str


def cuda_backend(platform: str | None = None) -> bool:
    """Whether the selected JAX device belongs to an NVIDIA CUDA backend.

    Without an explicit platform, respect ``jax.default_device`` before the
    process-wide preference. Pass ``jax.default_backend()`` for capability
    checks that must not depend on a temporary device-selection context.
    """
    import jax
    from jax.extend.backend import get_backend

    if platform is None:
        device = jax.config.jax_default_device
        if device is None:
            platform = jax.default_backend()
        else:
            platform = device if isinstance(device, str) else device.platform
    return platform == "gpu" and "cuda" in get_backend(platform).platform_version.lower()


def normalize_projector_backend(value: ProjectorBackendInput) -> ProjectorBackend:
    """Normalize a public projector backend value."""
    backend = option_name(value)
    if backend == "jax":
        return "jax"
    if backend == "cuda":
        return "cuda"
    raise ValueError("backend must be one of 'jax' or 'cuda'")
