"""State and error contracts for Pallas projector orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


class PallasProjectorUnsupported(ValueError):
    """Raised when the experimental Pallas projector cannot handle a call."""


def _unsupported(message: str) -> str:
    return f"pallas_projector_unsupported: {message}"


@dataclass(frozen=True)
class PallasProjectorTraversalMetadata:
    """Common per-ray traversal arrays and launch metadata for cached Pallas paths."""

    ix0: jnp.ndarray
    iy0: jnp.ndarray
    iz0: jnp.ndarray
    n_steps_ray: jnp.ndarray
    step_size: float
    n_steps: int
    resolved_n_steps: int
    nx: int
    ny: int
    nz: int
    nv: int
    nu: int
    tile_shape: tuple[int, int]
    num_warps: int
    kernel_variant: str
    kernel_variant_id: int
    gather_dtype: str


@dataclass(frozen=True)
class PallasForwardProjectorTraversalState:
    """Prepared single-view traversal state for fixed-geometry Pallas benchmarking."""

    traversal: PallasProjectorTraversalMetadata
    dix: float
    diy: float
    diz: float

    def __getattr__(self, name: str) -> object:
        return getattr(self.traversal, name)


@dataclass(frozen=True)
class PallasForwardProjectorStackTraversalState:
    """Prepared pose-stack traversal state for fixed pose-stack Pallas benchmarking."""

    traversal: PallasProjectorTraversalMetadata
    dix: jnp.ndarray
    diy: jnp.ndarray
    diz: jnp.ndarray
    n_views: int

    def __getattr__(self, name: str) -> object:
        return getattr(self.traversal, name)


__all__ = [
    "PallasForwardProjectorStackTraversalState",
    "PallasForwardProjectorTraversalState",
    "PallasProjectorTraversalMetadata",
    "PallasProjectorUnsupported",
]
