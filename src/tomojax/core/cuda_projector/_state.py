"""State and error contracts for CUDA projector orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


class CudaProjectorUnsupported(ValueError):
    """Raised when the experimental CUDA projector cannot handle a call."""


def _unsupported(message: str) -> str:
    return f"cuda_projector_unsupported: {message}"


@dataclass(frozen=True)
class CudaProjectorTraversalMetadata:
    """Common per-ray traversal arrays and launch metadata for cached CUDA paths."""

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
class CudaForwardProjectorTraversalState:
    """Prepared single-view traversal state for fixed-geometry CUDA benchmarking."""

    traversal: CudaProjectorTraversalMetadata
    dix: float
    diy: float
    diz: float

    def __getattr__(self, name: str) -> object:
        return getattr(self.traversal, name)


@dataclass(frozen=True)
class CudaForwardProjectorStackTraversalState:
    """Prepared pose-stack traversal state for fixed pose-stack CUDA benchmarking."""

    traversal: CudaProjectorTraversalMetadata
    dix: jnp.ndarray
    diy: jnp.ndarray
    diz: jnp.ndarray
    n_views: int

    def __getattr__(self, name: str) -> object:
        return getattr(self.traversal, name)


__all__ = [
    "CudaForwardProjectorStackTraversalState",
    "CudaForwardProjectorTraversalState",
    "CudaProjectorTraversalMetadata",
    "CudaProjectorUnsupported",
]
