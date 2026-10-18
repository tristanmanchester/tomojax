"""Backend-facing re-export for optional CUDA projector capability lookup."""

from __future__ import annotations

from tomojax.core.cuda_projector_resolver import (
    CudaModuleCapability,
    resolve_cuda_callable,
    resolve_cuda_module,
)

__all__ = [
    "CudaModuleCapability",
    "resolve_cuda_callable",
    "resolve_cuda_module",
]
