"""Backend-facing re-export for optional Pallas projector capability lookup."""

from __future__ import annotations

from tomojax.core.pallas_resolver import (
    PallasModuleCapability,
    resolve_pallas_callable,
    resolve_pallas_module,
)

__all__ = [
    "PallasModuleCapability",
    "resolve_pallas_callable",
    "resolve_pallas_module",
]
