"""Shared FISTA settings; callers retain geometry and execution policy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tomojax.recon.fista_tv import FistaConfig

if TYPE_CHECKING:
    from tomojax.alignment._config import AlignConfig


def fista_config(cfg: AlignConfig, *, iterations: int, **overrides: Any) -> FistaConfig:
    """Translate alignment reconstruction settings without choosing a backend."""
    settings: dict[str, Any] = {
        "iterations": iterations,
        "tv_weight": 0.0,
        "regulariser": "huber_tv",
        "lipschitz": cfg.lipschitz,
        "nonnegative": bool(cfg.nonnegative),
        "ray_integrator": getattr(cfg, "ray_integrator", "sampled"),
    }
    settings.update(overrides)
    return FistaConfig(**settings)
