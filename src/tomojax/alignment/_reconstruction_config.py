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
        "tv_weight": float(cfg.tv_weight),
        "regulariser": cfg.regulariser,
        "huber_delta": float(cfg.huber_delta),
        "tv_prox_iterations": int(cfg.tv_prox_iterations),
        "lipschitz": cfg.lipschitz,
        "nonnegative": bool(cfg.nonnegative),
        "ray_integrator": getattr(cfg, "ray_integrator", "sampled"),
    }
    settings.update(overrides)
    return FistaConfig(**settings)
