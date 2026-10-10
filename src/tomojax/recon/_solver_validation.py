"""Scalar configuration checks before iterative solvers allocate or compile."""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .fista_tv import FistaConfig
    from .spdhg_tv import SPDHGConfig


def _count(value: int, name: str, *, minimum: int, context: str) -> None:
    requirement = "nonnegative" if minimum == 0 else "positive"
    message = f"{context}: {name} must be a {requirement} integer"
    try:
        count = operator.index(value)
    except TypeError as error:
        raise ValueError(message) from error
    if isinstance(value, bool) or count < minimum:
        raise ValueError(message)


def _weight(value: float, name: str, *, positive: bool = False, context: str) -> None:
    requirement = "positive" if positive else "nonnegative"
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{context}: {name} must be finite and {requirement}")


def validate_fista_config(cfg: FistaConfig) -> tuple[bool, float | None, float | None]:
    """Check scalar settings and return normalized feasibility constraints."""
    context = "fista_tv config"
    for name in ("iterations", "recon_patience"):
        _count(getattr(cfg, name), name, minimum=0, context=context)
    for name in ("projector_unroll", "tv_prox_iterations", "power_iterations"):
        _count(getattr(cfg, name), name, minimum=1, context=context)
    if cfg.views_per_batch is not None:
        _count(cfg.views_per_batch, "views_per_batch", minimum=1, context=context)
    _weight(cfg.tv_weight, "tv_weight", context=context)
    if cfg.lipschitz is not None:
        _weight(cfg.lipschitz, "lipschitz", positive=True, context=context)
    if cfg.recon_rel_tol is not None:
        _weight(cfg.recon_rel_tol, "recon_rel_tol", context=context)
    if cfg.grad_mode not in {"auto", "batched", "stream"}:
        raise ValueError(f"{context}: grad_mode must be 'auto', 'batched' or 'stream'")

    return _fista_constraints(cfg)


def _fista_constraints(cfg: FistaConfig) -> tuple[bool, float | None, float | None]:
    lower = None if cfg.lower_bound is None else float(cfg.lower_bound)
    upper = None if cfg.upper_bound is None else float(cfg.upper_bound)
    if lower is not None and not math.isfinite(lower):
        raise ValueError("fista_tv constraints: lower_bound must be finite when provided")
    if upper is not None and not math.isfinite(upper):
        raise ValueError("fista_tv constraints: upper_bound must be finite when provided")
    effective_lower = max(0.0, lower) if cfg.nonnegative and lower is not None else lower
    if cfg.nonnegative and effective_lower is None:
        effective_lower = 0.0
    if upper is not None and effective_lower is not None and upper < effective_lower:
        raise ValueError(
            "fista_tv constraints: upper_bound must be greater than or equal to "
            "the effective lower bound"
        )
    return bool(cfg.nonnegative), lower, upper


def validate_spdhg_config(cfg: SPDHGConfig) -> None:
    """Check scalar settings, including independently optional step sizes."""
    context = "spdhg_tv config"
    for name in ("iterations", "log_every"):
        _count(getattr(cfg, name), name, minimum=0, context=context)
    for name in ("views_per_batch", "projector_unroll"):
        _count(getattr(cfg, name), name, minimum=1, context=context)
    for name in ("tv_weight", "theta"):
        _weight(getattr(cfg, name), name, context=context)
    for name in ("tau", "sigma_data", "sigma_tv"):
        value = getattr(cfg, name)
        if value is not None:
            _weight(value, name, positive=True, context=context)
