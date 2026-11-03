"""Detector-constant nuisance elimination without changing the projector."""

from __future__ import annotations

from .types import DataTerm


def project_residual(residual, data_term: DataTerm, *, weights=None, xp=None):
    """Profile one constant per complete detector view, never per detector tile.

    Weights specify the least-squares metric. The result is unweighted; a
    weighted gradient still applies its weights before the exact transpose.
    Unobserved pixels are zero, including when their input value is NaN.
    """
    if weights is None:
        return (
            residual
            if data_term == "absolute"
            else residual - residual.mean(axis=(-2, -1), keepdims=True)
        )
    if xp is None:
        import jax.numpy as xp

    if data_term == "absolute":
        return xp.where(weights > 0, residual, 0)
    active = weights > 0
    residual = xp.where(active, residual, 0)
    total = weights.sum(axis=(-2, -1), keepdims=True)
    mean = (weights * residual).sum(axis=(-2, -1), keepdims=True) / xp.where(total > 0, total, 1)
    return xp.where(active, residual - mean, 0)
