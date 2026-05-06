"""Public API for differentiable forward models."""

from __future__ import annotations

from tomojax.forward._filters import (
    ResidualFilterConfig,
    ResidualFilterKind,
    ResidualFilterResult,
    apply_residual_filter,
    apply_residual_filter_schedule,
)
from tomojax.forward._projector import project_parallel_reference, project_parallel_reference_arrays
from tomojax.forward._residuals import (
    ResidualResult,
    masked_whitened_residual,
    pseudo_huber_loss,
    pseudo_huber_weights,
    residual_loss,
)

__all__ = [
    "ResidualFilterConfig",
    "ResidualFilterKind",
    "ResidualFilterResult",
    "ResidualResult",
    "apply_residual_filter",
    "apply_residual_filter_schedule",
    "masked_whitened_residual",
    "project_parallel_reference",
    "project_parallel_reference_arrays",
    "pseudo_huber_loss",
    "pseudo_huber_weights",
    "residual_loss",
]
