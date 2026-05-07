"""Public API for differentiable forward models."""

from __future__ import annotations

from tomojax.forward._filters import (
    ResidualFilterConfig,
    ResidualFilterKind,
    ResidualFilterResult,
    apply_residual_filter,
    apply_residual_filter_schedule,
)
from tomojax.forward._projector import (
    PROJECTION_OPERATOR,
    CoreProjectionGeometry,
    V2ProjectionOperatorName,
    core_projection_geometry_from_arrays,
    core_projection_geometry_from_state,
    project_parallel_reference,
    project_parallel_reference_arrays,
)
from tomojax.forward._residuals import (
    ResidualResult,
    masked_whitened_residual,
    pseudo_huber_loss,
    pseudo_huber_weights,
    residual_loss,
    robust_residual_scale,
)

__all__ = [
    "PROJECTION_OPERATOR",
    "CoreProjectionGeometry",
    "ResidualFilterConfig",
    "ResidualFilterKind",
    "ResidualFilterResult",
    "ResidualResult",
    "V2ProjectionOperatorName",
    "apply_residual_filter",
    "apply_residual_filter_schedule",
    "core_projection_geometry_from_arrays",
    "core_projection_geometry_from_state",
    "masked_whitened_residual",
    "project_parallel_reference",
    "project_parallel_reference_arrays",
    "pseudo_huber_loss",
    "pseudo_huber_weights",
    "residual_loss",
    "robust_residual_scale",
]
