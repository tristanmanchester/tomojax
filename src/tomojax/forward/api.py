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
    ProjectionArrayGeometryInput,
    ProjectionOperatorName,
    core_projection_geometry_from_input,
    core_projection_geometry_from_state,
    nominal_axis_unit_from_geometry,
    project_parallel_reference,
    project_parallel_reference_from_input,
)
from tomojax.forward._residuals import (
    ResidualLossMode,
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
    "ProjectionArrayGeometryInput",
    "ProjectionOperatorName",
    "ResidualFilterConfig",
    "ResidualFilterKind",
    "ResidualFilterResult",
    "ResidualLossMode",
    "ResidualResult",
    "apply_residual_filter",
    "apply_residual_filter_schedule",
    "core_projection_geometry_from_input",
    "core_projection_geometry_from_state",
    "masked_whitened_residual",
    "nominal_axis_unit_from_geometry",
    "project_parallel_reference",
    "project_parallel_reference_from_input",
    "pseudo_huber_loss",
    "pseudo_huber_weights",
    "residual_loss",
    "robust_residual_scale",
]
