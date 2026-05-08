"""Forward-model public facade."""

from __future__ import annotations

from tomojax.forward.api import (
    PROJECTION_OPERATOR,
    CoreProjectionGeometry,
    ResidualFilterConfig,
    ResidualFilterKind,
    ResidualFilterResult,
    ResidualLossMode,
    ResidualResult,
    V2ProjectionOperatorName,
    apply_residual_filter,
    apply_residual_filter_schedule,
    core_projection_geometry_from_arrays,
    core_projection_geometry_from_state,
    masked_whitened_residual,
    nominal_axis_unit_from_geometry,
    project_parallel_reference,
    project_parallel_reference_arrays,
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
    "ResidualLossMode",
    "ResidualResult",
    "V2ProjectionOperatorName",
    "apply_residual_filter",
    "apply_residual_filter_schedule",
    "core_projection_geometry_from_arrays",
    "core_projection_geometry_from_state",
    "masked_whitened_residual",
    "nominal_axis_unit_from_geometry",
    "project_parallel_reference",
    "project_parallel_reference_arrays",
    "pseudo_huber_loss",
    "pseudo_huber_weights",
    "residual_loss",
    "robust_residual_scale",
]
