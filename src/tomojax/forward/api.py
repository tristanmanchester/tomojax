"""Public API for differentiable forward models."""

from __future__ import annotations

from tomojax.forward._filters import (
    ResidualFilterConfig,
    ResidualFilterKind,
    ResidualFilterResult,
    apply_residual_filter,
    apply_residual_filter_schedule,
)
from tomojax.forward._joseph import (
    joseph_l2_value_and_grad,
    joseph_pose_normal_equations,
    project_joseph,
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
    "ResidualFilterConfig",
    "ResidualFilterKind",
    "ResidualFilterResult",
    "ResidualLossMode",
    "ResidualResult",
    "apply_residual_filter",
    "apply_residual_filter_schedule",
    "joseph_l2_value_and_grad",
    "joseph_pose_normal_equations",
    "masked_whitened_residual",
    "project_joseph",
    "pseudo_huber_loss",
    "pseudo_huber_weights",
    "residual_loss",
    "robust_residual_scale",
]
