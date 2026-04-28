from __future__ import annotations

"""Compatibility facade for public alignment loss specs and adapters.

This module exists for historical ``tomojax.align.losses`` imports. New code
should import typed loss specs from ``align.objectives.loss_specs`` and adapters
from ``align.objectives.loss_adapters`` instead of treating the compatibility
facade as an extension point. Private kernels live in
``align.objectives.loss_kernels`` and are intentionally not re-exported here.
"""

from .loss_adapters import (
    GaussNewtonWeightFn,
    LossAdapter,
    LossBuilderFn,
    PerViewLossFn,
    build_loss,
    build_loss_adapter,
    loss_supports_setup_validation_lm,
)
from .loss_specs import (
    AlignmentLossConfig,
    AlignmentLossSchedule,
    AlignmentLossSpec,
    CorrelationLossSpec,
    EdgeL2LossSpec,
    GradientLossSpec,
    InformationLossSpec,
    L2LossSpec,
    L2OtsuLossSpec,
    LossScheduleEntry,
    MindLossSpec,
    PWLSLossSpec,
    PoissonLossSpec,
    RobustLossSpec,
    SSIMLossSpec,
    SWDLossSpec,
    TverskyLossSpec,
    canonicalize_loss_kind,
    loss_is_within_relative_tolerance,
    loss_spec_name,
    loss_spec_params,
    parse_loss_schedule,
    parse_loss_spec,
    resolve_loss_for_level,
    validate_loss_schedule_levels,
)
from .loss_state import LossState

__all__ = [
    "AlignmentLossConfig",
    "AlignmentLossSchedule",
    "AlignmentLossSpec",
    "CorrelationLossSpec",
    "EdgeL2LossSpec",
    "GaussNewtonWeightFn",
    "GradientLossSpec",
    "InformationLossSpec",
    "L2LossSpec",
    "L2OtsuLossSpec",
    "LossAdapter",
    "LossBuilderFn",
    "LossScheduleEntry",
    "LossState",
    "MindLossSpec",
    "PWLSLossSpec",
    "PerViewLossFn",
    "PoissonLossSpec",
    "RobustLossSpec",
    "SSIMLossSpec",
    "SWDLossSpec",
    "TverskyLossSpec",
    "build_loss",
    "build_loss_adapter",
    "canonicalize_loss_kind",
    "loss_is_within_relative_tolerance",
    "loss_spec_name",
    "loss_spec_params",
    "loss_supports_setup_validation_lm",
    "parse_loss_schedule",
    "parse_loss_spec",
    "resolve_loss_for_level",
    "validate_loss_schedule_levels",
]
