"""Developer and advanced API for alignment orchestration.

Production callers should prefer the small package-root facade
``tomojax.align``. This module is intentionally broader: it collects typed
schedule, loss, profile, geometry-state, objective, and solver helpers used by
the CLI, benchmarks, focused tests, and advanced integrations while the align
module is being productionized into deeper owners.
"""

from __future__ import annotations

from tomojax.align._alternating import (
    GeometryUpdateSolver,
    GeometryUpdateVolumeSource,
    PreviewInitialization,
    PreviewReconstructionMaskSource,
    PreviewResidualFilterMode,
    PreviewVolumeSupport,
    ProjectionLossMode,
    StoppedPreviewPolicy,
)
from tomojax.align._continuation import (
    ContinuationLevel,
    ContinuationSchedule,
    ContinuationScheduleName,
    reference_continuation_schedule,
)
from tomojax.align._joint_schur_lm import (
    JointSchurDiagnostics,
    JointSchurLMConfig,
    JointSchurLMResult,
    adapt_joint_schur_damping,
    adapt_joint_schur_trust_radius,
    schur_step_from_jacobian,
    solve_joint_schur_lm,
)
from tomojax.align._pose_lm import PoseOnlyLMConfig, PoseOnlyLMResult, solve_pose_only_lm
from tomojax.align._profiles import (
    AlignmentProfile,
    AlignmentProfileInput,
    AlignmentProfilePolicy,
    FallbackPolicy,
    QualityTier,
    alignment_profile_policy,
    normalize_alignment_profile,
    profile_policy_from_config,
    resolve_profiled_cli_defaults,
)
from tomojax.align._setup_lm import SetupOnlyLMConfig, SetupOnlyLMResult, solve_setup_only_lm
from tomojax.align._setup_stage import (
    _optimize_setup_geometry_bilevel_for_level as optimize_setup_geometry_bilevel_for_level,
)
from tomojax.align.geometry.geometry_applier import BaseGeometryArrays, apply_alignment_state
from tomojax.align.geometry.geometry_blocks import (
    GeometryCalibrationState,
    geometry_with_axis_state,
    level_detector_grid,
    normalize_geometry_dofs,
    summarize_geometry_calibration_stats,
)
from tomojax.align.geometry.parametrizations import se3_from_5d
from tomojax.align.model.dof_specs import DofSpec, dof_spec
from tomojax.align.model.dofs import (
    DofBounds,
    normalize_alignment_dofs,
    normalize_bounds,
)
from tomojax.align.model.gauge import GaugeFixMode
from tomojax.align.model.schedules import (
    PUBLIC_SCHEDULE_PRESETS,
    AlignmentSchedule,
    AlignmentStage,
    GaugePolicy,
    ResolvedAlignmentSchedule,
    ResolvedAlignmentStage,
    resolve_alignment_schedule,
    schedule_preset,
)
from tomojax.align.model.state import AlignmentState, PoseState, SetupGeometryState
from tomojax.align.objectives.fixed_volume import (
    FixedVolumeProjectionObjective,
    ObjectiveProvenance,
    ObjectiveResult,
    project_and_score_stack,
    project_stack,
)
from tomojax.align.objectives.loss_adapters import LossAdapter, build_loss_adapter
from tomojax.align.objectives.loss_specs import (
    AlignmentLossConfig,
    AlignmentLossSchedule,
    AlignmentLossSpec,
    EdgeL2LossSpec,
    L2LossSpec,
    L2OtsuLossSpec,
    LossScheduleEntry,
    PWLSLossSpec,
    loss_spec_name,
    parse_loss_schedule,
    parse_loss_spec,
    resolve_loss_for_level,
    validate_loss_schedule_levels,
)
from tomojax.align.pipeline import (
    AlignConfig,
    AlignMultiresResumeState,
    AlignResumeState,
    align,
    align_multires,
)
from tomojax.verify import joint_schur_normal_eq_summary, write_joint_schur_normal_eq_summary

# Kept for direct-import compatibility while diagnostic ownership moves out of
# the product alignment facade. These names are intentionally omitted from
# __all__.
_DIAGNOSTIC_COMPATIBILITY_EXPORTS = (
    JointSchurDiagnostics,
    joint_schur_normal_eq_summary,
    write_joint_schur_normal_eq_summary,
)

__all__ = [
    "PUBLIC_SCHEDULE_PRESETS",
    "AlignConfig",
    "AlignMultiresResumeState",
    "AlignResumeState",
    "AlignmentLossConfig",
    "AlignmentLossSchedule",
    "AlignmentLossSpec",
    "AlignmentProfile",
    "AlignmentProfileInput",
    "AlignmentProfilePolicy",
    "AlignmentSchedule",
    "AlignmentStage",
    "AlignmentState",
    "BaseGeometryArrays",
    "ContinuationLevel",
    "ContinuationSchedule",
    "ContinuationScheduleName",
    "DofBounds",
    "DofSpec",
    "EdgeL2LossSpec",
    "FallbackPolicy",
    "FixedVolumeProjectionObjective",
    "GaugeFixMode",
    "GaugePolicy",
    "GeometryCalibrationState",
    "GeometryUpdateSolver",
    "GeometryUpdateVolumeSource",
    "JointSchurLMConfig",
    "JointSchurLMResult",
    "L2LossSpec",
    "L2OtsuLossSpec",
    "LossAdapter",
    "LossScheduleEntry",
    "ObjectiveProvenance",
    "ObjectiveResult",
    "PWLSLossSpec",
    "PoseOnlyLMConfig",
    "PoseOnlyLMResult",
    "PoseState",
    "PreviewInitialization",
    "PreviewReconstructionMaskSource",
    "PreviewResidualFilterMode",
    "PreviewVolumeSupport",
    "ProjectionLossMode",
    "QualityTier",
    "ResolvedAlignmentSchedule",
    "ResolvedAlignmentStage",
    "SetupGeometryState",
    "SetupOnlyLMConfig",
    "SetupOnlyLMResult",
    "StoppedPreviewPolicy",
    "adapt_joint_schur_damping",
    "adapt_joint_schur_trust_radius",
    "align",
    "align_multires",
    "alignment_profile_policy",
    "apply_alignment_state",
    "build_loss_adapter",
    "dof_spec",
    "geometry_with_axis_state",
    "level_detector_grid",
    "loss_spec_name",
    "normalize_alignment_dofs",
    "normalize_alignment_profile",
    "normalize_bounds",
    "normalize_geometry_dofs",
    "optimize_setup_geometry_bilevel_for_level",
    "parse_loss_schedule",
    "parse_loss_spec",
    "profile_policy_from_config",
    "project_and_score_stack",
    "project_stack",
    "reference_continuation_schedule",
    "resolve_alignment_schedule",
    "resolve_loss_for_level",
    "resolve_profiled_cli_defaults",
    "schedule_preset",
    "schur_step_from_jacobian",
    "se3_from_5d",
    "solve_joint_schur_lm",
    "solve_pose_only_lm",
    "solve_setup_only_lm",
    "summarize_geometry_calibration_stats",
    "validate_loss_schedule_levels",
]
