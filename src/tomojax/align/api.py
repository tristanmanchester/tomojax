"""Public API for alignment orchestration."""

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
    joint_schur_normal_eq_summary,
    schur_step_from_jacobian,
    solve_joint_schur_lm,
    write_joint_schur_normal_eq_summary,
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
from tomojax.align.objectives.loss_specs import (
    AlignmentLossConfig,
    parse_loss_schedule,
    parse_loss_spec,
    validate_loss_schedule_levels,
)
from tomojax.align.pipeline import (
    AlignConfig,
    AlignMultiresResumeState,
    AlignResumeState,
    align,
    align_multires,
)

__all__ = [
    "PUBLIC_SCHEDULE_PRESETS",
    "AlignConfig",
    "AlignMultiresResumeState",
    "AlignResumeState",
    "AlignmentLossConfig",
    "AlignmentProfile",
    "AlignmentProfileInput",
    "AlignmentProfilePolicy",
    "AlignmentSchedule",
    "AlignmentStage",
    "ContinuationLevel",
    "ContinuationSchedule",
    "ContinuationScheduleName",
    "DofBounds",
    "DofSpec",
    "FallbackPolicy",
    "GaugeFixMode",
    "GaugePolicy",
    "GeometryUpdateSolver",
    "GeometryUpdateVolumeSource",
    "JointSchurDiagnostics",
    "JointSchurLMConfig",
    "JointSchurLMResult",
    "PoseOnlyLMConfig",
    "PoseOnlyLMResult",
    "PreviewInitialization",
    "PreviewReconstructionMaskSource",
    "PreviewResidualFilterMode",
    "PreviewVolumeSupport",
    "ProjectionLossMode",
    "QualityTier",
    "ResolvedAlignmentSchedule",
    "ResolvedAlignmentStage",
    "SetupOnlyLMConfig",
    "SetupOnlyLMResult",
    "StoppedPreviewPolicy",
    "adapt_joint_schur_damping",
    "adapt_joint_schur_trust_radius",
    "align",
    "align_multires",
    "alignment_profile_policy",
    "dof_spec",
    "joint_schur_normal_eq_summary",
    "normalize_alignment_dofs",
    "normalize_alignment_profile",
    "normalize_bounds",
    "parse_loss_schedule",
    "parse_loss_spec",
    "profile_policy_from_config",
    "reference_continuation_schedule",
    "resolve_alignment_schedule",
    "resolve_profiled_cli_defaults",
    "schedule_preset",
    "schur_step_from_jacobian",
    "se3_from_5d",
    "solve_joint_schur_lm",
    "solve_pose_only_lm",
    "solve_setup_only_lm",
    "validate_loss_schedule_levels",
    "write_joint_schur_normal_eq_summary",
]
