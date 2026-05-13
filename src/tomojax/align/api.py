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
from tomojax.align.pipeline import (
    AlignConfig,
    AlignMultiresResumeState,
    AlignResumeState,
    align,
    align_multires,
)

__all__ = [
    "AlignConfig",
    "AlignMultiresResumeState",
    "AlignResumeState",
    "AlignmentProfile",
    "AlignmentProfileInput",
    "AlignmentProfilePolicy",
    "ContinuationLevel",
    "ContinuationSchedule",
    "ContinuationScheduleName",
    "FallbackPolicy",
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
    "SetupOnlyLMConfig",
    "SetupOnlyLMResult",
    "StoppedPreviewPolicy",
    "adapt_joint_schur_damping",
    "adapt_joint_schur_trust_radius",
    "align",
    "align_multires",
    "alignment_profile_policy",
    "joint_schur_normal_eq_summary",
    "normalize_alignment_profile",
    "profile_policy_from_config",
    "reference_continuation_schedule",
    "resolve_profiled_cli_defaults",
    "schur_step_from_jacobian",
    "solve_joint_schur_lm",
    "solve_pose_only_lm",
    "solve_setup_only_lm",
    "write_joint_schur_normal_eq_summary",
]
