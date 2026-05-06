"""Public API for alignment orchestration."""

from __future__ import annotations

from tomojax.align._alternating import (
    AlternatingLevelSummary,
    AlternatingSmokeConfig,
    AlternatingSmokeResult,
    run_alternating_solver_smoke,
)
from tomojax.align._continuation import (
    ContinuationLevel,
    ContinuationSchedule,
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
from tomojax.align._setup_lm import SetupOnlyLMConfig, SetupOnlyLMResult, solve_setup_only_lm
from tomojax.align._smoke import AlignmentSmokeReport, run_alignment_smoke
from tomojax.align.pipeline import AlignConfig, align, align_multires

__all__ = [
    "AlignConfig",
    "AlignmentSmokeReport",
    "AlternatingLevelSummary",
    "AlternatingSmokeConfig",
    "AlternatingSmokeResult",
    "ContinuationLevel",
    "ContinuationSchedule",
    "JointSchurDiagnostics",
    "JointSchurLMConfig",
    "JointSchurLMResult",
    "PoseOnlyLMConfig",
    "PoseOnlyLMResult",
    "SetupOnlyLMConfig",
    "SetupOnlyLMResult",
    "adapt_joint_schur_damping",
    "adapt_joint_schur_trust_radius",
    "align",
    "align_multires",
    "joint_schur_normal_eq_summary",
    "reference_continuation_schedule",
    "run_alignment_smoke",
    "run_alternating_solver_smoke",
    "schur_step_from_jacobian",
    "solve_joint_schur_lm",
    "solve_pose_only_lm",
    "solve_setup_only_lm",
    "write_joint_schur_normal_eq_summary",
]
