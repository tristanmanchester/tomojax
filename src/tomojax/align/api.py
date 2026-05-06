"""Public API for alignment orchestration."""

from __future__ import annotations

from tomojax.align._joint_schur_lm import (
    JointSchurDiagnostics,
    JointSchurLMConfig,
    JointSchurLMResult,
    schur_step_from_jacobian,
    solve_joint_schur_lm,
)
from tomojax.align._pose_lm import PoseOnlyLMConfig, PoseOnlyLMResult, solve_pose_only_lm
from tomojax.align._setup_lm import SetupOnlyLMConfig, SetupOnlyLMResult, solve_setup_only_lm
from tomojax.align._smoke import AlignmentSmokeReport, run_alignment_smoke
from tomojax.align.pipeline import AlignConfig, align, align_multires

__all__ = [
    "AlignConfig",
    "AlignmentSmokeReport",
    "JointSchurDiagnostics",
    "JointSchurLMConfig",
    "JointSchurLMResult",
    "PoseOnlyLMConfig",
    "PoseOnlyLMResult",
    "SetupOnlyLMConfig",
    "SetupOnlyLMResult",
    "align",
    "align_multires",
    "run_alignment_smoke",
    "schur_step_from_jacobian",
    "solve_joint_schur_lm",
    "solve_pose_only_lm",
    "solve_setup_only_lm",
]
