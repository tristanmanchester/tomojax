"""Verification public facade."""

from __future__ import annotations

from tomojax.verify.api import (
    FAILURE_CLASSES,
    ArtifactValidationError,
    ArtifactValidationIssue,
    ArtifactValidationReport,
    failure_report_from_gates,
    failure_warnings_from_gates,
    inspect_run_artifacts,
    joint_schur_normal_eq_summary,
    residual_structure_summary,
    validate_run_artifacts,
    write_joint_schur_normal_eq_summary,
)

__all__ = [
    "FAILURE_CLASSES",
    "ArtifactValidationError",
    "ArtifactValidationIssue",
    "ArtifactValidationReport",
    "failure_report_from_gates",
    "failure_warnings_from_gates",
    "inspect_run_artifacts",
    "joint_schur_normal_eq_summary",
    "residual_structure_summary",
    "validate_run_artifacts",
    "write_joint_schur_normal_eq_summary",
]
