"""Public API for verification and artifact reports."""

from __future__ import annotations

from tomojax.verify._artifacts import (
    ArtifactValidationError,
    ArtifactValidationIssue,
    ArtifactValidationReport,
    inspect_run_artifacts,
    validate_run_artifacts,
)
from tomojax.verify._failure_report import (
    FAILURE_CLASSES,
    failure_report_from_gates,
    failure_warnings_from_gates,
)
from tomojax.verify._residual_structure import residual_structure_summary
from tomojax.verify._schur_summary import (
    joint_schur_normal_eq_summary,
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
