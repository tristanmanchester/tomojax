"""Failure-report assembly for verification artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

FAILURE_CLASSES = (
    "geometry_not_observable",
    "pose_overfit",
    "nuisance_unmodelled",
    "backend_fallback_unexpected",
    "reconstruction_underconverged",
    "motion_model_insufficient",
    "deformation_suspected",
    "bad_input_metadata",
    "nan_or_inf",
    "no_improvement",
)


def failure_report_from_gates(gates: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Build the common failure-report envelope from gate evidence."""
    gate_payloads = [dict(gate) for gate in gates]
    warning_gates = [gate for gate in gate_payloads if gate.get("passed") is False]
    warnings = failure_warnings_from_gates(warning_gates)
    return {
        "schema": "tomojax.failure_report.v1",
        "status": "warning" if warnings else "passed",
        "failure": None,
        "failure_classes": list(FAILURE_CLASSES),
        "gates": gate_payloads,
        "warnings": warnings,
    }


def failure_warnings_from_gates(
    warning_gates: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return warning payloads for failed warning-class gates."""
    return cast(
        "list[dict[str, object]]",
        [
            {
                "class": "no_improvement",
                "severity": "warning",
                "evidence": [str(gate["evidence"])],
                "recommended_action": (
                    "run a longer continuation profile or enable real geometry LM/GN updates"
                ),
            }
            for gate in warning_gates
            if gate.get("name") == "projection_residual_improvement"
        ]
        + [
            {
                "class": "nuisance_unmodelled",
                "severity": "warning",
                "evidence": [str(gate["evidence"])],
                "recommended_action": (
                    "enable gain/offset nuisance fitting or inspect flat-field correction"
                ),
            }
            for gate in warning_gates
            if gate.get("name") == "nuisance_residual_structure"
        ]
        + [
            {
                "class": "bad_input_metadata",
                "severity": "warning",
                "evidence": [str(gate["evidence"])],
                "recommended_action": (
                    "inspect generated synthetic sidecar manifest and artifact paths"
                ),
            }
            for gate in warning_gates
            if gate.get("name") == "synthetic_sidecar_consistency"
        ],
    )


__all__ = ["FAILURE_CLASSES", "failure_report_from_gates", "failure_warnings_from_gates"]
