from __future__ import annotations

from typing import cast

from tomojax.verify import FAILURE_CLASSES, failure_report_from_gates


def test_failure_report_warns_for_structured_nuisance_gate() -> None:
    report = failure_report_from_gates(
        [
            {
                "name": "finite_outputs",
                "passed": True,
                "severity": "error",
                "evidence": "finite",
            },
            {
                "name": "nuisance_residual_structure",
                "passed": False,
                "severity": "warning",
                "evidence": {"passed": False, "column_mean_rms_ratio": 0.8},
            },
        ]
    )

    assert report["schema"] == "tomojax.failure_report.v1"
    assert report["status"] == "warning"
    assert report["failure"] is None
    assert report["failure_classes"] == list(FAILURE_CLASSES)
    warnings = cast("list[dict[str, object]]", report["warnings"])
    assert [warning["class"] for warning in warnings] == ["nuisance_unmodelled"]


def test_failure_report_passes_when_warning_gates_pass() -> None:
    report = failure_report_from_gates(
        [
            {
                "name": "nuisance_residual_structure",
                "passed": True,
                "severity": "warning",
                "evidence": {"passed": True},
            }
        ]
    )

    assert report["status"] == "passed"
    assert report["warnings"] == []
