from __future__ import annotations

# pyright: reportPrivateUsage=false
from math import radians
from typing import cast

# check-public-imports: allow-private
from tomojax.align._alternating_artifacts import _benchmark_manifest_evaluation


def test_benchmark_manifest_evaluates_detector_roll_alias() -> None:
    evaluation = _benchmark_manifest_evaluation(
        criteria={"detector_roll_error_deg_lt": 0.10},
        geometry_recovery={"detector_roll_error_rad": radians(0.05)},
    )

    roll = cast("dict[str, object]", evaluation["detector_roll_error_deg_lt"])
    assert roll["status"] == "passed"
    assert roll["value"] == radians(0.05)
    assert roll["threshold"] == radians(0.10)


def test_benchmark_manifest_evaluates_axis_roll_combined_alias() -> None:
    evaluation = _benchmark_manifest_evaluation(
        criteria={"axis_roll_error_deg_lt": 0.20},
        geometry_recovery={
            "axis_error_rad": radians(0.18),
            "detector_roll_error_rad": radians(0.05),
        },
    )

    axis_roll = cast("dict[str, object]", evaluation["axis_roll_error_deg_lt"])
    assert axis_roll["status"] == "passed"
    assert axis_roll["value"] == radians(0.18)
    assert axis_roll["threshold"] == radians(0.20)


def test_benchmark_manifest_fails_backend_policy_without_explicit_fallback() -> None:
    evaluation = _benchmark_manifest_evaluation(
        criteria={"backend_policy": "calibrated_grid_fallback_explicit"},
        geometry_recovery={},
        backend={"fallbacks": []},
    )

    backend = cast("dict[str, object]", evaluation["backend_policy"])
    assert backend["status"] == "failed"
    assert backend["value"] == 0
    assert backend["reason"] == (
        "expected calibrated-grid fallback provenance but backend fallbacks were empty"
    )


def test_benchmark_manifest_passes_backend_policy_with_explicit_fallback() -> None:
    evaluation = _benchmark_manifest_evaluation(
        criteria={"backend_policy": "calibrated_grid_fallback_explicit"},
        geometry_recovery={},
        backend={"fallbacks": [{"reason": "calibrated_grid_fallback"}]},
    )

    backend = cast("dict[str, object]", evaluation["backend_policy"])
    assert backend["status"] == "passed"
    assert backend["value"] == 1
    assert backend["threshold"] == "calibrated_grid_fallback_explicit"


def test_benchmark_manifest_evaluates_det_v_policy_when_recovered() -> None:
    evaluation = _benchmark_manifest_evaluation(
        criteria={"det_v_policy": "recovered_or_reported_unobservable"},
        geometry_recovery={
            "det_v_realized_rmse_px": 0.05,
            "det_v_realized_rmse_px_passed": True,
        },
    )

    det_v = cast("dict[str, object]", evaluation["det_v_policy"])
    assert det_v["status"] == "passed"
    assert det_v["value"] == 0.05
    assert det_v["threshold"] == "recovered_or_reported_unobservable"


def test_benchmark_manifest_keeps_det_v_policy_not_evaluated_without_evidence() -> None:
    evaluation = _benchmark_manifest_evaluation(
        criteria={"det_v_policy": "recovered_or_reported_unobservable"},
        geometry_recovery={
            "det_v_realized_rmse_px": 4.0,
            "det_v_realized_rmse_px_passed": False,
        },
    )

    det_v = cast("dict[str, object]", evaluation["det_v_policy"])
    assert det_v["status"] == "not_evaluated"
    assert det_v["value"] == 4.0
    assert det_v["reason"] == (
        "det_v was not recovered and unobservability policy evidence is not in benchmark_result"
    )
