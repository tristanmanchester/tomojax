from __future__ import annotations

# pyright: reportPrivateUsage=false
from math import radians
from typing import cast

import numpy as np

# check-public-imports: allow-private
from tomojax.align._alternating_artifacts import (
    _backend_fallbacks_from_sidecar,
    _benchmark_manifest_evaluation,
    _pose_jump_exclusion_payload,
)
from tomojax.geometry import GeometryState


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


def test_backend_fallbacks_from_sidecar_reports_calibrated_noncanonical_grid() -> None:
    fallbacks = _backend_fallbacks_from_sidecar(
        {"detector_grid": "calibrated_noncanonical"},
    )

    assert fallbacks == [
        {
            "reason": "calibrated_noncanonical_detector_grid",
            "requested_policy": "calibrated_grid_fallback_explicit",
            "actual_backend": "core_trilinear_ray",
        }
    ]


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


def test_benchmark_manifest_passes_det_v_policy_when_reported_frozen() -> None:
    evaluation = _benchmark_manifest_evaluation(
        criteria={"det_v_policy": "recovered_or_reported_unobservable"},
        geometry_recovery={
            "det_v_realized_rmse_px": 4.0,
            "det_v_realized_rmse_px_passed": False,
        },
        weak_dof_policy={
            "decisions": {
                "det_v_px": {
                    "decision": "keep_frozen",
                    "reason": "det_v_px is frozen in the current geometry",
                }
            }
        },
    )

    det_v = cast("dict[str, object]", evaluation["det_v_policy"])
    assert det_v["status"] == "passed"
    assert det_v["value"] == 4.0
    assert det_v["reason"] == "det_v_px is frozen in the current geometry"


def test_benchmark_manifest_reports_missing_object_motion_suspicion_payload() -> None:
    evaluation = _benchmark_manifest_evaluation(
        criteria={"core_solver": "flags_object_motion_suspected"},
        geometry_recovery={},
    )

    core_solver = cast("dict[str, object]", evaluation["core_solver"])
    assert core_solver["status"] == "not_evaluated"
    assert core_solver["reason"] == "object-motion suspicion payload is not in benchmark_result"


def test_benchmark_manifest_reports_missing_bad_view_detection_payload() -> None:
    evaluation = _benchmark_manifest_evaluation(
        criteria={"bad_views_flagged": True},
        geometry_recovery={},
    )

    bad_views = cast("dict[str, object]", evaluation["bad_views_flagged"])
    assert bad_views["status"] == "not_evaluated"
    assert bad_views["reason"] == "bad-view detection payload is not in benchmark_result"


def test_benchmark_manifest_reports_missing_jump_exclusion_payload() -> None:
    evaluation = _benchmark_manifest_evaluation(
        criteria={"pose_dx_dz_rmse_px_lt_except_jumps": 2.0},
        geometry_recovery={},
    )

    pose = cast("dict[str, object]", evaluation["pose_dx_dz_rmse_px_lt_except_jumps"])
    assert pose["status"] == "not_evaluated"
    assert pose["reason"] == "pose jump-exclusion mask is not in benchmark_result"


def test_benchmark_manifest_evaluates_jump_excluded_pose_metric() -> None:
    evaluation = _benchmark_manifest_evaluation(
        criteria={"pose_dx_dz_rmse_px_lt_except_jumps": 2.0},
        geometry_recovery={},
        pose_jump_exclusion={"dx_dz_rmse_px_except_jumps": 1.25},
    )

    pose = cast("dict[str, object]", evaluation["pose_dx_dz_rmse_px_lt_except_jumps"])
    assert pose["status"] == "passed"
    assert pose["value"] == 1.25
    assert pose["threshold"] == 2.0


def test_pose_jump_exclusion_payload_uses_canonical_pose_and_excludes_jump() -> None:
    truth = GeometryState.zeros(8)
    truth = GeometryState(
        setup=truth.setup,
        pose=truth.pose.with_updates(
            dx_px=np.array([0.0, 0.1, 0.2, 20.0, 20.1, 20.2, 20.3, 20.4]),
            dz_px=np.array([0.0, -0.1, -0.2, -18.0, -18.1, -18.2, -18.3, -18.4]),
        ),
    )
    final = GeometryState(
        setup=truth.setup,
        pose=truth.pose.with_updates(
            dx_px=truth.pose.dx_px + 3.0,
            dz_px=truth.pose.dz_px,
        ),
    )

    payload = _pose_jump_exclusion_payload(
        true_geometry=truth,
        final_geometry=final,
    )

    assert payload["excluded_view_indices"] == [1, 2, 3, 4, 5]
    assert payload["evaluated_view_count"] == 3
    assert payload["dx_dz_rmse_px_except_jumps"] == 0.0
