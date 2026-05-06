from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING, cast

import numpy as np

from tomojax.align.api import (
    AlternatingAlignmentSolver,
    AlternatingSmokeConfig,
    reference_continuation_schedule,
    run_alternating_solver_smoke,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from tomojax.align.api import AlternatingSmokeResult


def test_alternating_solver_smoke_writes_artifacts(tmp_path: Path) -> None:
    result = run_alternating_solver_smoke(tmp_path)

    _assert_smoke_result_shape_and_exit(result)
    expected = _expected_artifacts()
    assert set(result.artifacts) == expected
    for path in result.artifacts.values():
        assert path.exists()
    _assert_artifact_index(result, expected)
    _assert_summary_rows(result)
    _assert_saved_volume(result)
    _assert_manifest(result)
    _assert_gauge_reports(result)
    _assert_audit_reports(result)
    _assert_residual_metrics(result)
    assert abs(float(np.mean(result.final_geometry.pose.dx_px))) < 1.0e-12
    assert abs(float(np.mean(result.final_geometry.pose.phi_residual_rad))) < 1.0e-12


def _assert_smoke_result_shape_and_exit(result: AlternatingSmokeResult) -> None:
    assert result.final_volume.shape == (32, 32, 32)
    assert result.verification["size"] == 32
    assert result.verification["seed"] == 17
    assert result.verification["coarse_verified"] is True
    assert result.verification["level1_geometry_skipped"] is True
    assert result.verification["skipped_levels"] == [2]
    thresholds = cast("dict[str, float]", result.verification["thresholds"])
    assert thresholds["gauge_stability_tolerance"] == 1.0e-10
    assert thresholds["parameter_update_tolerance"] == 2.0
    assert result.levels[0].geometry_updates == 1
    assert result.levels[0].executed_geometry_updates == 1
    assert result.levels[0].residual_filter_kinds == ("lowpass_gaussian",)
    assert result.levels[0].loss_nonincreasing
    assert result.levels[0].finite_loss
    assert result.levels[0].gauge_stable
    assert result.levels[0].parameter_update_small
    assert result.levels[0].parameter_update_norm == 1.25
    assert result.levels[1].skipped_level is True
    assert result.levels[1].residual_filter_kinds == (
        "lowpass_gaussian",
        "bandpass_difference_of_gaussians",
    )
    assert result.levels[-1].skipped_geometry is True
    assert result.levels[-1].residual_filter_kinds == ("raw",)
    assert result.levels[-1].geometry_updates == 1
    assert result.levels[-1].executed_geometry_updates == 0


def _expected_artifacts() -> set[str]:
    return {
        "alignment_summary_csv",
        "artifact_index_json",
        "backend_report_json",
        "config_resolved_toml",
        "failure_report_json",
        "final_volume_npy",
        "fista_trace_csv",
        "gauge_policy_json",
        "gauge_report_json",
        "geometry_final_json",
        "geometry_initial_json",
        "input_summary_json",
        "mask_summary_json",
        "observability_report_json",
        "pose_decomposition_csv",
        "pose_params_csv",
        "projection_stats_json",
        "residual_metrics_csv",
        "run_manifest_json",
        "verification_json",
    }


def _assert_artifact_index(result: AlternatingSmokeResult, expected: set[str]) -> None:
    artifact_index = cast(
        "dict[str, object]",
        json.loads(result.artifacts["artifact_index_json"].read_text(encoding="utf-8")),
    )
    indexed_artifacts = cast("list[dict[str, str]]", artifact_index["artifacts"])
    assert {item["name"] for item in indexed_artifacts} == expected - {"artifact_index_json"}
    assert all(item["type"] for item in indexed_artifacts)
    assert all(item["description"] for item in indexed_artifacts)


def _assert_summary_rows(result: AlternatingSmokeResult) -> None:
    with result.artifacts["alignment_summary_csv"].open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["level_factor"] for row in rows] == ["4", "2", "1"]
    assert rows[0]["residual_filter_kinds"] == "lowpass_gaussian"
    assert rows[1]["residual_filter_kinds"] == ("lowpass_gaussian|bandpass_difference_of_gaussians")
    assert rows[1]["skipped_level"] == "True"
    assert rows[-1]["executed_geometry_updates"] == "0"
    assert rows[0]["loss_nonincreasing"] == "True"
    assert rows[0]["gauge_stable"] == "True"


def _assert_saved_volume(result: AlternatingSmokeResult) -> None:
    saved_volume = cast("NDArray[np.float32]", np.load(result.artifacts["final_volume_npy"]))
    np.testing.assert_allclose(saved_volume, result.final_volume)


def _assert_manifest(result: AlternatingSmokeResult) -> None:
    manifest = cast(
        "dict[str, object]",
        json.loads(result.artifacts["run_manifest_json"].read_text(encoding="utf-8")),
    )
    assert manifest["align_mode"] == "auto"
    assert manifest["backend_actual"] == "jax_reference"
    assert manifest["status"] == "passed"


def _assert_gauge_reports(result: AlternatingSmokeResult) -> None:
    gauge_report = cast(
        "dict[str, object]",
        json.loads(result.artifacts["gauge_report_json"].read_text(encoding="utf-8")),
    )
    operations = cast("list[dict[str, object]]", gauge_report["operations"])
    assert {operation["target"] for operation in operations} >= {
        "setup.det_u_px",
        "setup.theta_offset_rad",
    }

    gauge_policy = cast(
        "dict[str, object]",
        json.loads(result.artifacts["gauge_policy_json"].read_text(encoding="utf-8")),
    )
    policy_operations = cast("list[dict[str, object]]", gauge_policy["operations"])
    assert {operation["name"] for operation in policy_operations} >= {
        "mean_dx_to_det_u",
        "mean_phi_to_theta_offset",
    }


def _assert_audit_reports(result: AlternatingSmokeResult) -> None:
    observability = cast(
        "dict[str, object]",
        json.loads(result.artifacts["observability_report_json"].read_text(encoding="utf-8")),
    )
    assert observability["status"] == "smoke_placeholder"

    failure_report = cast(
        "dict[str, object]",
        json.loads(result.artifacts["failure_report_json"].read_text(encoding="utf-8")),
    )
    assert failure_report["status"] == "passed"
    assert failure_report["failure"] is None


def _assert_residual_metrics(result: AlternatingSmokeResult) -> None:
    with result.artifacts["residual_metrics_csv"].open("r", newline="", encoding="utf-8") as fh:
        metric_rows = list(csv.DictReader(fh))
    assert [row["level_factor"] for row in metric_rows] == ["4", "2", "1"]
    assert metric_rows[0]["parameter_update_small"] == "True"


def test_alternating_solver_smoke_is_deterministic(tmp_path: Path) -> None:
    first = run_alternating_solver_smoke(tmp_path / "first")
    second = run_alternating_solver_smoke(tmp_path / "second")

    assert first.final_volume.shape == second.final_volume.shape
    np.testing.assert_allclose(first.final_volume, second.final_volume)
    np.testing.assert_allclose(first.final_geometry.pose.dx_px, second.final_geometry.pose.dx_px)
    assert first.verification == second.verification


def test_alternating_alignment_solver_runs_smoke_profile(tmp_path: Path) -> None:
    solver = AlternatingAlignmentSolver()

    result = solver.run_smoke(tmp_path)

    assert result.final_volume.shape == (32, 32, 32)
    assert result.artifacts["verification_json"].exists()


def test_alternating_smoke_records_non_default_profile(tmp_path: Path) -> None:
    result = run_alternating_solver_smoke(
        tmp_path,
        config=AlternatingSmokeConfig(schedule=reference_continuation_schedule("lightning")),
    )

    manifest = cast(
        "dict[str, object]",
        json.loads(result.artifacts["run_manifest_json"].read_text(encoding="utf-8")),
    )
    assert manifest["profile"] == "lightning"
    assert manifest["run_id"] == "lightning-deterministic"
    continuation = cast("dict[str, object]", manifest["continuation"])
    assert continuation["name"] == "lightning"
    config_text = result.artifacts["config_resolved_toml"].read_text(encoding="utf-8")
    assert 'profile = "lightning"' in config_text
    assert "level_factors = [4, 2, 1]" in config_text
