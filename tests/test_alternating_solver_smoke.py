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
    _assert_geometry_trace(result)
    _assert_plots_summary(result)
    _assert_schur_diagnostics(result)
    _assert_saved_volume(result)
    _assert_truth_artifacts(result)
    _assert_input_arrays(result)
    _assert_manifest(result)
    _assert_gauge_reports(result)
    _assert_audit_reports(result)
    _assert_preview_slice_artifacts(result)
    _assert_residual_map_artifacts(result)
    _assert_residual_metrics(result)
    _assert_recovery_tolerances(result)
    assert abs(float(np.mean(result.final_geometry.pose.dx_px))) < 1.0e-12
    assert abs(float(np.mean(result.final_geometry.pose.phi_residual_rad))) < 1.0e-12


def _assert_smoke_result_shape_and_exit(result: AlternatingSmokeResult) -> None:
    assert result.final_volume.shape == (32, 32, 32)
    assert result.verification["size"] == 32
    assert result.verification["seed"] == 17
    assert result.verification["coarse_verified"] is True
    assert result.verification["level1_geometry_skipped"] is True
    assert result.verification["skipped_levels"] == [2]
    assert result.verification["status"] == "passed"
    _assert_verification_contract(result)
    _assert_level_exit_contract(result)


def _assert_verification_contract(result: AlternatingSmokeResult) -> None:
    assert result.verification["geometry_update_volume_source"] == "stopped_reconstruction"
    assert result.verification["synthetic_dataset"] is None
    verification_summary = cast("dict[str, bool]", result.verification["summary"])
    assert verification_summary["final_reconstruction_valid"] is True
    assert verification_summary["gauge_constraints_satisfied"] is True
    assert verification_summary["backend_provenance_complete"] is True
    assert verification_summary["weak_dofs_handled"] is True
    verification_metrics = cast("dict[str, float]", result.verification["metrics"])
    assert verification_metrics["residual_before"] == result.levels[0].loss_before
    assert verification_metrics["residual_after"] == result.verification["final_loss"]
    assert verification_summary["projection_residual_improved"] is True
    assert verification_metrics["residual_after"] < verification_metrics["residual_before"]
    assert verification_summary["projection_residual_improved"] == (
        verification_metrics["residual_after"] <= verification_metrics["residual_before"] + 1.0e-5
    )
    assert verification_metrics["volume_nmse"] >= 0.0
    escalation = cast("dict[str, bool | str]", result.verification["escalation"])
    assert escalation["level_1_geometry_run"] is False
    assert escalation["reason"] == "level_2_verification_passed"
    thresholds = cast("dict[str, float]", result.verification["thresholds"])
    assert thresholds["gauge_stability_tolerance"] == 1.0e-10
    assert thresholds["parameter_update_tolerance"] == 2.0
    recovery = cast("dict[str, float | bool]", result.verification["geometry_recovery"])
    assert recovery["passed"] is True
    assert cast("float", recovery["theta_realized_rmse_rad"]) <= cast(
        "float", recovery["theta_realized_rmse_rad_limit"]
    )
    assert cast("float", recovery["det_u_realized_rmse_px"]) <= cast(
        "float", recovery["det_u_realized_rmse_px_limit"]
    )
    assert cast("float", recovery["det_v_realized_rmse_px"]) <= cast(
        "float", recovery["det_v_realized_rmse_px_limit"]
    )
    mean_dx_abs = cast("float", recovery["mean_dx_abs_px"])
    mean_phi_abs = cast("float", recovery["mean_phi_abs_rad"])
    assert mean_dx_abs <= 1.0e-10
    assert mean_phi_abs <= 1.0e-10
    assert recovery["mean_dx_abs_px_limit"] == 1.0e-10
    volume_recovery = cast("dict[str, float | bool]", result.verification["volume_recovery"])
    assert volume_recovery["passed"] is True
    assert cast("float", volume_recovery["nmse"]) <= cast("float", volume_recovery["nmse_limit"])
    assert cast("float", volume_recovery["rmse"]) >= 0.0


def _assert_level_exit_contract(result: AlternatingSmokeResult) -> None:
    assert result.levels[0].geometry_updates == 8
    assert result.levels[0].executed_geometry_updates == 8
    assert result.levels[0].residual_filter_kinds == ("lowpass_gaussian",)
    assert result.levels[0].loss_nonincreasing
    assert result.levels[0].finite_loss
    assert result.levels[0].residual_sigma_estimated > 0.0
    assert result.levels[0].residual_sigma_effective == 1.0
    assert result.levels[0].gauge_stable
    assert result.levels[0].parameter_update_small
    assert result.levels[0].parameter_update_norm > 0.0
    assert result.levels[0].schur_diagnostics is not None
    assert result.levels[0].schur_diagnostics.accepted is True
    assert result.levels[0].schur_diagnostics.actual_reduction > 0.0
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
        "geometry_corrupted_json",
        "gauge_policy_json",
        "gauge_report_json",
        "geometry_final_json",
        "geometry_initial_json",
        "geometry_trace_csv",
        "geometry_true_json",
        "ground_truth_volume_npy",
        "input_summary_json",
        "mask_summary_json",
        "observability_report_json",
        "observed_projections_npy",
        "plots_summary_json",
        "pose_decomposition_csv",
        "pose_params_csv",
        "preview_error_slice_npy",
        "preview_final_slice_npy",
        "preview_summary_json",
        "preview_truth_slice_npy",
        "projection_mask_npy",
        "projection_stats_json",
        "recovery_tolerances_json",
        "residual_map_raw_npy",
        "residual_map_summary_json",
        "residual_metrics_csv",
        "run_manifest_json",
        "schur_diagnostics_json",
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
    indexed_paths = {item["name"]: item["path"] for item in indexed_artifacts}
    assert indexed_paths["preview_error_slice_npy"] == "preview_slices/central_z_error.npy"
    assert indexed_paths["preview_final_slice_npy"] == "preview_slices/central_z_final.npy"
    assert indexed_paths["preview_summary_json"] == "preview_slices/summary.json"
    assert indexed_paths["preview_truth_slice_npy"] == "preview_slices/central_z_truth.npy"
    assert indexed_paths["plots_summary_json"] == "plots/summary.json"
    assert indexed_paths["residual_map_raw_npy"] == "residual_maps/final_raw_residual.npy"
    assert indexed_paths["residual_map_summary_json"] == "residual_maps/summary.json"


def _assert_summary_rows(result: AlternatingSmokeResult) -> None:
    with result.artifacts["alignment_summary_csv"].open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["level_factor"] for row in rows] == ["4", "2", "1"]
    assert rows[0]["residual_filter_kinds"] == "lowpass_gaussian"
    assert rows[1]["residual_filter_kinds"] == ("lowpass_gaussian|bandpass_difference_of_gaussians")
    assert rows[1]["skipped_level"] == "True"
    assert rows[-1]["executed_geometry_updates"] == "0"
    assert rows[0]["loss_nonincreasing"] == "True"
    assert float(rows[0]["residual_sigma_estimated"]) > 0.0
    assert rows[0]["residual_sigma_effective"] == "1.0"
    assert rows[0]["gauge_stable"] == "True"


def _assert_geometry_trace(result: AlternatingSmokeResult) -> None:
    with result.artifacts["geometry_trace_csv"].open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["level_factor"] for row in rows] == ["4", "2", "1"]
    assert rows[0]["geometry_updates_requested"] == "8"
    assert rows[0]["geometry_updates_executed"] == "8"
    assert float(rows[0]["parameter_update_norm"]) > 0.0
    assert rows[0]["verified"] == "True"
    assert rows[0]["schur_accepted"] == "True"
    assert float(rows[0]["schur_actual_reduction"]) > 0.0
    assert float(rows[0]["schur_dense_step_difference_norm"]) < 5.0e-3
    assert rows[1]["skipped_level"] == "True"
    assert rows[2]["skipped_geometry"] == "True"
    assert rows[2]["early_exit_reason"] == "coarse_verification_passed"


def _assert_plots_summary(result: AlternatingSmokeResult) -> None:
    payload = cast(
        "dict[str, object]",
        json.loads(result.artifacts["plots_summary_json"].read_text(encoding="utf-8")),
    )
    assert payload["schema"] == "tomojax.plots_summary.v1"
    assert payload["rendered"] is False
    fista_loss = cast("list[dict[str, object]]", payload["fista_loss"])
    geometry_loss = cast("list[dict[str, object]]", payload["geometry_loss"])
    assert fista_loss[0]["iteration"] == 0
    assert [row["level_factor"] for row in geometry_loss] == [4, 2, 1]


def _assert_schur_diagnostics(result: AlternatingSmokeResult) -> None:
    payload = cast(
        "dict[str, object]",
        json.loads(result.artifacts["schur_diagnostics_json"].read_text(encoding="utf-8")),
    )
    assert payload["schema"] == "tomojax.schur_diagnostics.v1"
    assert payload["status"] == "passed"
    assert payload["solver"] == "joint_schur_lm_reference"
    assert payload["geometry_update_volume_source"] == "stopped_reconstruction"
    assert payload["active_setup_parameters"] == [
        "theta_offset_rad",
        "det_u_px",
        "det_v_px",
    ]
    diagnostics = cast("dict[str, object]", payload["diagnostics"])
    assert diagnostics["accepted"] is True
    assert float(cast("float", diagnostics["actual_reduction"])) > 0.0
    assert len(cast("list[float]", diagnostics["current_loss_by_view"])) == 4


def _assert_saved_volume(result: AlternatingSmokeResult) -> None:
    saved_volume = cast("NDArray[np.float32]", np.load(result.artifacts["final_volume_npy"]))
    np.testing.assert_allclose(saved_volume, result.final_volume)
    assert float(np.max(saved_volume)) > 0.01


def _assert_truth_artifacts(result: AlternatingSmokeResult) -> None:
    truth = cast("NDArray[np.float32]", np.load(result.artifacts["ground_truth_volume_npy"]))
    assert truth.shape == (32, 32, 32)
    assert truth.dtype == np.float32

    true_geometry = cast(
        "dict[str, object]",
        json.loads(result.artifacts["geometry_true_json"].read_text(encoding="utf-8")),
    )
    corrupted_geometry = cast(
        "dict[str, object]",
        json.loads(result.artifacts["geometry_corrupted_json"].read_text(encoding="utf-8")),
    )
    true_setup = cast("dict[str, dict[str, object]]", true_geometry["setup"])
    corrupted_setup = cast("dict[str, dict[str, object]]", corrupted_geometry["setup"])
    assert true_setup["det_u_px"]["value"] == 0.045
    assert true_setup["det_u_px"]["value"] != corrupted_setup["det_u_px"]["value"]
    assert true_setup["theta_offset_rad"]["value"] != corrupted_setup["theta_offset_rad"]["value"]


def _assert_input_arrays(result: AlternatingSmokeResult) -> None:
    projections = cast("NDArray[np.float32]", np.load(result.artifacts["observed_projections_npy"]))
    mask = cast("NDArray[np.bool_]", np.load(result.artifacts["projection_mask_npy"]))

    assert projections.shape == (4, 32, 32)
    assert projections.dtype == np.float32
    assert mask.shape == projections.shape
    assert mask.dtype == np.bool_
    assert bool(np.all(mask))


def _assert_manifest(result: AlternatingSmokeResult) -> None:
    manifest = cast(
        "dict[str, object]",
        json.loads(result.artifacts["run_manifest_json"].read_text(encoding="utf-8")),
    )
    assert manifest["align_mode"] == "auto"
    assert isinstance(manifest["tomojax_version"], str)
    assert isinstance(manifest["git_commit"], str)
    assert manifest["started_at"] == "deterministic-smoke"
    assert manifest["finished_at"] == "deterministic-smoke"
    assert manifest["geometry_model"] == "parallel_tomography_reference"
    assert manifest["geometry_update_volume_source"] == "stopped_reconstruction"
    assert "synthetic128_benchmark" not in cast("dict[str, object]", manifest["dataset"])
    assert manifest["backend_requested"] == "jax_reference"
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
    assert observability["status"] == "smoke_not_evaluated"
    dofs = cast("dict[str, dict[str, dict[str, object]]]", observability["dofs"])
    assert dofs["setup"]["det_u_px"]["status"] == "weak_not_evaluated"
    assert dofs["setup"]["det_v_px"]["status"] == "frozen"
    assert dofs["pose"]["dx_px"]["status"] == "gauge_canonicalised"
    weak_modes = cast("list[dict[str, object]]", observability["weak_modes"])
    assert weak_modes[0]["name"] == "smoke_curvature_uncomputed"
    assert observability["handled_frozen_dofs"] == ["det_v_px", "theta_scale"]

    failure_report = cast(
        "dict[str, object]",
        json.loads(result.artifacts["failure_report_json"].read_text(encoding="utf-8")),
    )
    assert failure_report["status"] == "passed"
    assert failure_report["failure"] is None
    assert "nan_or_inf" in cast("list[str]", failure_report["failure_classes"])
    gates = cast("list[dict[str, object]]", failure_report["gates"])
    gates_by_name = {str(gate["name"]): gate for gate in gates}
    assert gates_by_name["finite_outputs"]["passed"] is True
    assert gates_by_name["gauge_stability"]["passed"] is True
    assert gates_by_name["optimiser_health"]["passed"] is True
    assert gates_by_name["backend_provenance"]["passed"] is True
    assert gates_by_name["projection_residual_improvement"]["severity"] == "warning"
    warnings = cast("list[dict[str, object]]", failure_report["warnings"])
    assert gates_by_name["projection_residual_improvement"]["passed"] is True
    assert warnings == []

    backend_report = cast(
        "dict[str, object]",
        json.loads(result.artifacts["backend_report_json"].read_text(encoding="utf-8")),
    )
    assert backend_report["actual_projector"] == "jax_reference"
    assert backend_report["actual_backprojector"] == "jax_reference"
    assert backend_report["actual_geometry_reductions"] == "jax_reference"
    assert backend_report["canonical_detector_grid"] is True
    assert backend_report["calibrated_detector_grid"] is False
    assert backend_report["pallas_eligible"] is False
    assert backend_report["fallbacks"] == []
    agreement_tests = cast("list[dict[str, object]]", backend_report["agreement_tests"])
    assert agreement_tests == [
        {
            "component": "residual",
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "status": "reference_baseline",
        }
    ]


def _assert_preview_slice_artifacts(result: AlternatingSmokeResult) -> None:
    truth = cast("NDArray[np.float32]", np.load(result.artifacts["preview_truth_slice_npy"]))
    final = cast("NDArray[np.float32]", np.load(result.artifacts["preview_final_slice_npy"]))
    error = cast("NDArray[np.float32]", np.load(result.artifacts["preview_error_slice_npy"]))
    summary = cast(
        "dict[str, object]",
        json.loads(result.artifacts["preview_summary_json"].read_text(encoding="utf-8")),
    )

    assert truth.shape == (32, 32)
    assert final.shape == truth.shape
    assert error.shape == truth.shape
    assert truth.dtype == np.float32
    np.testing.assert_allclose(error, final - truth, atol=1.0e-37)
    assert summary["schema"] == "tomojax.preview_slices.v1"
    assert summary["axis"] == "z"
    assert summary["index"] == 16
    assert summary["shape"] == [32, 32]
    assert summary["dtype"] == "float32"
    assert float(cast("float", summary["error_rmse"])) >= 0.0


def _assert_residual_map_artifacts(result: AlternatingSmokeResult) -> None:
    residual_map = cast("NDArray[np.float32]", np.load(result.artifacts["residual_map_raw_npy"]))
    summary = cast(
        "dict[str, object]",
        json.loads(result.artifacts["residual_map_summary_json"].read_text(encoding="utf-8")),
    )

    assert residual_map.shape == (4, 32, 32)
    assert residual_map.dtype == np.float32
    assert summary["schema"] == "tomojax.residual_map_summary.v1"
    assert summary["shape"] == [4, 32, 32]
    assert summary["dtype"] == "float32"
    assert summary["valid_pixel_fraction"] == 1.0
    assert float(cast("float", summary["rmse"])) >= 0.0


def _assert_residual_metrics(result: AlternatingSmokeResult) -> None:
    with result.artifacts["residual_metrics_csv"].open("r", newline="", encoding="utf-8") as fh:
        metric_rows = list(csv.DictReader(fh))
    level_rows = [row for row in metric_rows if row["row_type"] == "level_summary"]
    view_rows = [row for row in metric_rows if row["row_type"] == "view_residual"]
    assert [row["level_factor"] for row in level_rows] == ["4", "2", "1"]
    assert level_rows[0]["parameter_update_small"] == "True"
    assert [int(row["view_index"]) for row in view_rows] == [0, 1, 2, 3]
    assert all(float(row["rmse"]) >= 0.0 for row in view_rows)
    assert all(row["valid_pixel_fraction"] == "1.0" for row in view_rows)
    assert all(row["raw_rmse"] == row["rmse"] for row in view_rows)


def _assert_recovery_tolerances(result: AlternatingSmokeResult) -> None:
    payload = cast(
        "dict[str, object]",
        json.loads(result.artifacts["recovery_tolerances_json"].read_text(encoding="utf-8")),
    )
    geometry = cast("dict[str, float]", payload["geometry"])
    volume = cast("dict[str, float]", payload["volume"])
    verification = cast("dict[str, bool]", payload["verification"])
    assert geometry["theta_realized_rmse_rad_lt"] == 8.5e-2
    assert geometry["mean_gauge_abs_lt"] == 1.0e-10
    assert volume["nmse_lt"] == 10.0
    assert verification["gauge_stable"] is True


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
    assert 'geometry_update_volume_source = "stopped_reconstruction"' in config_text
    assert "level_factors = [4, 2, 1]" in config_text
