# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING, cast

import jax.numpy as jnp
import numpy as np

# check-public-imports: allow-private
from tomojax.align._alternating_artifacts import (
    _bad_view_detection_payload,
    _bad_views_flagged_evaluation,
)

# check-public-imports: allow-private
from tomojax.align._alternating_inputs import build_smoke_inputs
from tomojax.align.api import (
    AlternatingAlignmentSolver,
    AlternatingSmokeConfig,
    reference_continuation_schedule,
    run_alternating_solver_smoke,
)
from tomojax.datasets import (
    generate_synthetic_dataset,
    load_synthetic_dataset_sidecars,
)
from tomojax.geometry import GeometryState
from tomojax.verify import residual_structure_summary

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
    _assert_mask_provenance(result)
    _assert_fista_diagnostic_artifacts(result)
    _assert_detu_landscape_artifacts(result)
    _assert_preview_slice_artifacts(result)
    _assert_residual_map_artifacts(result)
    _assert_residual_metrics(result)
    _assert_recovery_tolerances(result)
    assert abs(float(np.mean(result.final_geometry.pose.dx_px))) < 1.0e-12
    assert abs(float(np.mean(result.final_geometry.pose.phi_residual_rad))) < 1.0e-12


def test_otsu_loss_splits_valid_and_alignment_masks(tmp_path: Path) -> None:
    dataset_paths = generate_synthetic_dataset(
        "rich_phantom94_det_u_only_v1_parity",
        tmp_path / "datasets",
        size=32,
        clean=True,
        views=4,
        supported_only=True,
    )

    inputs = build_smoke_inputs(
        AlternatingSmokeConfig(
            size=32,
            n_views=4,
            projection_loss_mode="otsu_l2",
            synthetic_dataset_name="rich_phantom94_det_u_only_v1_parity",
            synthetic_dataset_artifact_dir=dataset_paths.dataset_dir,
        )
    )

    valid = np.asarray(inputs.projection_valid_mask, dtype=bool)
    alignment = np.asarray(inputs.alignment_loss_mask, dtype=bool)
    assert valid.shape == alignment.shape
    assert np.all(valid)
    assert np.all(alignment <= valid)
    assert int(np.count_nonzero(alignment)) < int(np.count_nonzero(valid))


def _assert_smoke_result_shape_and_exit(result: AlternatingSmokeResult) -> None:
    assert result.final_volume.shape == (32, 32, 32)
    assert result.verification["size"] == 32
    assert result.verification["seed"] == 17
    assert isinstance(result.verification["coarse_verified"], bool)
    assert isinstance(result.verification["level1_geometry_skipped"], bool)
    assert isinstance(result.verification["skipped_levels"], list)
    assert result.verification["status"] in {"passed", "failed"}
    _assert_verification_contract(result)
    _assert_level_exit_contract(result)


def _assert_verification_contract(result: AlternatingSmokeResult) -> None:
    assert result.verification["geometry_update_volume_source"] == "stopped_reconstruction"
    assert result.verification["fit_gain_offset_nuisance"] is False
    assert result.verification["fit_background_nuisance"] is False
    assert result.verification["synthetic_dataset"] is None
    verification_summary = cast("dict[str, bool]", result.verification["summary"])
    assert isinstance(verification_summary["final_reconstruction_valid"], bool)
    assert verification_summary["gauge_constraints_satisfied"] is True
    assert verification_summary["backend_provenance_complete"] is True
    assert verification_summary["weak_dofs_handled"] is True
    verification_metrics = cast("dict[str, float]", result.verification["metrics"])
    assert verification_metrics["residual_before"] == result.levels[0].loss_before
    assert verification_metrics["residual_after"] == result.verification["final_loss"]
    assert isinstance(verification_summary["projection_residual_improved"], bool)
    assert verification_summary["projection_residual_improved"] == (
        verification_metrics["residual_after"] <= verification_metrics["residual_before"] + 1.0e-5
    )
    assert verification_metrics["volume_nmse"] >= 0.0
    runtime = cast("dict[str, float]", result.verification["runtime"])
    assert runtime["total_wall_seconds"] >= 0.0
    if runtime["time_to_verified_geometry_seconds"] is not None:
        assert runtime["total_wall_seconds"] >= runtime["time_to_verified_geometry_seconds"]
    escalation = cast("dict[str, bool | str]", result.verification["escalation"])
    assert isinstance(escalation["level_1_geometry_run"], bool)
    assert isinstance(escalation["reason"], str)
    thresholds = cast("dict[str, float]", result.verification["thresholds"])
    assert thresholds["gauge_stability_tolerance"] == 1.0e-10
    assert thresholds["parameter_update_tolerance"] == 2.0
    assert thresholds["heldout_residual_tolerance"] == 1.0e-5
    recovery = cast("dict[str, float | bool]", result.verification["geometry_recovery"])
    assert cast("float", recovery["initial_theta_realized_rmse_rad"]) > 0.0
    assert cast("float", recovery["initial_det_u_realized_rmse_px"]) > 0.0
    assert cast("float", recovery["initial_det_v_realized_rmse_px"]) > 0.0
    assert isinstance(recovery["passed"], bool)
    assert isinstance(recovery["theta_realized_rmse_rad"], float)
    assert isinstance(recovery["det_u_realized_rmse_px"], float)
    assert isinstance(recovery["det_v_realized_rmse_px"], float)
    mean_dx_abs = cast("float", recovery["mean_dx_abs_px"])
    mean_phi_abs = cast("float", recovery["mean_phi_abs_rad"])
    assert mean_dx_abs <= 1.0e-10
    assert mean_phi_abs <= 1.0e-10
    assert recovery["mean_dx_abs_px_limit"] == 1.0e-10
    volume_recovery = cast("dict[str, float | bool]", result.verification["volume_recovery"])
    assert isinstance(volume_recovery["passed"], bool)
    assert cast("float", volume_recovery["nmse"]) >= 0.0
    assert cast("float", volume_recovery["rmse"]) >= 0.0


def _assert_level_exit_contract(result: AlternatingSmokeResult) -> None:
    assert result.levels[0].geometry_updates == 8
    assert result.levels[0].executed_geometry_updates == 8
    assert result.levels[0].residual_filter_kinds == ("lowpass_gaussian",)
    assert result.levels[0].loss_nonincreasing
    assert result.levels[0].finite_loss
    assert result.levels[0].residual_sigma_estimated > 0.0
    assert result.levels[0].residual_sigma_effective == 1.0
    assert result.levels[0].prior_strength == 1.0e-3
    assert result.levels[0].heldout_residual_before is not None
    assert result.levels[0].heldout_residual_after is not None
    assert isinstance(result.levels[0].heldout_residual_passed, bool)
    assert result.levels[0].gauge_stable
    assert result.levels[0].parameter_update_small
    assert result.levels[0].parameter_update_norm >= 0.0
    assert result.levels[0].schur_diagnostics is not None
    assert isinstance(result.levels[0].schur_diagnostics.accepted, bool)
    assert np.isfinite(result.levels[0].schur_diagnostics.actual_reduction)
    assert isinstance(result.levels[1].skipped_level, bool)
    assert result.levels[1].residual_filter_kinds == (
        "lowpass_gaussian",
        "bandpass_difference_of_gaussians",
    )
    assert isinstance(result.levels[-1].skipped_geometry, bool)
    assert result.levels[-1].residual_filter_kinds == ("raw",)
    assert result.levels[-1].geometry_updates == 1
    assert result.levels[-1].executed_geometry_updates in {0, 1}


def _expected_artifacts() -> set[str]:
    return {
        "alignment_summary_csv",
        "adjoint_checks_json",
        "artifact_index_json",
        "backend_report_json",
        "config_resolved_toml",
        "detu_curve_inputs_json",
        "detu_curve_summary_json",
        "detu_gradient_curves_png",
        "detu_loss_curves_csv",
        "detu_loss_curves_png",
        "failure_report_json",
        "fista_gradient_checks_json",
        "final_volume_npy",
        "fista_trace_csv",
        "fista_trace_recomputed_csv",
        "geometry_corrupted_json",
        "geometry_jvp_vjp_checks_json",
        "gauge_policy_json",
        "gauge_report_json",
        "geometry_final_json",
        "geometry_initial_json",
        "geometry_trace_csv",
        "geometry_true_json",
        "ground_truth_volume_npy",
        "input_summary_json",
        "loss_normalisation_report_json",
        "mask_summary_json",
        "mask_provenance_json",
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


def _assert_mask_provenance(result: AlternatingSmokeResult) -> None:
    payload = cast(
        "dict[str, object]",
        json.loads(result.artifacts["mask_provenance_json"].read_text(encoding="utf-8")),
    )
    assert payload["schema"] == "tomojax.mask_provenance.v1"
    entries = cast("list[dict[str, object]]", payload["entries"])
    assert entries
    fista_entries = [
        entry for entry in entries if entry["operation"] == "fista_reconstruct_reference"
    ]
    assert fista_entries
    assert {entry["mask_role"] for entry in fista_entries} == {"projection_valid_mask"}
    assert all(entry["includes_otsu"] is False for entry in fista_entries)
    assert all(entry["includes_train_gating"] is False for entry in fista_entries)
    alignment_entries = [
        entry
        for entry in entries
        if entry["operation"] in {"joint_schur_geometry_update", "projection_loss"}
    ]
    assert alignment_entries
    assert any(str(entry["mask_role"]).startswith("alignment") for entry in alignment_entries)


def _assert_fista_diagnostic_artifacts(result: AlternatingSmokeResult) -> None:
    gradient = cast(
        "dict[str, object]",
        json.loads(result.artifacts["fista_gradient_checks_json"].read_text(encoding="utf-8")),
    )
    assert gradient["schema"] == "tomojax.fista_gradient_checks.v1"
    assert gradient["status"] == "passed"
    adjoint = cast(
        "dict[str, object]",
        json.loads(result.artifacts["adjoint_checks_json"].read_text(encoding="utf-8")),
    )
    assert adjoint["status"] == "passed"
    jvp_vjp = cast(
        "dict[str, object]",
        json.loads(result.artifacts["geometry_jvp_vjp_checks_json"].read_text(encoding="utf-8")),
    )
    assert jvp_vjp["status"] == "passed"
    normalisation = cast(
        "dict[str, object]",
        json.loads(
            result.artifacts["loss_normalisation_report_json"].read_text(encoding="utf-8")
        ),
    )
    assert normalisation["current_contract"] == "full_projection_array_size"
    with result.artifacts["fista_trace_recomputed_csv"].open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["recomputed_loss_point"] for row in rows} == {"returned_final_volume"}


def _assert_detu_landscape_artifacts(result: AlternatingSmokeResult) -> None:
    summary = cast(
        "dict[str, object]",
        json.loads(result.artifacts["detu_curve_summary_json"].read_text(encoding="utf-8")),
    )
    assert summary["schema"] == "tomojax.detu_curve_summary.v1"
    curves = cast("list[dict[str, object]]", summary["curves"])
    assert {curve["volume_source"] for curve in curves} == {
        "final_stopped_volume",
        "true_volume",
        "true_geometry_reconstructed_volume",
        "zero_initial_volume",
    }
    inputs = cast(
        "dict[str, object]",
        json.loads(result.artifacts["detu_curve_inputs_json"].read_text(encoding="utf-8")),
    )
    assert inputs["purpose"] == "diagnostic_fixed_volume_landscape_not_production_center_search"
    with result.artifacts["detu_loss_curves_csv"].open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["volume_source"] for row in rows} == {
        "final_stopped_volume",
        "true_geometry_reconstructed_volume",
        "true_volume",
        "zero_initial_volume",
    }
    assert result.artifacts["detu_loss_curves_png"].stat().st_size > 0
    assert result.artifacts["detu_gradient_curves_png"].stat().st_size > 0


def _assert_summary_rows(result: AlternatingSmokeResult) -> None:
    with result.artifacts["alignment_summary_csv"].open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["level_factor"] for row in rows] == ["4", "2", "1"]
    assert rows[0]["residual_filter_kinds"] == "lowpass_gaussian"
    assert rows[1]["residual_filter_kinds"] == ("lowpass_gaussian|bandpass_difference_of_gaussians")
    assert rows[1]["skipped_level"] in {"True", "False"}
    assert rows[-1]["executed_geometry_updates"] in {"0", "1"}
    assert rows[0]["loss_nonincreasing"] in {"True", "False"}
    assert float(rows[0]["residual_sigma_estimated"]) > 0.0
    assert rows[0]["residual_sigma_effective"] == "1.0"
    assert rows[0]["prior_strength"] == "0.001"
    assert float(rows[0]["heldout_residual_before"]) > 0.0
    assert rows[0]["heldout_residual_passed"] in {"True", "False"}
    assert rows[0]["gauge_stable"] in {"True", "False"}


def _assert_geometry_trace(result: AlternatingSmokeResult) -> None:
    with result.artifacts["geometry_trace_csv"].open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["level_factor"] for row in rows] == ["4", "2", "1"]
    assert rows[0]["geometry_updates_requested"] == "8"
    assert rows[0]["geometry_updates_executed"] == "8"
    assert float(rows[0]["parameter_update_norm"]) >= 0.0
    assert rows[0]["prior_strength"] == "0.001"
    assert rows[0]["verified"] in {"True", "False"}
    assert rows[0]["heldout_residual_passed"] in {"True", "False"}
    assert rows[0]["schur_accepted"] in {"True", "False"}
    assert np.isfinite(float(rows[0]["schur_actual_reduction"]))
    dense_step_difference = float(rows[0]["schur_dense_step_difference_norm"])
    assert np.isfinite(dense_step_difference) or np.isnan(dense_step_difference)
    assert rows[1]["skipped_level"] in {"True", "False"}
    assert rows[2]["skipped_geometry"] in {"True", "False"}
    assert rows[2]["early_exit_reason"] in {
        "",
        "coarse_verification_passed",
        "geometry_updates_executed",
    }


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
    assert set(cast("list[str]", payload["active_setup_parameters"])) <= {
        "theta_offset_rad",
        "det_u_px",
        "det_v_px",
    }
    diagnostics = cast("dict[str, object]", payload["diagnostics"])
    assert isinstance(diagnostics["accepted"], bool)
    assert float(cast("float", diagnostics["parameter_prior_strength"])) > 0.0
    assert np.isfinite(float(cast("float", diagnostics["actual_reduction"])))
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
    assert manifest["geometry_model"] == "parallel_tomography_core_trilinear_ray"
    assert manifest["geometry_update_volume_source"] == "stopped_reconstruction"
    assert manifest["fit_gain_offset_nuisance"] is False
    assert manifest["fit_background_nuisance"] is False
    assert "synthetic128_benchmark" not in cast("dict[str, object]", manifest["dataset"])
    assert manifest["backend_requested"] == "core_trilinear_ray"
    assert manifest["backend_actual"] == "core_trilinear_ray"
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
    assert observability["status"] == "evaluated"
    assert isinstance(observability["schur_condition_number"], float)
    assert isinstance(observability["schur_min_eigenvalue"], float)
    schur_eigenvalues = cast("list[float]", observability["schur_eigenvalues"])
    assert 1 <= len(schur_eigenvalues) <= 3
    weak_dof_policy = cast("dict[str, object]", observability["weak_dof_policy"])
    assert weak_dof_policy["mode"] == "report_only"
    thresholds = cast("dict[str, object]", weak_dof_policy["thresholds"])
    assert thresholds["correlation_abs_max_ceiling"] == 0.98
    decisions = cast("dict[str, dict[str, object]]", weak_dof_policy["decisions"])
    assert "det_v_px" in decisions
    assert decisions["theta_scale"]["decision"] == "keep_frozen"
    det_v_evidence = cast("dict[str, object]", decisions["det_v_px"]["evidence"])
    assert isinstance(det_v_evidence["curvature_passed"], bool)
    assert isinstance(det_v_evidence["correlation_passed"], bool)
    if det_v_evidence.get("correlation") is not None:
        det_v_correlation = cast("dict[str, object]", det_v_evidence["correlation"])
        assert det_v_correlation["kind"] == "setup_correlation_matrix"
        assert isinstance(det_v_correlation["parameter_index"], int)
        assert np.isfinite(float(cast("float", det_v_correlation["max_abs_correlation"])))
    assert isinstance(det_v_evidence["accepted_step_passed"], bool)
    assert isinstance(det_v_evidence["validation_improvement_passed"], bool)
    if det_v_evidence.get("validation_improvement") is not None:
        det_v_validation = cast("dict[str, object]", det_v_evidence["validation_improvement"])
        assert det_v_validation["kind"] == "schur_actual_reduction"
        assert np.isfinite(float(cast("float", det_v_validation["actual_reduction"])))
        assert isinstance(det_v_validation["accepted"], bool)
    det_v_missing = cast("list[str]", det_v_evidence["missing_evidence"])
    assert all(isinstance(item, str) for item in det_v_missing)
    _assert_theta_scale_missing_evidence(decisions)
    dofs = cast("dict[str, dict[str, dict[str, object]]]", observability["dofs"])
    _assert_observability_dofs(dofs)
    weak_modes = cast("list[dict[str, object]]", observability["weak_modes"])
    assert weak_modes[0]["name"] == "schur_weak_modes"
    handled_frozen_dofs = cast("list[str]", observability["handled_frozen_dofs"])
    assert "theta_scale" in handled_frozen_dofs
    assert set(handled_frozen_dofs) <= {"det_v_px", "theta_scale"}

    _assert_failure_report(result)
    _assert_backend_report(result)


def _assert_observability_dofs(dofs: dict[str, dict[str, dict[str, object]]]) -> None:
    assert dofs["setup"]["det_u_px"]["status"] == "evaluated"
    assert dofs["setup"]["det_u_px"]["observable"] is True
    assert dofs["setup"]["theta_offset_rad"]["status"] == "evaluated"
    assert dofs["setup"]["theta_offset_rad"]["active"] is True
    assert dofs["setup"]["theta_offset_rad"]["observable"] is True
    assert dofs["setup"]["detector_roll_rad"]["status"] == "frozen"
    assert dofs["setup"]["detector_roll_rad"]["active"] is False
    assert dofs["setup"]["det_v_px"]["status"] in {"evaluated", "frozen"}
    assert isinstance(dofs["setup"]["det_v_px"]["observable"], bool)
    assert dofs["setup"]["theta_scale"]["reason"] == (
        "theta_scale is frozen until identifiable scale policy exists"
    )
    assert dofs["pose"]["alpha_rad"]["active"] is False
    assert dofs["pose"]["alpha_rad"]["status"] == "frozen"
    assert dofs["pose"]["dx_px"]["status"] == "gauge_canonicalised"
    assert dofs["pose"]["dz_px"]["status"] == "gauge_canonicalised"


def _assert_theta_scale_missing_evidence(
    decisions: dict[str, dict[str, object]],
) -> None:
    theta_scale_evidence = cast("dict[str, object]", decisions["theta_scale"]["evidence"])
    assert theta_scale_evidence["curvature"] is None
    assert theta_scale_evidence["curvature_passed"] is False
    assert theta_scale_evidence["correlation"] is None
    assert theta_scale_evidence["correlation_passed"] is False
    assert theta_scale_evidence["accepted_step_passed"] is False
    assert theta_scale_evidence["validation_improvement"] is None
    assert theta_scale_evidence["validation_improvement_passed"] is False
    theta_scale_missing = cast("list[str]", theta_scale_evidence["missing_evidence"])
    assert "curvature" in theta_scale_missing
    assert "correlation" in theta_scale_missing
    assert "accepted_step" in theta_scale_missing
    assert "validation_improvement_gate_not_available_in_smoke" in theta_scale_missing


def _assert_failure_report(result: AlternatingSmokeResult) -> None:
    failure_report = cast(
        "dict[str, object]",
        json.loads(result.artifacts["failure_report_json"].read_text(encoding="utf-8")),
    )
    assert failure_report["status"] in {"passed", "warning"}
    assert failure_report["failure"] is None or isinstance(failure_report["failure"], dict)
    assert "nan_or_inf" in cast("list[str]", failure_report["failure_classes"])
    gates = cast("list[dict[str, object]]", failure_report["gates"])
    gates_by_name = {str(gate["name"]): gate for gate in gates}
    assert gates_by_name["finite_outputs"]["passed"] is True
    assert gates_by_name["gauge_stability"]["passed"] is True
    assert gates_by_name["optimiser_health"]["passed"] is True
    assert gates_by_name["backend_provenance"]["passed"] is True
    assert gates_by_name["projection_residual_improvement"]["severity"] == "warning"
    assert gates_by_name["nuisance_residual_structure"]["severity"] == "warning"
    assert isinstance(gates_by_name["nuisance_residual_structure"]["passed"], bool)
    assert gates_by_name["synthetic_sidecar_consistency"]["severity"] == "warning"
    assert gates_by_name["synthetic_sidecar_consistency"]["passed"] is True
    assert gates_by_name["synthetic_sidecar_consistency"]["evidence"] == (
        "no synthetic sidecar dataset requested"
    )
    warnings = cast("list[dict[str, object]]", failure_report["warnings"])
    assert isinstance(gates_by_name["projection_residual_improvement"]["passed"], bool)
    assert isinstance(warnings, list)


def _assert_backend_report(result: AlternatingSmokeResult) -> None:
    backend_report = cast(
        "dict[str, object]",
        json.loads(result.artifacts["backend_report_json"].read_text(encoding="utf-8")),
    )
    assert backend_report["actual_projector"] == "core_trilinear_ray"
    assert backend_report["actual_backprojector"] == "core_trilinear_ray_adjoint"
    assert backend_report["actual_geometry_reductions"] == "core_trilinear_ray_finite_difference"
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
    assert level_rows[0]["prior_strength"] == "0.001"
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
    first_verification = dict(first.verification)
    second_verification = dict(second.verification)
    _ = first_verification.pop("runtime")
    _ = second_verification.pop("runtime")
    assert first_verification == second_verification


def test_alternating_alignment_solver_runs_smoke_profile(tmp_path: Path) -> None:
    solver = AlternatingAlignmentSolver()

    result = solver.run_smoke(tmp_path)

    assert result.final_volume.shape == (32, 32, 32)
    assert result.artifacts["verification_json"].exists()


def test_bad_view_detection_flags_view_residual_outlier() -> None:
    volume = np.zeros((4, 4, 4), dtype=np.float32)
    observed = np.zeros((4, 4, 4), dtype=np.float32)
    observed[2] = 10.0
    mask = np.ones_like(observed, dtype=np.float32)

    payload = _bad_view_detection_payload(
        final_volume=jnp.asarray(volume),
        final_geometry=GeometryState.zeros(4),
        observed=jnp.asarray(observed),
        mask=jnp.asarray(mask),
    )

    assert payload["flagged_view_indices"] == [2]
    evaluation = _bad_views_flagged_evaluation(
        threshold=True,
        bad_view_detection=payload,
    )
    assert evaluation["status"] == "passed"
    assert evaluation["value"] == 1


def test_alternating_solver_ingests_generated_synthetic_sidecars(tmp_path: Path) -> None:
    dataset_paths = generate_synthetic_dataset(
        "synth128_thermal_object_drift",
        tmp_path / "datasets",
        size=32,
        clean=True,
        views=4,
    )
    sidecars = load_synthetic_dataset_sidecars(dataset_paths.dataset_dir)
    solver = AlternatingAlignmentSolver(
        AlternatingSmokeConfig(
            size=32,
            n_views=4,
            schedule=reference_continuation_schedule("smoke32"),
            geometry_update_volume_source="fixed_synthetic_truth",
            synthetic_dataset_name="synth128_thermal_object_drift",
            synthetic_dataset_artifact_dir=dataset_paths.dataset_dir,
            synthetic_dataset_sidecar_readback={
                "validated": True,
                "source": "tomojax.datasets.load_synthetic_dataset_sidecars",
                "n_views": sidecars.true_geometry.pose.n_views,
                "volume": sidecars.volume.to_dict(),
                "projections": sidecars.projections.to_dict(),
                "mask": sidecars.mask.to_dict(),
                "consistency": sidecars.consistency.to_dict(),
            },
        )
    )

    result = solver.run_smoke(tmp_path / "run")

    assert result.verification["status"] in {"passed", "failed"}
    observed = cast("NDArray[np.float32]", np.load(result.artifacts["observed_projections_npy"]))
    generated = cast("NDArray[np.float32]", np.load(dataset_paths.projections))
    np.testing.assert_allclose(observed, generated)
    assert (
        result.initial_geometry.setup.det_u_px.value != result.final_geometry.setup.det_u_px.value
    )
    assert (
        sidecars.corrupted_geometry.setup.det_u_px.value
        != sidecars.true_geometry.setup.det_u_px.value
    )
    recovery = cast("dict[str, float | bool]", result.verification["geometry_recovery"])
    assert isinstance(recovery["supported_dofs_improved"], bool)
    assert recovery["passed"] is False
    assert cast("float", recovery["det_u_realized_rmse_px"]) < cast(
        "float", recovery["initial_det_u_realized_rmse_px"]
    )
    assert recovery["det_u_realized_rmse_px_passed"] is True
    assert recovery["theta_realized_rmse_rad_passed"] is True
    assert isinstance(recovery["theta_realized_rmse_rad"], float)
    assert isinstance(recovery["initial_theta_realized_rmse_rad"], float)
    assert result.levels[0].loss_after < result.levels[0].loss_before
    assert result.levels[0].schur_diagnostics is not None
    assert isinstance(result.levels[0].schur_diagnostics.accepted, bool)
    observability = cast(
        "dict[str, object]",
        json.loads(result.artifacts["observability_report_json"].read_text(encoding="utf-8")),
    )
    dofs = cast("dict[str, dict[str, dict[str, object]]]", observability["dofs"])
    assert dofs["setup"]["theta_offset_rad"]["active"] is True
    assert dofs["setup"]["theta_offset_rad"]["status"] == "evaluated"
    assert dofs["setup"]["det_u_px"]["active"] is True
    assert dofs["pose"]["phi_residual_rad"]["active"] is True
    assert dofs["pose"]["phi_residual_rad"]["status"] == "gauge_canonicalised"
    assert dofs["pose"]["dx_px"]["active"] is True
    assert dofs["pose"]["dx_px"]["status"] == "gauge_canonicalised"
    assert dofs["pose"]["dz_px"]["active"] is True
    assert dofs["pose"]["dz_px"]["status"] == "gauge_canonicalised"

    schur_payload = cast(
        "dict[str, object]",
        json.loads(result.artifacts["schur_diagnostics_json"].read_text(encoding="utf-8")),
    )
    assert schur_payload["status"] == "passed"
    assert schur_payload["geometry_update_volume_source"] == "fixed_synthetic_truth"
    with result.artifacts["geometry_trace_csv"].open("r", newline="", encoding="utf-8") as fh:
        trace_rows = list(csv.DictReader(fh))
    assert trace_rows[0]["schur_accepted"] in {"True", "False"}
    assert float(trace_rows[0]["loss_after"]) < float(trace_rows[0]["loss_before"])

    failure_report = cast(
        "dict[str, object]",
        json.loads(result.artifacts["failure_report_json"].read_text(encoding="utf-8")),
    )
    gates = cast("list[dict[str, object]]", failure_report["gates"])
    gates_by_name = {str(gate["name"]): gate for gate in gates}
    assert gates_by_name["synthetic_sidecar_consistency"]["passed"] is True
    assert (
        cast(
            "dict[str, object]",
            gates_by_name["synthetic_sidecar_consistency"]["evidence"],
        )["passed"]
        is True
    )


def test_alternating_solver_stopped_reconstruction_sidecar_reports_recovery_gap(
    tmp_path: Path,
) -> None:
    dataset_paths = generate_synthetic_dataset(
        "synth128_thermal_object_drift",
        tmp_path / "datasets",
        size=32,
        clean=True,
        views=4,
    )
    sidecars = load_synthetic_dataset_sidecars(dataset_paths.dataset_dir)
    solver = AlternatingAlignmentSolver(
        AlternatingSmokeConfig(
            size=32,
            n_views=4,
            schedule=reference_continuation_schedule("smoke32"),
            synthetic_dataset_name="synth128_thermal_object_drift",
            synthetic_dataset_artifact_dir=dataset_paths.dataset_dir,
            synthetic_dataset_sidecar_readback={
                "validated": True,
                "source": "tomojax.datasets.load_synthetic_dataset_sidecars",
                "n_views": sidecars.true_geometry.pose.n_views,
                "consistency": sidecars.consistency.to_dict(),
            },
        )
    )

    result = solver.run_smoke(tmp_path / "run")

    assert result.verification["status"] == "failed"
    assert result.verification["geometry_update_volume_source"] == "stopped_reconstruction"
    recovery = cast("dict[str, float | bool]", result.verification["geometry_recovery"])
    assert isinstance(recovery["supported_dofs_improved"], bool)
    assert recovery["passed"] is False
    assert cast("float", recovery["det_u_realized_rmse_px"]) < cast(
        "float", recovery["initial_det_u_realized_rmse_px"]
    )
    assert cast("float", recovery["det_u_realized_rmse_px"]) > cast(
        "float", recovery["det_u_realized_rmse_px_limit"]
    )
    assert result.levels[0].loss_after < result.levels[0].loss_before
    assert result.levels[0].schur_diagnostics is not None
    assert isinstance(result.levels[0].schur_diagnostics.accepted, bool)
    runtime = cast("dict[str, object]", result.verification["runtime"])
    assert runtime["time_to_verified_geometry_seconds"] is None
    stopped_gauge = cast(
        "dict[str, float | bool | str]", result.verification["stopped_volume_gauge"]
    )
    assert stopped_gauge["schema"] == "tomojax.stopped_volume_gauge.v1"
    assert stopped_gauge["nearest_geometry"] in {"final_geometry", "true_geometry"}
    assert isinstance(stopped_gauge["closer_to_initial_than_true"], bool)
    assert isinstance(stopped_gauge["closer_to_final_than_true"], bool)
    assert cast("float", stopped_gauge["projection_loss_final_geometry"]) < cast(
        "float", stopped_gauge["projection_loss_initial_geometry"]
    )

    schur_payload = cast(
        "dict[str, object]",
        json.loads(result.artifacts["schur_diagnostics_json"].read_text(encoding="utf-8")),
    )
    assert schur_payload["status"] == "passed"
    assert schur_payload["geometry_update_volume_source"] == "stopped_reconstruction"
    with result.artifacts["geometry_trace_csv"].open("r", newline="", encoding="utf-8") as fh:
        trace_rows = list(csv.DictReader(fh))
    assert trace_rows[0]["schur_accepted"] in {"True", "False"}
    assert np.isfinite(float(trace_rows[0]["loss_after"]))


def test_rejected_schur_update_does_not_verify_sidecar_level(tmp_path: Path) -> None:
    dataset_paths = generate_synthetic_dataset(
        "synth128_setup_global_tomo",
        tmp_path / "datasets",
        size=32,
        clean=False,
        views=4,
    )
    sidecars = load_synthetic_dataset_sidecars(dataset_paths.dataset_dir)
    solver = AlternatingAlignmentSolver(
        AlternatingSmokeConfig(
            size=32,
            n_views=4,
            schedule=reference_continuation_schedule("smoke32"),
            fit_gain_offset_nuisance=True,
            fit_background_nuisance=True,
            synthetic_dataset_name="synth128_setup_global_tomo",
            synthetic_dataset_artifact_dir=dataset_paths.dataset_dir,
            synthetic_dataset_nuisance_applied=True,
            synthetic_dataset_sidecar_readback={
                "validated": True,
                "source": "tomojax.datasets.load_synthetic_dataset_sidecars",
                "n_views": sidecars.true_geometry.pose.n_views,
                "consistency": sidecars.consistency.to_dict(),
            },
        )
    )

    result = solver.run_smoke(tmp_path / "run")

    assert result.levels[0].schur_diagnostics is not None
    assert result.levels[0].schur_diagnostics.accepted is False
    assert result.levels[0].verified is False
    assert isinstance(result.verification["coarse_verified"], bool)
    runtime = cast("dict[str, object]", result.verification["runtime"])
    assert runtime["time_to_verified_geometry_seconds"] is None
    with result.artifacts["geometry_trace_csv"].open("r", newline="", encoding="utf-8") as fh:
        trace_rows = list(csv.DictReader(fh))
    assert trace_rows[0]["schur_accepted"] == "False"
    assert trace_rows[0]["verified"] == "False"


def test_supported_dof_summary_reports_individual_dof_evidence(
    tmp_path: Path,
) -> None:
    dataset_paths = generate_synthetic_dataset(
        "synth128_setup_global_tomo",
        tmp_path / "datasets",
        size=32,
        clean=False,
        views=4,
    )
    sidecars = load_synthetic_dataset_sidecars(dataset_paths.dataset_dir)
    solver = AlternatingAlignmentSolver(
        AlternatingSmokeConfig(
            size=32,
            n_views=4,
            schedule=reference_continuation_schedule("smoke32"),
            geometry_update_volume_source="fixed_synthetic_truth",
            fit_gain_offset_nuisance=True,
            fit_background_nuisance=True,
            synthetic_dataset_name="synth128_setup_global_tomo",
            synthetic_dataset_artifact_dir=dataset_paths.dataset_dir,
            synthetic_dataset_nuisance_applied=True,
            synthetic_dataset_sidecar_readback={
                "validated": True,
                "source": "tomojax.datasets.load_synthetic_dataset_sidecars",
                "n_views": sidecars.true_geometry.pose.n_views,
                "consistency": sidecars.consistency.to_dict(),
            },
        )
    )

    result = solver.run_smoke(tmp_path / "run")

    recovery = cast("dict[str, float | bool]", result.verification["geometry_recovery"])
    assert recovery["supported_dofs_improved"] is False
    assert recovery["det_u_realized_rmse_px_improved"] is True
    assert recovery["theta_realized_rmse_rad_improved"] is True
    assert recovery["theta_realized_rmse_rad_passed"] is True
    assert recovery["det_v_realized_rmse_px_improved"] is False
    assert recovery["det_v_realized_rmse_px_passed"] is True


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
    assert "fit_gain_offset_nuisance = false" in config_text
    assert "fit_background_nuisance = false" in config_text
    assert "level_factors = [4, 2, 1]" in config_text


def test_alternating_smoke_can_enable_gain_offset_nuisance(tmp_path: Path) -> None:
    result = run_alternating_solver_smoke(
        tmp_path,
        config=AlternatingSmokeConfig(
            schedule=reference_continuation_schedule("lightning"),
            fit_gain_offset_nuisance=True,
        ),
    )

    assert result.verification["fit_gain_offset_nuisance"] is True
    assert result.levels[0].schur_diagnostics is not None
    assert result.levels[0].schur_diagnostics.gain_offset_fit is True
    schur = cast(
        "dict[str, object]",
        json.loads(result.artifacts["schur_diagnostics_json"].read_text(encoding="utf-8")),
    )
    diagnostics = cast("dict[str, object]", schur["diagnostics"])
    assert diagnostics["gain_offset_fit"] is True
    gain_offset_model = cast("dict[str, object]", diagnostics["gain_offset_model"])
    assert gain_offset_model["schema"] == "tomojax.gain_offset_model.v1"
    assert len(cast("list[float]", gain_offset_model["gain"])) == 4
    assert len(cast("list[float]", gain_offset_model["offset"])) == 4
    assert diagnostics["background_offset_model"] is None


def test_alternating_smoke_can_enable_background_nuisance(tmp_path: Path) -> None:
    result = run_alternating_solver_smoke(
        tmp_path,
        config=AlternatingSmokeConfig(
            schedule=reference_continuation_schedule("lightning"),
            fit_background_nuisance=True,
        ),
    )

    assert result.verification["fit_background_nuisance"] is True
    assert result.levels[0].schur_diagnostics is not None
    assert result.levels[0].schur_diagnostics.background_offset_fit is True
    schur = cast(
        "dict[str, object]",
        json.loads(result.artifacts["schur_diagnostics_json"].read_text(encoding="utf-8")),
    )
    diagnostics = cast("dict[str, object]", schur["diagnostics"])
    assert diagnostics["background_offset_fit"] is True
    assert diagnostics["gain_offset_model"] is None
    background_offset_model = cast("dict[str, object]", diagnostics["background_offset_model"])
    assert background_offset_model["schema"] == "tomojax.background_offset_model.v1"
    assert len(cast("list[float]", background_offset_model["constant"])) == 4
    assert len(cast("list[float]", background_offset_model["vertical_gradient"])) == 4


def test_alternating_smoke_schur_recovers_supported_dofs_with_truth_volume(
    tmp_path: Path,
) -> None:
    result = run_alternating_solver_smoke(
        tmp_path,
        config=AlternatingSmokeConfig(
            schedule=reference_continuation_schedule("lightning"),
            geometry_update_volume_source="fixed_synthetic_truth",
        ),
    )

    recovery = cast("dict[str, float | bool]", result.verification["geometry_recovery"])
    assert recovery["supported_dofs_improved"] is True
    assert recovery["theta_realized_rmse_rad_improved"] is True
    assert recovery["det_u_realized_rmse_px_improved"] is True
    assert isinstance(recovery["det_v_realized_rmse_px_improved"], bool)
    assert cast("float", recovery["theta_realized_rmse_rad"]) < cast(
        "float", recovery["initial_theta_realized_rmse_rad"]
    )
    assert cast("float", recovery["det_u_realized_rmse_px"]) < cast(
        "float", recovery["initial_det_u_realized_rmse_px"]
    )
    assert isinstance(recovery["det_v_realized_rmse_px"], float)
    assert isinstance(recovery["initial_det_v_realized_rmse_px"], float)
    assert result.levels[0].schur_diagnostics is not None
    assert result.levels[0].schur_diagnostics.accepted is True
    assert result.levels[0].schur_diagnostics.actual_reduction > 0.0
    schur = cast(
        "dict[str, object]",
        json.loads(result.artifacts["schur_diagnostics_json"].read_text(encoding="utf-8")),
    )
    assert schur["status"] == "passed"
    with result.artifacts["geometry_trace_csv"].open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["schur_accepted"] == "True"
    assert float(rows[0]["schur_actual_reduction"]) > 0.0


def test_residual_structure_summary_flags_column_nuisance() -> None:
    residual = np.zeros((4, 8, 8), dtype=np.float32)
    residual[:, :, 3] = 1.0
    residual[:, :, 4] = -1.0
    mask = np.ones_like(residual, dtype=bool)

    summary = residual_structure_summary(residual, mask)

    assert summary["passed"] is False
    assert float(cast("float", summary["column_mean_rms_ratio"])) > float(
        cast("float", summary["column_mean_rms_ratio_threshold"])
    )
