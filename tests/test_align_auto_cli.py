from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import tomojax.cli.align_auto as align_auto_cli
from tomojax.datasets import generate_synthetic_dataset

# pyright: reportPrivateUsage=false

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def test_align_auto_smoke_help_documents_outputs(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _ = align_auto_cli.main(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--out-dir" in captured.out
    assert "verification artifacts" in captured.out
    assert "diagnostic-fast" in captured.out
    assert "smoke32" not in captured.out
    assert "--synthetic-dataset" in captured.out
    assert "--synthetic-case" in captured.out
    assert "--synthetic-tomo-mvp-case" not in captured.out
    assert "--synthetic-dataset-dir" in captured.out
    assert "--current-default-baseline-json" in captured.out
    assert "--projection-loss-mode" in captured.out
    assert "--apply-synthetic-nuisance" in captured.out
    assert "--fit-gain-offset-nuisance" in captured.out
    assert "--fit-background-nuisance" in captured.out
    assert "--supported-only-setup-global" in captured.out
    assert "--geometry-update-pose-frozen" in captured.out
    assert "--geometry-update-active-pose-dofs" in captured.out
    assert "--geometry-update-phi-polish-updates" in captured.out
    assert "--geometry-update-final-pose-polish-updates" in captured.out
    assert "alpha_rad, beta_rad" in captured.out
    assert "--geometry-update-pose-activate-at-level-factor" in captured.out
    assert "detector_roll_rad" in captured.out
    assert "axis_rot_x_rad" in captured.out
    assert "axis_rot_y_rad" in captured.out
    assert "theta_scale" in captured.out


def test_align_auto_cli_sets_jax_no_preallocate_before_tomojax_import() -> None:
    env = dict(os.environ)
    _ = env.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "os.environ.pop('XLA_PYTHON_CLIENT_PREALLOCATE', None); "
                "import tomojax.cli.align_auto; "
                "print(os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'))"
            ),
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "false"


def test_current_default_baseline_payload_reads_direct_and_nested_nmse(tmp_path: Path) -> None:
    direct = tmp_path / "direct.json"
    _ = direct.write_text(json.dumps({"volume_nmse": 0.45}), encoding="utf-8")
    nested = tmp_path / "nested.json"
    _ = nested.write_text(
        json.dumps({"reconstruction": {"volume_nmse": 0.67}}),
        encoding="utf-8",
    )

    assert align_auto_cli._current_default_baseline_payload(direct)["volume_nmse"] == 0.45
    assert align_auto_cli._current_default_baseline_payload(nested)["volume_nmse"] == 0.67


def test_align_auto_parses_supported_geometry_update_dofs() -> None:
    assert align_auto_cli._parse_active_pose_dofs(
        "alpha_rad,beta_rad,phi_residual_rad,dx_px,dz_px"
    ) == ("alpha_rad", "beta_rad", "phi_residual_rad", "dx_px", "dz_px")
    assert align_auto_cli._parse_active_setup_parameters(
        "theta_offset_rad,det_u_px,det_v_px,detector_roll_rad,"
        "axis_rot_x_rad,axis_rot_y_rad,theta_scale"
    ) == (
        "theta_offset_rad",
        "det_u_px",
        "det_v_px",
        "detector_roll_rad",
        "axis_rot_x_rad",
        "axis_rot_y_rad",
        "theta_scale",
    )
    assert align_auto_cli._parse_active_pose_dofs("none") == ()
    assert align_auto_cli._parse_active_setup_parameters("") == ()


def test_align_auto_rejects_unknown_geometry_update_dofs() -> None:
    with pytest.raises(ValueError, match="unsupported --geometry-update-active-pose-dofs"):
        _ = align_auto_cli._parse_active_pose_dofs("omega_rad")
    with pytest.raises(ValueError, match="unsupported --geometry-update-active-setup-parameters"):
        _ = align_auto_cli._parse_active_setup_parameters("tilt_rad")


def test_synthetic_setup_global_case_resolves_bounded_oracle(
    tmp_path: Path,
) -> None:
    parser = align_auto_cli._build_parser()
    args = parser.parse_args(
        [
            "--out-dir",
            str(tmp_path),
            "--synthetic-case",
            "setup-global",
        ]
    )

    align_auto_cli._apply_synthetic_case(args)

    assert cast("str", args.profile) == "diagnostic-fast"
    assert cast("int", args.size) == 32
    assert cast("int", args.views) == 8
    assert cast("str", args.synthetic_dataset) == "synth128_setup_global_tomo"
    assert cast("str", args.geometry_update_volume_source) == "fixed_synthetic_truth"
    assert cast("str", args.geometry_update_solver) == "joint_schur"
    assert cast("bool", args.geometry_update_pose_frozen) is True
    assert (
        cast("str", args.geometry_update_active_setup_parameters)
        == "det_u_px,theta_offset_rad"
    )


def test_synthetic_pose_random_case_resolves_bounded_oracle(
    tmp_path: Path,
) -> None:
    parser = align_auto_cli._build_parser()
    args = parser.parse_args(
        [
            "--out-dir",
            str(tmp_path),
            "--synthetic-case",
            "pose-random",
            "--size",
            "64",
            "--views",
            "16",
        ]
    )

    align_auto_cli._apply_synthetic_case(args)

    assert cast("str", args.profile) == "diagnostic-fast"
    assert cast("int", args.size) == 64
    assert cast("int", args.views) == 16
    assert cast("str", args.synthetic_dataset) == "synth128_pose_random_extreme"
    assert cast("str", args.geometry_update_volume_source) == "fixed_synthetic_truth"
    assert cast("str", args.geometry_update_solver) == "joint_schur"
    assert cast("str", args.geometry_update_active_setup_parameters) == "none"
    assert (
        cast("str", args.geometry_update_active_pose_dofs)
        == "phi_residual_rad,dx_px,dz_px"
    )
    assert cast("int", args.geometry_update_alpha_beta_activate_at_level_factor) == 1
    assert cast("float", args.geometry_update_pose_trust_radius) == -1.0


def test_legacy_synthetic_tomo_mvp_case_is_hidden_alias(tmp_path: Path) -> None:
    parser = align_auto_cli._build_parser()
    args = parser.parse_args(
        [
            "--out-dir",
            str(tmp_path),
            "--synthetic-tomo-mvp-case",
            "setup_global",
        ]
    )

    align_auto_cli._apply_synthetic_case(args)

    assert cast("str", args.profile) == "diagnostic-fast"
    assert cast("str", args.synthetic_dataset) == "synth128_setup_global_tomo"


def test_align_auto_smoke_command_writes_core_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / "auto-smoke"

    exit_code = align_auto_cli.main(["--out-dir", str(out_dir)])

    assert exit_code == 0
    assert (out_dir / "final_volume.npy").exists()
    assert (out_dir / "geometry_final.json").exists()
    assert (out_dir / "verification.json").exists()
    assert not (out_dir / "benchmark_report.md").exists()
    assert not (out_dir / "benchmark_result.json").exists()
    verification = cast(
        "dict[str, object]",
        json.loads((out_dir / "verification.json").read_text(encoding="utf-8")),
    )
    assert verification["status"] == "passed"
    assert isinstance(verification["level1_geometry_skipped"], bool)
    assert verification["fit_gain_offset_nuisance"] is False
    assert verification["fit_background_nuisance"] is False
    summary = cast("dict[str, bool]", verification["summary"])
    assert isinstance(summary["projection_residual_improved"], bool)
    assert summary["gauge_constraints_satisfied"] is True
    assert isinstance(summary["all_levels_verified"], bool)
    metrics = cast("dict[str, float]", verification["metrics"])
    assert isinstance(metrics["residual_after"], float)
    assert isinstance(metrics["residual_before"], float)
    recovery = cast("dict[str, bool | float]", verification["geometry_recovery"])
    assert recovery["passed"] is True
    levels = cast("list[dict[str, object]]", verification["levels"])
    assert levels[0]["prior_strength"] == 1.0e-3

    with (out_dir / "alignment_summary.csv").open("r", newline="", encoding="utf-8") as fh:
        summary_rows = list(csv.DictReader(fh))
    assert [row["level_factor"] for row in summary_rows] == ["4", "2", "1"]
    assert summary_rows[0]["verified"] in {"True", "False"}
    assert summary_rows[0]["prior_strength"] == "0.001"
    assert int(summary_rows[-1]["executed_geometry_updates"]) >= 0

    with (out_dir / "geometry_trace.csv").open("r", newline="", encoding="utf-8") as fh:
        trace_rows = list(csv.DictReader(fh))
    assert trace_rows[0]["schur_accepted"] in {"True", "False", ""}
    assert trace_rows[0]["heldout_residual_passed"] in {"True", "False", ""}
    assert isinstance(trace_rows[-1]["early_exit_reason"], str)

    schur = cast(
        "dict[str, object]",
        json.loads((out_dir / "schur_diagnostics.json").read_text(encoding="utf-8")),
    )
    assert schur["status"] == "passed"
    assert schur["geometry_update_volume_source"] == "stopped_reconstruction"
    diagnostics = cast("dict[str, object]", schur["diagnostics"])
    assert isinstance(diagnostics["parameter_prior_strength"], float)
    assert diagnostics["gain_offset_fit"] is False
    assert diagnostics["background_offset_fit"] is False
    captured = capsys.readouterr()
    assert "verification:" in captured.out


def test_align_auto_records_geometry_first_bootstrap_stage(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "auto-bootstrap"

    exit_code = align_auto_cli.main(
        [
            "--out-dir",
            str(out_dir),
            "--synthetic-dataset",
            "synth128_setup_global_tomo",
            "--supported-only-setup-global",
            "--geometry-update-volume-source",
            "stopped_reconstruction",
            "--geometry-update-pose-frozen",
            "--geometry-update-active-setup-parameters",
            "det_u_px",
            "--views",
            "4",
        ]
    )

    assert exit_code == 0
    bootstrap = cast(
        "dict[str, object]",
        json.loads((out_dir / "bootstrap_stage.json").read_text(encoding="utf-8")),
    )
    assert bootstrap["role"] == "geometry_first_bootstrap"
    assert bootstrap["level_factor"] == 4
    assert bootstrap["schur_passes"] == 2
    assert int(cast("int", bootstrap["executed_geometry_updates"])) > 0
    assert int(cast("int", bootstrap["fista_refresh_iterations"])) > 0
    assert isinstance(cast("dict[str, object]", bootstrap["losses"])["after_final_schur"], float)
    assert isinstance(
        cast("dict[str, object]", bootstrap["final_geometry"])["det_u_px"],
        float,
    )
    diagnostics = cast("dict[str, object]", bootstrap["diagnostics"])
    assert cast("dict[str, object]", diagnostics["final_schur"])["accepted"] in {
        True,
        False,
    }
    provenance = cast(
        "dict[str, object]",
        json.loads((out_dir / "mask_provenance.json").read_text(encoding="utf-8")),
    )
    provenance_entries = cast("list[dict[str, object]]", provenance["entries"])
    bootstrap_fista = [
        entry
        for entry in provenance_entries
        if entry["stage"] == "bootstrap_fista_refresh"
        and entry["operation"] == "fista_reconstruct_reference"
    ]
    assert len(bootstrap_fista) == 1
    assert bootstrap_fista[0]["mask_role"] == "projection_valid_mask"
    assert bootstrap_fista[0]["includes_otsu"] is False
    assert bootstrap_fista[0]["includes_train_gating"] is False
    assert any(
        entry["stage"] == "bootstrap_first_schur"
        and entry["operation"] == "joint_schur_geometry_update"
        and entry["mask_role"] == "alignment_train_mask"
        for entry in provenance_entries
    )

    with (out_dir / "alignment_summary.csv").open("r", newline="", encoding="utf-8") as fh:
        summary_rows = list(csv.DictReader(fh))
    assert [row["role"] for row in summary_rows] == ["preview", "preview", "final"]

    benchmark_result = cast(
        "dict[str, object]",
        json.loads((out_dir / "benchmark_result.json").read_text(encoding="utf-8")),
    )
    runtime = cast("dict[str, object]", benchmark_result["runtime"])
    assert cast("dict[str, object]", runtime["bootstrap_stage"])["role"] == (
        "geometry_first_bootstrap"
    )
    report_text = (out_dir / "benchmark_report.md").read_text(encoding="utf-8")
    assert "## Bootstrap Runtime" in report_text
    assert "geometry_first_bootstrap" in report_text


def test_align_auto_smoke_command_can_enable_gain_offset_nuisance(tmp_path: Path) -> None:
    out_dir = tmp_path / "auto-nuisance"

    exit_code = align_auto_cli.main(
        [
            "--out-dir",
            str(out_dir),
            "--profile",
            "lightning",
            "--fit-gain-offset-nuisance",
        ]
    )

    assert exit_code == 0
    verification = cast(
        "dict[str, object]",
        json.loads((out_dir / "verification.json").read_text(encoding="utf-8")),
    )
    assert verification["fit_gain_offset_nuisance"] is True
    manifest = cast(
        "dict[str, object]",
        json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8")),
    )
    assert manifest["fit_gain_offset_nuisance"] is True
    config_text = (out_dir / "config_resolved.toml").read_text(encoding="utf-8")
    assert "fit_gain_offset_nuisance = true" in config_text
    schur = cast(
        "dict[str, object]",
        json.loads((out_dir / "schur_diagnostics.json").read_text(encoding="utf-8")),
    )
    diagnostics = cast("dict[str, object]", schur["diagnostics"])
    assert diagnostics["gain_offset_fit"] is True


def test_align_auto_smoke_command_can_enable_background_nuisance(tmp_path: Path) -> None:
    out_dir = tmp_path / "auto-background"

    exit_code = align_auto_cli.main(
        [
            "--out-dir",
            str(out_dir),
            "--profile",
            "lightning",
            "--fit-background-nuisance",
        ]
    )

    assert exit_code == 0
    verification = cast(
        "dict[str, object]",
        json.loads((out_dir / "verification.json").read_text(encoding="utf-8")),
    )
    assert verification["fit_background_nuisance"] is True
    manifest = cast(
        "dict[str, object]",
        json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8")),
    )
    assert manifest["fit_background_nuisance"] is True
    config_text = (out_dir / "config_resolved.toml").read_text(encoding="utf-8")
    assert "fit_background_nuisance = true" in config_text
    schur = cast(
        "dict[str, object]",
        json.loads((out_dir / "schur_diagnostics.json").read_text(encoding="utf-8")),
    )
    diagnostics = cast("dict[str, object]", schur["diagnostics"])
    assert diagnostics["background_offset_fit"] is True


def test_align_auto_smoke_command_generates_named_synthetic_dataset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / "auto-benchmark"

    exit_code = align_auto_cli.main(
        [
            "--out-dir",
            str(out_dir),
            "--synthetic-dataset",
            "synth128_setup_global_tomo",
            "--views",
            "4",
        ]
    )

    assert exit_code == 0
    dataset_dir = out_dir / "datasets" / "synth128_setup_global_tomo_32"
    assert (dataset_dir / "dataset_manifest.json").exists()
    verification = cast(
        "dict[str, object]",
        json.loads((out_dir / "verification.json").read_text(encoding="utf-8")),
    )
    assert verification["synthetic_dataset"] == {
        "artifact_dir": str(dataset_dir),
        "name": "synth128_setup_global_tomo",
        "nuisance_applied_to_projections": False,
        "sidecar_readback": {
            "consistency": {
                "checks": {
                    "geometry_views_match_manifest": True,
                    "mask_shape_matches_projections": True,
                    "projection_detector_shape_matches_manifest": True,
                    "projection_views_match_manifest": True,
                    "volume_shape_matches_manifest": True,
                },
                "passed": True,
            },
            "corrupted_det_u_px": 0.0,
            "n_views": 4,
            "nominal_det_u_px": 0.0,
            "mask": {
                "dtype": "bool",
                "path": str(dataset_dir / "mask.npy"),
                "shape": [4, 32, 32],
            },
            "projections": {
                "dtype": "float32",
                "path": str(dataset_dir / "projections.npy"),
                "shape": [4, 32, 32],
            },
            "recovery_tolerances": {
                "axis_error_deg_lt": 0.1,
                "det_u_error_px_lt": 0.5,
                "roll_error_deg_lt": 0.05,
                "theta_offset_error_deg_lt": 0.1,
            },
            "source": "tomojax.datasets.load_synthetic_dataset_sidecars",
            "true_det_u_px": 3.625,
            "validated": True,
            "volume": {
                "dtype": "float32",
                "path": str(dataset_dir / "ground_truth_volume.npy"),
                "shape": [32, 32, 32],
            },
        },
        "source": "synthetic128_spec",
    }
    manifest = cast(
        "dict[str, object]",
        json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8")),
    )
    dataset = cast("dict[str, object]", manifest["dataset"])
    synthetic_benchmark = cast("dict[str, object]", dataset["synthetic128_benchmark"])
    assert synthetic_benchmark["name"] == "synth128_setup_global_tomo"
    assert synthetic_benchmark["nuisance_applied_to_projections"] is False
    sidecar_readback = cast("dict[str, object]", synthetic_benchmark["sidecar_readback"])
    assert sidecar_readback["validated"] is True
    assert sidecar_readback["n_views"] == 4
    assert sidecar_readback["true_det_u_px"] == 3.625
    projections = cast("dict[str, object]", sidecar_readback["projections"])
    assert projections["shape"] == [4, 32, 32]
    assert projections["dtype"] == "float32"
    consistency = cast("dict[str, object]", sidecar_readback["consistency"])
    assert consistency["passed"] is True
    config_text = (out_dir / "config_resolved.toml").read_text(encoding="utf-8")
    assert 'synthetic_dataset_name = "synth128_setup_global_tomo"' in config_text
    assert "synthetic_dataset_nuisance_applied = false" in config_text
    assert "synthetic_dataset_sidecars_validated = true" in config_text
    assert "synthetic_dataset_sidecar_views = 4" in config_text
    assert "synthetic_dataset_projection_shape = [4, 32, 32]" in config_text
    assert 'synthetic_dataset_projection_dtype = "float32"' in config_text
    assert "synthetic_dataset_sidecar_consistency_passed = true" in config_text
    benchmark_result = cast(
        "dict[str, object]",
        json.loads((out_dir / "benchmark_result.json").read_text(encoding="utf-8")),
    )
    evaluation = cast(
        "dict[str, dict[str, object]]",
        benchmark_result["benchmark_manifest_evaluation"],
    )
    assert evaluation["det_u_error_px_lt"]["status"] == "failed"
    assert evaluation["det_u_error_px_lt"]["threshold"] == 0.5
    assert evaluation["theta_offset_error_deg_lt"]["status"] == "failed"
    np.testing.assert_allclose(
        float(cast("float", evaluation["theta_offset_error_deg_lt"]["threshold"])),
        math.radians(0.1),
    )
    assert evaluation["axis_error_deg_lt"]["status"] == "failed"
    np.testing.assert_allclose(
        float(cast("float", evaluation["axis_error_deg_lt"]["threshold"])),
        math.radians(0.1),
    )
    assert evaluation["roll_error_deg_lt"]["status"] == "failed"
    np.testing.assert_allclose(
        float(cast("float", evaluation["roll_error_deg_lt"]["threshold"])),
        math.radians(0.05),
    )
    evaluation_summary = cast(
        "dict[str, object]",
        benchmark_result["benchmark_manifest_evaluation_summary"],
    )
    assert evaluation_summary == {
        "failed": 4,
        "not_evaluated": 0,
        "passed": 0,
        "status": "failed",
        "total": 4,
    }
    nuisance = cast(
        "dict[str, object]",
        json.loads((dataset_dir / "nuisance_truth.json").read_text(encoding="utf-8")),
    )
    assert nuisance["applied_to_projections"] is False
    failure_report = cast(
        "dict[str, object]",
        json.loads((out_dir / "failure_report.json").read_text(encoding="utf-8")),
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
    captured = capsys.readouterr()
    assert "synthetic_dataset:" in captured.out


def test_align_auto_smoke_command_can_generate_dirty_synthetic_dataset(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "auto-dirty-benchmark"

    exit_code = align_auto_cli.main(
        [
            "--out-dir",
            str(out_dir),
            "--synthetic-dataset",
            "synth128_combined_nuisance_jumps",
            "--apply-synthetic-nuisance",
            "--views",
            "4",
        ]
    )

    assert exit_code == 0
    dataset_dir = out_dir / "datasets" / "synth128_combined_nuisance_jumps_32"
    verification = cast(
        "dict[str, object]",
        json.loads((out_dir / "verification.json").read_text(encoding="utf-8")),
    )
    synthetic_dataset = cast("dict[str, object]", verification["synthetic_dataset"])
    assert synthetic_dataset["nuisance_applied_to_projections"] is True
    sidecar_readback = cast("dict[str, object]", synthetic_dataset["sidecar_readback"])
    assert sidecar_readback["validated"] is True
    assert sidecar_readback["n_views"] == 4
    assert sidecar_readback["unsupported_dofs_not_evaluated"] == ["object_motion"]
    readback_tolerances = cast("dict[str, object]", sidecar_readback["recovery_tolerances"])
    assert "det_u_error_px_lt" in readback_tolerances
    projections = cast("dict[str, object]", sidecar_readback["projections"])
    assert projections["shape"] == [4, 32, 32]
    consistency = cast("dict[str, object]", sidecar_readback["consistency"])
    assert consistency["passed"] is True
    manifest = cast(
        "dict[str, object]",
        json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8")),
    )
    dataset = cast("dict[str, object]", manifest["dataset"])
    synthetic_benchmark = cast("dict[str, object]", dataset["synthetic128_benchmark"])
    assert synthetic_benchmark["nuisance_applied_to_projections"] is True
    config_text = (out_dir / "config_resolved.toml").read_text(encoding="utf-8")
    assert "synthetic_dataset_nuisance_applied = true" in config_text
    nuisance = cast(
        "dict[str, object]",
        json.loads((dataset_dir / "nuisance_truth.json").read_text(encoding="utf-8")),
    )
    assert nuisance["applied_to_projections"] is True
    benchmark_result = cast(
        "dict[str, object]",
        json.loads((out_dir / "benchmark_result.json").read_text(encoding="utf-8")),
    )
    bad_view_detection = cast("dict[str, object]", benchmark_result["bad_view_detection"])
    assert bad_view_detection["schema"] == "tomojax.bad_view_detection.v1"
    assert isinstance(bad_view_detection["flagged_view_indices"], list)
    evaluation = cast(
        "dict[str, dict[str, object]]",
        benchmark_result["benchmark_manifest_evaluation"],
    )
    assert evaluation["bad_views_flagged"]["status"] in {"passed", "failed"}
    assert evaluation["bad_views_flagged"]["reason"] == (
        "evaluated from robust per-view residual outlier detection"
    )


def test_align_auto_smoke_command_ingests_existing_synthetic_dataset_dir(  # noqa: PLR0915 - broad artifact smoke
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_paths = generate_synthetic_dataset(
        "synth128_thermal_object_drift",
        tmp_path / "prepared-datasets",
        size=32,
        clean=True,
        views=4,
    )
    out_dir = tmp_path / "auto-existing-benchmark"

    exit_code = align_auto_cli.main(
        [
            "--out-dir",
            str(out_dir),
            "--synthetic-dataset-dir",
            str(dataset_paths.dataset_dir),
        ]
    )

    assert exit_code == 0
    assert not (out_dir / "datasets").exists()
    verification = cast(
        "dict[str, object]",
        json.loads((out_dir / "verification.json").read_text(encoding="utf-8")),
    )
    synthetic_dataset = cast("dict[str, object]", verification["synthetic_dataset"])
    assert synthetic_dataset["name"] == "synth128_thermal_object_drift"
    assert synthetic_dataset["artifact_dir"] == str(dataset_paths.dataset_dir)
    assert synthetic_dataset["nuisance_applied_to_projections"] is False
    sidecar_readback = cast("dict[str, object]", synthetic_dataset["sidecar_readback"])
    assert sidecar_readback["validated"] is True
    assert sidecar_readback["n_views"] == 4
    true_object_motion = cast("dict[str, object]", sidecar_readback["true_object_motion"])
    assert true_object_motion["has_nonzero_motion"] is True
    assert float(cast("float", true_object_motion["tx_zero_model_rmse_px"])) > 0.0
    readback_tolerances = cast("dict[str, object]", sidecar_readback["recovery_tolerances"])
    assert readback_tolerances["core_solver"] == "flags_object_motion_suspected"
    observed = cast("NDArray[np.float32]", np.load(out_dir / "observed_projections.npy"))
    generated = cast("NDArray[np.float32]", np.load(dataset_paths.projections))
    np.testing.assert_allclose(observed, generated)
    benchmark_result = cast(
        "dict[str, object]",
        json.loads((out_dir / "benchmark_result.json").read_text(encoding="utf-8")),
    )
    assert benchmark_result["schema"] == "tomojax.synthetic_benchmark_result.v1"
    assert benchmark_result["benchmark"] == "synth128_thermal_object_drift"
    assert benchmark_result["implementation"] == "reimagined_align_auto_smoke"
    assert benchmark_result["profile"] == "smoke32"
    _assert_benchmark_criteria_and_runtime(benchmark_result, readback_tolerances)
    dataset = cast("dict[str, object]", benchmark_result["dataset"])
    assert dataset["artifact_dir"] == str(dataset_paths.dataset_dir)
    assert dataset["volume_shape"] == [32, 32, 32]
    _assert_projection_loss_provenance(benchmark_result)
    geometry_recovery = cast("dict[str, object]", benchmark_result["geometry_recovery"])
    assert isinstance(geometry_recovery["supported_dofs_improved"], bool)
    backend = cast("dict[str, object]", benchmark_result["backend"])
    assert backend["actual"] == "core_trilinear_ray"
    assert all(
        isinstance(backend[key], str) and backend[key]
        for key in ("jax_default_backend", "selected_jax_device")
    )
    object_motion = cast("dict[str, object]", benchmark_result["object_motion_suspicion"])
    assert object_motion["suspected"] is True
    assert object_motion["evidence_sources"] == [
        "synthetic_sidecar_unsupported_dof",
        "smooth_pose_drift",
    ]
    object_recovery = cast("dict[str, object]", benchmark_result["object_motion_recovery"])
    assert object_recovery["enabled"] is False
    assert object_recovery["tx_rmse_px"] == true_object_motion["tx_zero_model_rmse_px"]
    benchmark_report = (out_dir / "benchmark_report.md").read_text(encoding="utf-8")
    assert "# Benchmark: synth128_thermal_object_drift" in benchmark_report
    assert "reimagined_align_auto_smoke" in benchmark_report
    assert "## Geometry Recovery" in benchmark_report
    assert "## Benchmark Manifest Criteria" in benchmark_report
    assert "## Benchmark Manifest Evaluation" in benchmark_report
    assert "## Projection Loss Provenance" in benchmark_report
    assert "Schur train loss" in benchmark_report
    assert "flags_object_motion_suspected" in benchmark_report
    assert "## Backend Provenance" in benchmark_report
    assert "| reimagined_align_auto_smoke | smoke32 |" in benchmark_report
    assert "core_trilinear_ray" in benchmark_report
    config_text = (out_dir / "config_resolved.toml").read_text(encoding="utf-8")
    assert f'synthetic_dataset_artifact_dir = "{dataset_paths.dataset_dir}"' in config_text
    assert 'geometry_update_volume_source = "stopped_reconstruction"' in config_text
    captured = capsys.readouterr()
    assert f"synthetic_dataset: {dataset_paths.dataset_dir}" in captured.out


def _assert_projection_loss_provenance(benchmark_result: dict[str, object]) -> None:
    reconstruction = cast("dict[str, object]", benchmark_result["reconstruction"])
    assert isinstance(reconstruction["final_residual"], float)
    assert (
        reconstruction["final_residual"]
        == reconstruction["final_volume_final_geometry_loss_all_views"]
    )
    assert isinstance(reconstruction["schur_train_loss"], float)
    assert isinstance(reconstruction["final_volume_true_geometry_loss_all_views"], float)
    assert isinstance(reconstruction["true_volume_final_geometry_loss_all_views"], float)
    assert isinstance(reconstruction["true_volume_true_geometry_loss_all_views"], float)
    assert isinstance(reconstruction["projection_loss_classification"], str)


def test_align_auto_accepts_geometry_update_volume_source(
    tmp_path: Path,
) -> None:
    dataset_paths = generate_synthetic_dataset(
        "synth128_thermal_object_drift",
        tmp_path / "datasets",
        size=32,
        clean=True,
        views=4,
    )
    out_dir = tmp_path / "auto-truth-source"

    exit_code = align_auto_cli.main(
        [
            "--out-dir",
            str(out_dir),
            "--synthetic-dataset-dir",
            str(dataset_paths.dataset_dir),
            "--geometry-update-volume-source",
            "fixed_synthetic_truth",
            "--geometry-update-setup-prior-strength",
            "0.002",
            "--geometry-update-pose-prior-strength",
            "10.0",
            "--geometry-update-pose-trust-radius",
            "-1",
        ]
    )

    assert exit_code == 0
    verification = cast(
        "dict[str, object]",
        json.loads((out_dir / "verification.json").read_text(encoding="utf-8")),
    )
    assert verification["geometry_update_volume_source"] == "fixed_synthetic_truth"
    config_text = (out_dir / "config_resolved.toml").read_text(encoding="utf-8")
    assert 'geometry_update_volume_source = "fixed_synthetic_truth"' in config_text
    assert "geometry_update_setup_prior_strength = 0.002" in config_text
    assert "geometry_update_pose_prior_strength = 10.0" in config_text
    assert "geometry_update_pose_trust_radius = -1.0" in config_text


def test_align_auto_generates_supported_only_pose_frozen_oracle(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "auto-supported-only"

    exit_code = align_auto_cli.main(
        [
            "--out-dir",
            str(out_dir),
            "--synthetic-dataset",
            "synth128_setup_global_tomo",
            "--supported-only-setup-global",
            "--geometry-update-volume-source",
            "fixed_synthetic_truth",
            "--geometry-update-pose-frozen",
            "--geometry-update-active-pose-dofs",
            "dx_px,dz_px",
            "--geometry-update-active-setup-parameters",
            "det_u_px",
            "--geometry-update-pose-activate-at-level-factor",
            "1",
            "--geometry-update-alpha-beta-activate-at-level-factor",
            "1",
            "--geometry-update-theta-activate-at-level-factor",
            "1",
            "--preview-volume-support",
            "cylindrical",
            "--preview-initialization",
            "zero",
            "--preview-tv-scale",
            "0.0",
            "--preview-residual-filter-mode",
            "raw",
            "--preview-center-l2-weight",
            "0.25",
            "--stopped-preview-policy",
            "constant_cylindrical_first_level",
            "--views",
            "4",
        ]
    )

    assert exit_code == 0
    dataset_dir = out_dir / "datasets" / "synth128_setup_global_tomo_32_supported_only"
    assert dataset_dir.is_dir()
    manifest = cast(
        "dict[str, object]",
        json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8")),
    )
    assert manifest["variant"] == "supported_only"
    config_text = (out_dir / "config_resolved.toml").read_text(encoding="utf-8")
    assert 'geometry_update_volume_source = "fixed_synthetic_truth"' in config_text
    assert 'geometry_update_solver = "joint_schur"' in config_text
    assert "geometry_update_pose_frozen = true" in config_text
    assert "geometry_update_pose_activate_at_level_factor = 1" in config_text
    assert "geometry_update_alpha_beta_activate_at_level_factor = 1" in config_text
    assert "geometry_update_theta_activate_at_level_factor = 1" in config_text
    assert 'geometry_update_active_setup_parameters = ["det_u_px"]' in config_text
    assert 'geometry_update_active_pose_dofs = ["dx_px", "dz_px"]' in config_text
    assert 'preview_volume_support = "cylindrical"' in config_text
    assert 'preview_initialization = "zero"' in config_text
    assert 'preview_reconstruction_mask_source = "all_views"' in config_text
    assert "preview_tv_scale = 0.0" in config_text
    assert 'preview_residual_filter_mode = "raw"' in config_text
    assert "preview_center_l2_weight = 0.25" in config_text
    assert 'stopped_preview_policy = "constant_cylindrical_first_level"' in config_text
    schur = cast(
        "dict[str, object]",
        json.loads((out_dir / "schur_diagnostics.json").read_text(encoding="utf-8")),
    )
    assert schur["active_pose_dofs"] == []
    assert "phi_residual_rad" in cast("list[str]", schur["frozen_parameters"])
    benchmark_result = cast(
        "dict[str, object]",
        json.loads((out_dir / "benchmark_result.json").read_text(encoding="utf-8")),
    )
    assert benchmark_result["geometry_update_volume_source"] == "fixed_synthetic_truth"
    assert benchmark_result["geometry_update_solver"] == "joint_schur"
    assert benchmark_result["preview_volume_support"] == "cylindrical"
    assert benchmark_result["preview_initialization"] == "zero"
    assert benchmark_result["preview_tv_scale"] == 0.0
    assert benchmark_result["preview_residual_filter_mode"] == "raw"
    assert benchmark_result["preview_center_l2_weight"] == 0.25
    assert benchmark_result["stopped_preview_policy"] == "constant_cylindrical_first_level"
    reconstruction = cast("dict[str, object]", benchmark_result["reconstruction"])
    assert (
        reconstruction["final_residual"]
        == reconstruction["final_volume_final_geometry_loss_all_views"]
    )
    assert isinstance(reconstruction["true_volume_true_geometry_loss_all_views"], float)
    criteria = cast("dict[str, object]", benchmark_result["benchmark_manifest_criteria"])
    assert set(criteria) == {"det_u_error_px_lt", "theta_offset_error_deg_lt"}


def _assert_benchmark_criteria_and_runtime(
    benchmark_result: dict[str, object],
    readback_tolerances: dict[str, object],
) -> None:
    manifest_criteria = cast(
        "dict[str, object]",
        benchmark_result["benchmark_manifest_criteria"],
    )
    assert manifest_criteria == readback_tolerances
    criteria_evaluation = cast(
        "dict[str, dict[str, object]]",
        benchmark_result["benchmark_manifest_evaluation"],
    )
    assert criteria_evaluation["core_solver"]["status"] == "passed"
    assert criteria_evaluation["core_solver"]["threshold"] == "flags_object_motion_suspected"
    criteria_summary = cast(
        "dict[str, object]",
        benchmark_result["benchmark_manifest_evaluation_summary"],
    )
    assert criteria_summary == {
        "failed": 1,
        "not_evaluated": 0,
        "passed": 1,
        "status": "failed",
        "total": 2,
    }
    runtime = cast("dict[str, object]", benchmark_result["runtime"])
    assert runtime["time_to_verified_geometry_seconds"] is None or isinstance(
        runtime["time_to_verified_geometry_seconds"],
        float,
    )
    assert isinstance(runtime["total_wall_seconds"], float)
    if runtime["time_to_verified_geometry_seconds"] is not None:
        assert float(cast("float", runtime["time_to_verified_geometry_seconds"])) > 0.0
        assert float(cast("float", runtime["total_wall_seconds"])) >= float(
            cast("float", runtime["time_to_verified_geometry_seconds"])
        )
    assert int(cast("int", runtime["geometry_updates_executed"])) > 0
