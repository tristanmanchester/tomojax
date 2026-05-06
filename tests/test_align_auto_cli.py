from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import tomojax.cli.align_auto as align_auto_cli
from tomojax.datasets import generate_synthetic_dataset

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
    assert "smoke32" in captured.out
    assert "--synthetic-dataset" in captured.out
    assert "--synthetic-dataset-dir" in captured.out
    assert "--apply-synthetic-nuisance" in captured.out
    assert "--fit-gain-offset-nuisance" in captured.out
    assert "--fit-background-nuisance" in captured.out


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
    assert verification["level1_geometry_skipped"] is True
    assert verification["fit_gain_offset_nuisance"] is False
    assert verification["fit_background_nuisance"] is False
    summary = cast("dict[str, bool]", verification["summary"])
    assert summary["projection_residual_improved"] is True
    assert summary["gauge_constraints_satisfied"] is True
    assert summary["all_levels_verified"] is True
    metrics = cast("dict[str, float]", verification["metrics"])
    assert metrics["residual_after"] < metrics["residual_before"]
    recovery = cast("dict[str, bool | float]", verification["geometry_recovery"])
    assert recovery["passed"] is True
    levels = cast("list[dict[str, object]]", verification["levels"])
    assert levels[0]["prior_strength"] == 1.0e-3

    with (out_dir / "alignment_summary.csv").open("r", newline="", encoding="utf-8") as fh:
        summary_rows = list(csv.DictReader(fh))
    assert [row["level_factor"] for row in summary_rows] == ["4", "2", "1"]
    assert summary_rows[0]["verified"] == "True"
    assert summary_rows[0]["prior_strength"] == "0.001"
    assert summary_rows[-1]["executed_geometry_updates"] == "0"

    with (out_dir / "geometry_trace.csv").open("r", newline="", encoding="utf-8") as fh:
        trace_rows = list(csv.DictReader(fh))
    assert trace_rows[0]["schur_accepted"] == "True"
    assert trace_rows[0]["heldout_residual_passed"] == "True"
    assert trace_rows[-1]["early_exit_reason"] == "coarse_verification_passed"

    schur = cast(
        "dict[str, object]",
        json.loads((out_dir / "schur_diagnostics.json").read_text(encoding="utf-8")),
    )
    assert schur["status"] == "passed"
    assert schur["geometry_update_volume_source"] == "stopped_reconstruction"
    diagnostics = cast("dict[str, object]", schur["diagnostics"])
    assert diagnostics["parameter_prior_strength"] == 1.0e-3
    assert diagnostics["gain_offset_fit"] is False
    assert diagnostics["background_offset_fit"] is False
    captured = capsys.readouterr()
    assert "verification:" in captured.out


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
    assert evaluation["axis_error_deg_lt"]["status"] == "not_evaluated"
    evaluation_summary = cast(
        "dict[str, object]",
        benchmark_result["benchmark_manifest_evaluation_summary"],
    )
    assert evaluation_summary == {
        "failed": 1,
        "not_evaluated": 3,
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


def test_align_auto_smoke_command_ingests_existing_synthetic_dataset_dir(
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
    reconstruction = cast("dict[str, object]", benchmark_result["reconstruction"])
    assert isinstance(reconstruction["final_residual"], float)
    geometry_recovery = cast("dict[str, object]", benchmark_result["geometry_recovery"])
    assert geometry_recovery["supported_dofs_improved"] is True
    backend = cast("dict[str, object]", benchmark_result["backend"])
    assert backend["actual"] == "jax_reference"
    benchmark_report = (out_dir / "benchmark_report.md").read_text(encoding="utf-8")
    assert "# Benchmark: synth128_thermal_object_drift" in benchmark_report
    assert "reimagined_align_auto_smoke" in benchmark_report
    assert "## Geometry Recovery" in benchmark_report
    assert "## Benchmark Manifest Criteria" in benchmark_report
    assert "## Benchmark Manifest Evaluation" in benchmark_report
    assert "flags_object_motion_suspected" in benchmark_report
    assert "## Backend Provenance" in benchmark_report
    assert "| reimagined_align_auto_smoke | smoke32 |" in benchmark_report
    assert "jax_reference" in benchmark_report
    config_text = (out_dir / "config_resolved.toml").read_text(encoding="utf-8")
    assert f'synthetic_dataset_artifact_dir = "{dataset_paths.dataset_dir}"' in config_text
    captured = capsys.readouterr()
    assert f"synthetic_dataset: {dataset_paths.dataset_dir}" in captured.out


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
    assert criteria_evaluation["core_solver"]["status"] == "not_evaluated"
    assert criteria_evaluation["core_solver"]["threshold"] == "flags_object_motion_suspected"
    criteria_summary = cast(
        "dict[str, object]",
        benchmark_result["benchmark_manifest_evaluation_summary"],
    )
    assert criteria_summary == {
        "failed": 0,
        "not_evaluated": 2,
        "passed": 0,
        "status": "partially_evaluated",
        "total": 2,
    }
    runtime = cast("dict[str, object]", benchmark_result["runtime"])
    assert isinstance(runtime["time_to_verified_geometry_seconds"], float)
    assert isinstance(runtime["total_wall_seconds"], float)
    assert float(cast("float", runtime["time_to_verified_geometry_seconds"])) > 0.0
    assert float(cast("float", runtime["total_wall_seconds"])) >= float(
        cast("float", runtime["time_to_verified_geometry_seconds"])
    )
    assert int(cast("int", runtime["geometry_updates_executed"])) > 0
