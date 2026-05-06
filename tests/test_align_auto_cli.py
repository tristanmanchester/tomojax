from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING, cast

import pytest

import tomojax.cli.align_auto as align_auto_cli

if TYPE_CHECKING:
    from pathlib import Path


def test_align_auto_smoke_help_documents_outputs(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _ = align_auto_cli.main(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--out-dir" in captured.out
    assert "verification artifacts" in captured.out
    assert "smoke32" in captured.out
    assert "--synthetic-dataset" in captured.out
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
            "corrupted_det_u_px": 0.0,
            "n_views": 4,
            "nominal_det_u_px": 0.0,
            "source": "tomojax.datasets.load_synthetic_dataset_sidecars",
            "true_det_u_px": 14.5,
            "validated": True,
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
    assert sidecar_readback["true_det_u_px"] == 14.5
    config_text = (out_dir / "config_resolved.toml").read_text(encoding="utf-8")
    assert 'synthetic_dataset_name = "synth128_setup_global_tomo"' in config_text
    assert "synthetic_dataset_nuisance_applied = false" in config_text
    assert "synthetic_dataset_sidecars_validated = true" in config_text
    assert "synthetic_dataset_sidecar_views = 4" in config_text
    nuisance = cast(
        "dict[str, object]",
        json.loads((dataset_dir / "nuisance_truth.json").read_text(encoding="utf-8")),
    )
    assert nuisance["applied_to_projections"] is False
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
