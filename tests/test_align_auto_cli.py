from __future__ import annotations

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
    captured = capsys.readouterr()
    assert "verification:" in captured.out


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
        "source": "synthetic128_spec",
    }
    manifest = cast(
        "dict[str, object]",
        json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8")),
    )
    dataset = cast("dict[str, object]", manifest["dataset"])
    synthetic_benchmark = cast("dict[str, object]", dataset["synthetic128_benchmark"])
    assert synthetic_benchmark["name"] == "synth128_setup_global_tomo"
    config_text = (out_dir / "config_resolved.toml").read_text(encoding="utf-8")
    assert 'synthetic_dataset_name = "synth128_setup_global_tomo"' in config_text
    captured = capsys.readouterr()
    assert "synthetic_dataset:" in captured.out
