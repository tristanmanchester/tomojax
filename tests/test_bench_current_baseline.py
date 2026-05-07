from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest

from tomojax.bench import current_baseline_payload, write_current_baseline_artifacts
from tomojax.bench.current_baseline import main as current_baseline_main

if TYPE_CHECKING:
    from pathlib import Path


def test_current_baseline_payload_reads_nested_metrics(tmp_path: Path) -> None:
    source = tmp_path / "current_metrics.json"
    _ = source.write_text(
        json.dumps(
            {
                "reconstruction": {
                    "volume_nmse": 0.42,
                    "final_residual": 0.031,
                },
                "command": "current tomojax ...",
            }
        ),
        encoding="utf-8",
    )

    payload = current_baseline_payload(
        source_path=source,
        benchmark="synth128_combined_nuisance_jumps",
        profile="default",
    )

    assert payload["schema"] == "tomojax.current_default_baseline.v1"
    assert payload["benchmark"] == "synth128_combined_nuisance_jumps"
    assert payload["volume_nmse"] == 0.42
    reconstruction = payload["reconstruction"]
    assert isinstance(reconstruction, dict)
    assert reconstruction["final_residual"] == 0.031


def test_write_current_baseline_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "current_metrics.json"
    _ = source.write_text(json.dumps({"volume_nmse": 0.5}), encoding="utf-8")

    json_path, md_path = write_current_baseline_artifacts(
        source_path=source,
        output_dir=tmp_path / "out",
        benchmark="synth128_setup_global_tomo",
        profile="lightning",
    )

    payload = cast("dict[str, object]", json.loads(json_path.read_text(encoding="utf-8")))
    assert payload["volume_nmse"] == 0.5
    assert "synth128_setup_global_tomo" in md_path.read_text(encoding="utf-8")


def test_current_baseline_cli_writes_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "current_metrics.json"
    _ = source.write_text(json.dumps({"volume_nmse": 0.25}), encoding="utf-8")
    out_dir = tmp_path / "out"

    exit_code = current_baseline_main(
        [
            str(source),
            "--out-dir",
            str(out_dir),
            "--benchmark",
            "synth128_setup_global_tomo",
            "--profile",
            "default",
        ]
    )

    assert exit_code == 0
    assert (out_dir / "benchmark_baseline_current.json").is_file()
    assert (out_dir / "benchmark_baseline_current.md").is_file()
    assert "benchmark_baseline_current_json" in capsys.readouterr().out


def test_current_baseline_payload_requires_volume_nmse(tmp_path: Path) -> None:
    source = tmp_path / "current_metrics.json"
    _ = source.write_text(json.dumps({"final_residual": 0.1}), encoding="utf-8")

    with pytest.raises(ValueError, match="volume_nmse"):
        _ = current_baseline_payload(
            source_path=source,
            benchmark="synth128_setup_global_tomo",
            profile="default",
        )
