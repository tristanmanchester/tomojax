from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING, cast

import numpy as np

from tomojax.align.api import run_alternating_solver_smoke

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def test_alternating_solver_smoke_writes_artifacts(tmp_path: Path) -> None:
    result = run_alternating_solver_smoke(tmp_path)

    assert result.final_volume.shape == (32, 32, 32)
    assert result.verification["size"] == 32
    assert result.verification["seed"] == 17
    assert result.verification["coarse_verified"] is True
    assert result.verification["level1_geometry_skipped"] is True
    assert result.levels[0].geometry_updates == 1
    assert result.levels[-1].skipped_geometry is True

    expected = {
        "alignment_summary_csv",
        "artifact_index_json",
        "backend_report_json",
        "config_resolved_toml",
        "final_volume_npy",
        "fista_trace_csv",
        "gauge_report_json",
        "geometry_final_json",
        "geometry_initial_json",
        "input_summary_json",
        "mask_summary_json",
        "pose_decomposition_csv",
        "pose_params_csv",
        "projection_stats_json",
        "residual_metrics_csv",
        "run_manifest_json",
        "verification_json",
    }
    assert set(result.artifacts) == expected
    for path in result.artifacts.values():
        assert path.exists()

    artifact_index = cast(
        "dict[str, object]",
        json.loads(result.artifacts["artifact_index_json"].read_text(encoding="utf-8")),
    )
    indexed_artifacts = cast("list[dict[str, str]]", artifact_index["artifacts"])
    assert {item["name"] for item in indexed_artifacts} == expected - {"artifact_index_json"}
    assert all(item["type"] for item in indexed_artifacts)
    assert all(item["description"] for item in indexed_artifacts)

    with result.artifacts["alignment_summary_csv"].open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["level_factor"] for row in rows] == ["4", "1"]

    saved_volume = cast("NDArray[np.float32]", np.load(result.artifacts["final_volume_npy"]))
    np.testing.assert_allclose(saved_volume, result.final_volume)

    manifest = cast(
        "dict[str, object]",
        json.loads(result.artifacts["run_manifest_json"].read_text(encoding="utf-8")),
    )
    assert manifest["align_mode"] == "auto"
    assert manifest["backend_actual"] == "jax_reference"
    assert manifest["status"] == "passed"

    gauge_report = cast(
        "dict[str, object]",
        json.loads(result.artifacts["gauge_report_json"].read_text(encoding="utf-8")),
    )
    operations = cast("list[dict[str, object]]", gauge_report["operations"])
    assert {operation["target"] for operation in operations} >= {
        "setup.det_u_px",
        "setup.theta_offset_rad",
    }

    with result.artifacts["residual_metrics_csv"].open("r", newline="", encoding="utf-8") as fh:
        metric_rows = list(csv.DictReader(fh))
    assert [row["level_factor"] for row in metric_rows] == ["4", "1"]

    assert abs(float(np.mean(result.final_geometry.pose.dx_px))) < 1.0e-12
    assert abs(float(np.mean(result.final_geometry.pose.phi_residual_rad))) < 1.0e-12


def test_alternating_solver_smoke_is_deterministic(tmp_path: Path) -> None:
    first = run_alternating_solver_smoke(tmp_path / "first")
    second = run_alternating_solver_smoke(tmp_path / "second")

    assert first.final_volume.shape == second.final_volume.shape
    np.testing.assert_allclose(first.final_volume, second.final_volume)
    np.testing.assert_allclose(first.final_geometry.pose.dx_px, second.final_geometry.pose.dx_px)
    assert first.verification == second.verification
