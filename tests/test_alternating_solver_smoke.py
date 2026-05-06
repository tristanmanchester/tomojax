from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING, cast

import numpy as np

from tomojax.align.api import run_alternating_solver_smoke

if TYPE_CHECKING:
    from pathlib import Path


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
        "fista_trace_csv",
        "geometry_final_json",
        "geometry_initial_json",
        "pose_decomposition_csv",
        "pose_params_csv",
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

    with result.artifacts["alignment_summary_csv"].open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["level_factor"] for row in rows] == ["4", "1"]

    assert abs(float(np.mean(result.final_geometry.pose.dx_px))) < 1.0e-12
    assert abs(float(np.mean(result.final_geometry.pose.phi_residual_rad))) < 1.0e-12


def test_alternating_solver_smoke_is_deterministic(tmp_path: Path) -> None:
    first = run_alternating_solver_smoke(tmp_path / "first")
    second = run_alternating_solver_smoke(tmp_path / "second")

    assert first.final_volume.shape == second.final_volume.shape
    np.testing.assert_allclose(first.final_volume, second.final_volume)
    np.testing.assert_allclose(first.final_geometry.pose.dx_px, second.final_geometry.pose.dx_px)
    assert first.verification == second.verification
