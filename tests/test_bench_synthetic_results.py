from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tomojax.bench import (
    load_synthetic_benchmark_result,
    load_synthetic_benchmark_results,
    synthetic_benchmark_comparison_markdown,
    write_synthetic_benchmark_comparison_markdown,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_loads_and_renders_synthetic_benchmark_comparison(tmp_path: Path) -> None:
    second = _write_result(
        tmp_path / "b" / "benchmark_result.json",
        benchmark="synth128_pose_random_extreme",
        status="failed",
        criteria_status="failed",
        geometry_passed=False,
        volume_nmse=1.25,
    )
    first = _write_result(
        tmp_path / "a" / "benchmark_result.json",
        benchmark="synth128_setup_global_tomo",
        status="passed",
        criteria_status="partially_evaluated",
        geometry_passed=True,
        volume_nmse=0.25,
    )

    results = load_synthetic_benchmark_results((second, first))
    markdown = synthetic_benchmark_comparison_markdown(results)

    assert markdown.startswith("# Synthetic Benchmark Comparison\n")
    assert "| Benchmark | Impl | Profile | Status | Criteria | Geometry |" in markdown
    assert markdown.index("synth128_pose_random_extreme") < markdown.index(
        "synth128_setup_global_tomo"
    )
    assert _row_cells(markdown, "synth128_pose_random_extreme") == [
        "synth128_pose_random_extreme",
        "reimagined_align_auto_smoke",
        "smoke32",
        "failed",
        "failed",
        "failed",
        "1.25",
        "0.0125",
        "0.5",
        "0.75",
        str(second),
    ]
    assert _row_cells(markdown, "synth128_setup_global_tomo") == [
        "synth128_setup_global_tomo",
        "reimagined_align_auto_smoke",
        "smoke32",
        "passed",
        "partially_evaluated",
        "passed",
        "0.25",
        "0.0125",
        "0.5",
        "0.75",
        str(first),
    ]


def test_writes_synthetic_benchmark_comparison(tmp_path: Path) -> None:
    result_path = _write_result(
        tmp_path / "run" / "benchmark_result.json",
        benchmark="synth128_setup_global_tomo",
    )
    result = load_synthetic_benchmark_result(result_path)

    out_path = write_synthetic_benchmark_comparison_markdown([result], tmp_path / "comparison.md")

    assert out_path == tmp_path / "comparison.md"
    assert "synth128_setup_global_tomo" in out_path.read_text(encoding="utf-8")


def test_rejects_non_benchmark_result_artifact(tmp_path: Path) -> None:
    path = tmp_path / "benchmark_result.json"
    _ = path.write_text(json.dumps({"schema": "wrong"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a synthetic benchmark result"):
        _ = load_synthetic_benchmark_result(path)


def test_rejects_empty_comparison() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _ = synthetic_benchmark_comparison_markdown([])


def _write_result(
    path: Path,
    *,
    benchmark: str,
    status: str = "passed",
    criteria_status: str = "passed",
    geometry_passed: bool = True,
    volume_nmse: float = 0.1,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(
            {
                "schema": "tomojax.synthetic_benchmark_result.v1",
                "benchmark": benchmark,
                "implementation": "reimagined_align_auto_smoke",
                "profile": "smoke32",
                "status": status,
                "runtime": {
                    "time_to_verified_geometry_seconds": 0.5,
                    "total_wall_seconds": 0.75,
                },
                "reconstruction": {
                    "volume_nmse": volume_nmse,
                    "final_residual": 0.0125,
                },
                "geometry_recovery": {
                    "passed": geometry_passed,
                },
                "benchmark_manifest_evaluation_summary": {
                    "status": criteria_status,
                    "passed": 1,
                    "failed": 0,
                    "not_evaluated": 0,
                    "total": 1,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _row_cells(markdown: str, benchmark: str) -> list[str]:
    for line in markdown.splitlines():
        if line.startswith(f"| {benchmark} |"):
            return [cell.strip() for cell in line.strip("|").split("|")]
    raise AssertionError(f"missing benchmark row {benchmark!r}")
