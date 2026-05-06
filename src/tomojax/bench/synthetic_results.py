"""Synthetic benchmark result ingestion and comparison reports."""
# pyright: reportAny=false

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import cast


@dataclass(frozen=True)
class SyntheticBenchmarkResult:
    """Loaded synthetic benchmark result artifact."""

    path: Path
    payload: dict[str, object]


def load_synthetic_benchmark_result(path: str | Path) -> SyntheticBenchmarkResult:
    """Load and validate one `benchmark_result.json` artifact."""
    result_path = Path(path)
    payload = cast("dict[str, object]", json.loads(result_path.read_text(encoding="utf-8")))
    _validate_synthetic_benchmark_result(payload, result_path)
    return SyntheticBenchmarkResult(path=result_path, payload=payload)


def load_synthetic_benchmark_results(
    paths: list[str | Path] | tuple[str | Path, ...],
) -> tuple[SyntheticBenchmarkResult, ...]:
    """Load and validate multiple synthetic benchmark result artifacts."""
    return tuple(load_synthetic_benchmark_result(path) for path in paths)


def synthetic_benchmark_comparison_markdown(
    results: tuple[SyntheticBenchmarkResult, ...] | list[SyntheticBenchmarkResult],
) -> str:
    """Render a deterministic markdown comparison over loaded benchmark results."""
    if not results:
        raise ValueError("at least one synthetic benchmark result is required")
    sorted_results = sorted(
        results,
        key=lambda result: (
            str(result.payload.get("benchmark", "")),
            str(result.payload.get("implementation", "")),
            str(result.payload.get("profile", "")),
            result.path.as_posix(),
        ),
    )
    rows = [
        "| Benchmark | Impl | Profile | Status | Criteria | Geometry | Volume NMSE | "
        "Final residual | Time to verified | Total time | Source |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    rows.extend(_comparison_row(result) for result in sorted_results)
    return "\n".join(["# Synthetic Benchmark Comparison", "", *rows, ""])


def write_synthetic_benchmark_comparison_markdown(
    results: tuple[SyntheticBenchmarkResult, ...] | list[SyntheticBenchmarkResult],
    path: str | Path,
) -> Path:
    """Write a synthetic benchmark markdown comparison report."""
    out_path = Path(path)
    _ = out_path.write_text(synthetic_benchmark_comparison_markdown(results), encoding="utf-8")
    return out_path


def _validate_synthetic_benchmark_result(payload: dict[str, object], path: Path) -> None:
    if payload.get("schema") != "tomojax.synthetic_benchmark_result.v1":
        raise ValueError(f"{path} is not a synthetic benchmark result artifact")
    missing = [
        key
        for key in (
            "benchmark",
            "implementation",
            "profile",
            "status",
            "runtime",
            "reconstruction",
            "geometry_recovery",
            "benchmark_manifest_evaluation_summary",
        )
        if key not in payload
    ]
    if missing:
        raise ValueError(f"{path} missing required synthetic benchmark fields: {missing}")


def _comparison_row(result: SyntheticBenchmarkResult) -> str:
    payload = result.payload
    runtime = _mapping(payload.get("runtime"))
    reconstruction = _mapping(payload.get("reconstruction"))
    geometry = _mapping(payload.get("geometry_recovery"))
    criteria_summary = _mapping(payload.get("benchmark_manifest_evaluation_summary"))
    criteria_status = criteria_summary.get("status", "")
    geometry_status = "passed" if geometry.get("passed") is True else "failed"
    return (
        "| "
        + " | ".join(
            [
                _markdown_cell(payload.get("benchmark")),
                _markdown_cell(payload.get("implementation")),
                _markdown_cell(payload.get("profile")),
                _markdown_cell(payload.get("status")),
                _markdown_cell(criteria_status),
                _markdown_cell(geometry_status),
                _markdown_cell(reconstruction.get("volume_nmse")),
                _markdown_cell(reconstruction.get("final_residual")),
                _markdown_cell(runtime.get("time_to_verified_geometry_seconds")),
                _markdown_cell(runtime.get("total_wall_seconds")),
                _markdown_cell(result.path.as_posix()),
            ]
        )
        + " |"
    )


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): item for key, item in cast("dict[object, object]", value).items()}
    return {}


def _markdown_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")
