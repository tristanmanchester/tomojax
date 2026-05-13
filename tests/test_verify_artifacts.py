from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tomojax.bench import (
    AlternatingAlignmentSolver,
    AlternatingSmokeConfig,
    run_alternating_solver_smoke,
)
from tomojax.datasets import generate_synthetic_dataset
from tomojax.verify import (
    ArtifactValidationError,
    inspect_run_artifacts,
    validate_run_artifacts,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_validate_run_artifacts_accepts_smoke_bundle(tmp_path: Path) -> None:
    result = run_alternating_solver_smoke(tmp_path)

    report = validate_run_artifacts(tmp_path)

    assert report.passed
    assert report.run_dir == tmp_path
    assert inspect_run_artifacts(tmp_path).passed
    assert result.artifacts["artifact_index_json"].exists()


def test_validate_run_artifacts_reports_missing_indexed_file(tmp_path: Path) -> None:
    result = run_alternating_solver_smoke(tmp_path)
    result.artifacts["verification_json"].unlink()

    report = inspect_run_artifacts(tmp_path)

    assert not report.passed
    assert any(
        issue.artifact == "verification.json" and "missing" in issue.reason
        for issue in report.issues
    )
    with pytest.raises(ArtifactValidationError) as exc_info:
        _ = validate_run_artifacts(tmp_path)
    assert exc_info.value.report.issues == report.issues


def test_validate_run_artifacts_checks_optional_benchmark_result(tmp_path: Path) -> None:
    dataset_paths = generate_synthetic_dataset(
        "synth128_setup_global_tomo",
        tmp_path / "datasets",
        size=32,
        clean=True,
        views=4,
    )
    solver = AlternatingAlignmentSolver(
        AlternatingSmokeConfig(
            synthetic_dataset_name="synth128_setup_global_tomo",
            synthetic_dataset_artifact_dir=dataset_paths.dataset_dir,
        )
    )
    _ = solver.run_smoke(tmp_path / "run")
    report = validate_run_artifacts(tmp_path / "run")
    assert report.passed

    benchmark_result = tmp_path / "run" / "benchmark_result.json"
    text = benchmark_result.read_text(encoding="utf-8")
    _ = benchmark_result.write_text(
        text.replace('"benchmark":', '"benchmark_missing":', 1),
        encoding="utf-8",
    )

    broken_report = inspect_run_artifacts(tmp_path / "run")

    assert not broken_report.passed
    assert any(
        issue.artifact == "benchmark_result.json" and issue.reason == "missing benchmark"
        for issue in broken_report.issues
    )


def test_validate_run_artifacts_requires_report_for_benchmark_result(tmp_path: Path) -> None:
    dataset_paths = generate_synthetic_dataset(
        "synth128_setup_global_tomo",
        tmp_path / "datasets",
        size=32,
        clean=True,
        views=4,
    )
    solver = AlternatingAlignmentSolver(
        AlternatingSmokeConfig(
            synthetic_dataset_name="synth128_setup_global_tomo",
            synthetic_dataset_artifact_dir=dataset_paths.dataset_dir,
        )
    )
    _ = solver.run_smoke(tmp_path / "run")
    (tmp_path / "run" / "benchmark_report.md").unlink()

    report = inspect_run_artifacts(tmp_path / "run")

    assert not report.passed
    assert any(
        issue.artifact == "benchmark_report.md" and issue.reason == "missing"
        for issue in report.issues
    )
