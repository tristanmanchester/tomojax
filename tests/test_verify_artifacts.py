from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tomojax.align.api import run_alternating_solver_smoke
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
