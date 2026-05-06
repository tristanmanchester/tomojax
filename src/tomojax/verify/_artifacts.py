"""Artifact-bundle validation for v2 runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import cast


@dataclass(frozen=True)
class ArtifactValidationIssue:
    """One artifact validation failure."""

    artifact: str
    reason: str


@dataclass(frozen=True)
class ArtifactValidationReport:
    """Validation result for a run artifact bundle."""

    run_dir: Path
    issues: tuple[ArtifactValidationIssue, ...]

    @property
    def passed(self) -> bool:
        """Return whether the artifact bundle passed validation."""
        return not self.issues


class ArtifactValidationError(RuntimeError):
    """Raised when a required run artifact is missing or malformed."""

    def __init__(self, report: ArtifactValidationReport) -> None:
        self.report = report
        details = "; ".join(f"{issue.artifact}: {issue.reason}" for issue in report.issues)
        super().__init__(f"artifact validation failed for {report.run_dir}: {details}")


_REQUIRED_JSON_SCHEMAS = {
    "artifact_index.json": "tomojax.artifact_index.v1",
    "backend_report.json": "tomojax.backend_report.v1",
    "failure_report.json": "tomojax.failure_report.v1",
    "observability_report.json": "tomojax.observability_report.v1",
    "run_manifest.json": "tomojax.run_manifest.v1",
    "verification.json": "tomojax.alternating_smoke.verification.v1",
}


def validate_run_artifacts(run_dir: str | Path) -> ArtifactValidationReport:
    """Validate required v2 run artifacts and raise on failure."""
    path = Path(run_dir)
    report = inspect_run_artifacts(path)
    if not report.passed:
        raise ArtifactValidationError(report)
    return report


def inspect_run_artifacts(run_dir: str | Path) -> ArtifactValidationReport:
    """Inspect required v2 run artifacts without raising."""
    path = Path(run_dir)
    issues: list[ArtifactValidationIssue] = []
    payloads: dict[str, dict[str, object]] = {}
    for filename, schema in _REQUIRED_JSON_SCHEMAS.items():
        payload = _load_json_object(path / filename, filename, issues)
        if payload is None:
            continue
        payloads[filename] = payload
        if payload.get("schema") != schema:
            issues.append(
                ArtifactValidationIssue(
                    artifact=filename,
                    reason=f"expected schema {schema!r}",
                )
            )
    _validate_artifact_index(path, payloads.get("artifact_index.json"), issues)
    geometry_payload = _load_json_object(
        path / "geometry_final.json", "geometry_final.json", issues
    )
    _validate_geometry_final(geometry_payload, issues)
    _validate_manifest(payloads.get("run_manifest.json"), issues)
    _validate_verification(payloads.get("verification.json"), issues)
    _validate_backend_report(payloads.get("backend_report.json"), issues)
    _validate_failure_report(payloads.get("failure_report.json"), issues)
    _validate_observability(payloads.get("observability_report.json"), issues)
    benchmark_payload = _load_optional_json_object(
        path / "benchmark_result.json",
        "benchmark_result.json",
        issues,
    )
    _validate_benchmark_result(benchmark_payload, issues)
    _validate_benchmark_report(path / "benchmark_report.md", benchmark_payload, issues)
    return ArtifactValidationReport(run_dir=path, issues=tuple(issues))


def _load_json_object(
    path: Path,
    artifact: str,
    issues: list[ArtifactValidationIssue],
) -> dict[str, object] | None:
    if not path.exists():
        issues.append(ArtifactValidationIssue(artifact=artifact, reason="missing"))
        return None
    try:
        raw_payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        issues.append(
            ArtifactValidationIssue(
                artifact=artifact,
                reason=f"invalid JSON at line {exc.lineno}",
            )
        )
        return None
    if not isinstance(raw_payload, dict):
        issues.append(ArtifactValidationIssue(artifact=artifact, reason="not a JSON object"))
        return None
    return cast("dict[str, object]", raw_payload)


def _load_optional_json_object(
    path: Path,
    artifact: str,
    issues: list[ArtifactValidationIssue],
) -> dict[str, object] | None:
    if not path.exists():
        return None
    return _load_json_object(path, artifact, issues)


def _validate_artifact_index(
    run_dir: Path,
    payload: dict[str, object] | None,
    issues: list[ArtifactValidationIssue],
) -> None:
    if payload is None:
        return
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list):
        issues.append(
            ArtifactValidationIssue(
                artifact="artifact_index.json",
                reason="artifacts must be a list",
            )
        )
        return
    artifacts = cast("list[object]", raw_artifacts)
    for index, item in enumerate(artifacts):
        if not isinstance(item, dict):
            issues.append(
                ArtifactValidationIssue(
                    artifact="artifact_index.json",
                    reason=f"artifact entry {index} is not an object",
                )
            )
            continue
        entry = cast("dict[str, object]", item)
        name = entry.get("name")
        relative_path = entry.get("path")
        artifact_label = str(name) if isinstance(name, str) else f"entry {index}"
        if not isinstance(relative_path, str) or not relative_path:
            issues.append(
                ArtifactValidationIssue(
                    artifact=artifact_label,
                    reason="indexed path missing",
                )
            )
            continue
        if not (run_dir / relative_path).is_file():
            issues.append(
                ArtifactValidationIssue(
                    artifact=artifact_label,
                    reason=f"indexed file {relative_path!r} missing",
                )
            )


def _validate_manifest(
    payload: dict[str, object] | None,
    issues: list[ArtifactValidationIssue],
) -> None:
    if payload is None:
        return
    _append_missing_keys(
        payload,
        issues,
        artifact="run_manifest.json",
        keys=(
            "tomojax_version",
            "git_commit",
            "run_id",
            "started_at",
            "finished_at",
            "profile",
            "align_mode",
            "dataset",
            "geometry_model",
            "backend_requested",
            "backend_actual",
            "status",
        ),
    )


def _validate_geometry_final(
    payload: dict[str, object] | None,
    issues: list[ArtifactValidationIssue],
) -> None:
    if payload is None:
        return
    if payload.get("schema_version") != 1:
        issues.append(
            ArtifactValidationIssue(
                artifact="geometry_final.json",
                reason="expected schema_version 1",
            )
        )
    _append_missing_keys(
        payload,
        issues,
        artifact="geometry_final.json",
        keys=("setup", "pose"),
    )


def _validate_verification(
    payload: dict[str, object] | None,
    issues: list[ArtifactValidationIssue],
) -> None:
    if payload is None:
        return
    _append_missing_keys(
        payload,
        issues,
        artifact="verification.json",
        keys=(
            "status",
            "summary",
            "metrics",
            "escalation",
            "initial_loss",
            "final_loss",
            "levels",
            "geometry_recovery",
        ),
    )


def _validate_backend_report(
    payload: dict[str, object] | None,
    issues: list[ArtifactValidationIssue],
) -> None:
    if payload is None:
        return
    _append_missing_keys(
        payload,
        issues,
        artifact="backend_report.json",
        keys=(
            "requested",
            "actual",
            "actual_projector",
            "actual_backprojector",
            "actual_geometry_reductions",
            "canonical_detector_grid",
            "calibrated_detector_grid",
            "pallas_eligible",
            "fallbacks",
            "agreement_tests",
        ),
    )


def _validate_observability(
    payload: dict[str, object] | None,
    issues: list[ArtifactValidationIssue],
) -> None:
    if payload is None:
        return
    _append_missing_keys(
        payload,
        issues,
        artifact="observability_report.json",
        keys=("status", "dofs", "weak_modes", "handled_frozen_dofs"),
    )


def _validate_failure_report(
    payload: dict[str, object] | None,
    issues: list[ArtifactValidationIssue],
) -> None:
    if payload is None:
        return
    _append_missing_keys(
        payload,
        issues,
        artifact="failure_report.json",
        keys=("status", "failure", "failure_classes", "gates", "warnings"),
    )


def _validate_benchmark_result(
    payload: dict[str, object] | None,
    issues: list[ArtifactValidationIssue],
) -> None:
    if payload is None:
        return
    if payload.get("schema") != "tomojax.synthetic_benchmark_result.v1":
        issues.append(
            ArtifactValidationIssue(
                artifact="benchmark_result.json",
                reason="expected schema 'tomojax.synthetic_benchmark_result.v1'",
            )
        )
    _append_missing_keys(
        payload,
        issues,
        artifact="benchmark_result.json",
        keys=(
            "benchmark",
            "implementation",
            "profile",
            "status",
            "dataset",
            "runtime",
            "reconstruction",
            "geometry_recovery",
            "backend",
            "failure_labels",
            "benchmark_manifest_criteria",
            "benchmark_manifest_evaluation",
            "benchmark_manifest_evaluation_summary",
        ),
    )


def _validate_benchmark_report(
    path: Path,
    benchmark_payload: dict[str, object] | None,
    issues: list[ArtifactValidationIssue],
) -> None:
    if benchmark_payload is None:
        return
    if not path.is_file():
        issues.append(ArtifactValidationIssue(artifact="benchmark_report.md", reason="missing"))
        return
    text = path.read_text(encoding="utf-8")
    if "# Benchmark:" not in text:
        issues.append(
            ArtifactValidationIssue(
                artifact="benchmark_report.md",
                reason="missing benchmark title",
            )
        )
    if "## Benchmark Manifest Evaluation" not in text:
        issues.append(
            ArtifactValidationIssue(
                artifact="benchmark_report.md",
                reason="missing benchmark manifest evaluation section",
            )
        )


def _append_missing_keys(
    payload: dict[str, object],
    issues: list[ArtifactValidationIssue],
    *,
    artifact: str,
    keys: tuple[str, ...],
) -> None:
    issues.extend(
        ArtifactValidationIssue(artifact=artifact, reason=f"missing {key}")
        for key in keys
        if key not in payload
    )
