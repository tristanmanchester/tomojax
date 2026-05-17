from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tomojax.bench import (
    mark_real_lamino_stage_failed,
    real_lamino_loss_summary,
    real_lamino_safe_params_summary,
    write_real_lamino_planned_stage_manifests,
    write_real_lamino_skipped_stage_manifests,
)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_real_lamino_report_records_stage_failure_provenance(tmp_path: Path) -> None:
    stage_dir = tmp_path / "02_pose_phi"
    stage_dir.mkdir()
    (stage_dir / "stage_manifest.json").write_text(
        json.dumps({"stage": "02_pose_phi", "status": "completed"}) + "\n",
        encoding="utf-8",
    )
    validation = {
        "schema": "tomojax.real_lamino_stage_validation.v1",
        "stage": "02_pose_phi",
        "passed": False,
        "failures": ["non-finite reconstruction"],
    }

    mark_real_lamino_stage_failed(
        stage_dir,
        stage_name="02_pose_phi",
        validation=validation,
    )

    manifest = _read_json(stage_dir / "stage_manifest.json")
    provenance = _read_json(stage_dir / "failure_provenance.json")
    assert manifest["status"] == "failed"
    assert manifest["failure_provenance"] == validation
    assert provenance == validation


def test_real_lamino_report_summarizes_finite_loss_values() -> None:
    assert real_lamino_loss_summary({"loss": [3.0, 2.0], "effective_iters": 7}) == {
        "first": 3.0,
        "last": 2.0,
        "iters": 7,
    }
    assert real_lamino_loss_summary({"loss": [float("nan"), float("inf")]}) == {
        "first": None,
        "last": None,
        "iters": 2,
    }
    assert real_lamino_loss_summary({}) == {"first": None, "last": None, "iters": 0}


def test_real_lamino_report_writes_skipped_stage_manifests_without_overwrite(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "02_pose_phi"
    existing.mkdir()
    (existing / "stage_manifest.json").write_text(
        json.dumps({"stage": "02_pose_phi", "status": "completed"}) + "\n",
        encoding="utf-8",
    )

    write_real_lamino_skipped_stage_manifests(
        tmp_path,
        stages=["02_pose_phi", "03_pose_dx_dz"],
        reason="upstream stage failed validation",
    )

    preserved = _read_json(existing / "stage_manifest.json")
    skipped = _read_json(tmp_path / "03_pose_dx_dz" / "stage_manifest.json")
    assert preserved["status"] == "completed"
    assert skipped["status"] == "skipped"
    assert skipped["skip_reason"] == "upstream stage failed validation"
    assert skipped["failure_provenance"] == {
        "schema": "tomojax.real_lamino_stage_validation.v1",
        "stage": "03_pose_dx_dz",
        "passed": False,
        "failures": ["upstream stage failed validation"],
    }


def test_real_lamino_report_writes_planned_stage_manifests_without_overwrite(
    tmp_path: Path,
) -> None:
    staged_path = (
        {"label": "baseline", "stage": "00_baseline", "active_dofs": [], "status": "required"},
        {
            "label": "pose_phi",
            "stage": "02_pose_phi",
            "active_dofs": ["phi"],
            "status": "planned",
        },
        {
            "label": "final",
            "stage": "05_final",
            "active_dofs": ["pose"],
            "status": "planned",
        },
    )
    existing = tmp_path / "02_pose_phi"
    existing.mkdir()
    (existing / "stage_manifest.json").write_text(
        json.dumps({"stage": "02_pose_phi", "status": "completed"}) + "\n",
        encoding="utf-8",
    )

    write_real_lamino_planned_stage_manifests(
        tmp_path,
        staged_path=staged_path,
        planned_after="after cor",
    )

    preserved = _read_json(existing / "stage_manifest.json")
    planned = _read_json(tmp_path / "05_final" / "stage_manifest.json")
    assert not (tmp_path / "00_baseline" / "stage_manifest.json").exists()
    assert preserved["status"] == "completed"
    assert planned == {
        "stage": "05_final",
        "label": "final",
        "status": "planned",
        "active_dofs": ["pose"],
        "planned_after": "after cor",
    }


def test_real_lamino_safe_params_summary_rejects_nonfinite_values() -> None:
    calls = 0

    def summarize(params: np.ndarray) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"shape": list(params.shape), "dtype": str(params.dtype)}

    assert real_lamino_safe_params_summary(
        np.array([[1.0, np.nan, 0.0, 0.0, 0.0]], dtype=np.float32),
        summarize=summarize,
    ) is None
    assert calls == 0

    summary = real_lamino_safe_params_summary(
        np.zeros((2, 5), dtype=np.float64),
        summarize=summarize,
    )
    assert summary == {"shape": [2, 5], "dtype": "float32"}
    assert calls == 1
