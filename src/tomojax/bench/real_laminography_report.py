"""Report helpers for real-laminography benchmark runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import csv
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from tomojax.bench.real_laminography_profiles import (
    REAL_LAMINO_STAGED_PATH,
    REFERENCE_REGRESSION_STAGE_MAP,
)
from tomojax.io import read_json_object, write_json_object

REAL_LAMINO_REPORT_STAGED_PATH: tuple[tuple[str, str], ...] = (
    ("baseline", "00_baseline"),
    ("cor_detu", "01_setup_geometry/01_cor"),
    ("detector_roll", "01_setup_geometry/02_detector_roll"),
    ("axis_direction", "01_setup_geometry/03_axis_direction"),
    ("pose_phi", "02_pose_phi"),
    ("pose_dx_dz", "03_pose_dx_dz"),
    ("pose_5dof_polish", "04_pose_polish"),
    ("final_reconstruction", "05_final"),
)
REAL_LAMINO_COR_ONLY_STAGE = "06_cor_only_fista"
REAL_LAMINO_PUBLICATION_IMAGES: tuple[tuple[str, str, str], ...] = (
    ("before", "00_baseline", "orthos.png"),
    ("before_xy", "00_baseline", "aligned_xy_global_z209.png"),
    ("cor_only", REAL_LAMINO_COR_ONLY_STAGE, "orthos.png"),
    ("cor_only_xy", REAL_LAMINO_COR_ONLY_STAGE, "aligned_xy_global_z209.png"),
    ("full", "05_final", "orthos.png"),
    ("full_xy", "05_final", "aligned_xy_global_z209.png"),
    ("full_delta_xy", "05_final", "delta_xy_global_z209.png"),
)

PARTIAL_REQUIRED_STAGES = frozenset(
    ("00_baseline", "01_setup_geometry/01_cor", REAL_LAMINO_COR_ONLY_STAGE)
)
FULL_REQUIRED_STAGES = frozenset(
    (
        "00_baseline",
        "01_setup_geometry/01_cor",
        "01_setup_geometry/02_detector_roll",
        "01_setup_geometry/03_axis_direction",
        "02_pose_phi",
        "03_pose_dx_dz",
        "04_pose_polish",
        "05_final",
        REAL_LAMINO_COR_ONLY_STAGE,
    )
)


def real_lamino_method_constraints() -> dict[str, Any]:
    return {
        "cor_grid_search_added": False,
        "sinogram_or_correlation_method_added": False,
        "sharpness_or_autofocus_method_added": False,
        "benchmark_only_knobs_promoted": False,
        "cor_only_role": "first v2 final reconstruction comparator",
    }


def real_lamino_success_payload(
    reconstruction: Mapping[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    completed = {
        str(record.get("stage")) for record in records if str(record.get("status")) == "completed"
    }
    planned = {
        str(record.get("stage")) for record in records if str(record.get("status")) == "planned"
    }
    failed_or_skipped = [
        record for record in records if str(record.get("status")) in {"failed", "skipped"}
    ]
    final_loss = reconstruction["final"]["loss"]["last"]
    cor_loss = reconstruction["cor_only"]["loss"]["last"]
    full_complete = completed >= FULL_REQUIRED_STAGES
    full_improved = (
        full_complete
        and final_loss is not None
        and cor_loss is not None
        and float(final_loss) < float(cor_loss)
        and bool(reconstruction.get("same_volume_shape"))
    )
    phase = (
        "v2_full_staged_failed_validation"
        if failed_or_skipped
        else "v2_full_staged"
        if full_complete
        else "v2_cor_only_partial"
    )
    partial_complete = completed >= PARTIAL_REQUIRED_STAGES and cor_loss is not None
    validation_failed = bool(failed_or_skipped)
    passed = bool((full_improved if full_complete else partial_complete) and not validation_failed)
    reason = (
        "v2 staged path failed validation; final report uses the last finite candidate"
        if validation_failed
        else "v2 full staged reconstruction improves COR-only FISTA loss"
        if full_improved
        else "v2 full staged reconstruction did not improve COR-only FISTA loss"
        if full_complete
        else "v2 partial path completed baseline, det_u setup, and COR-only FISTA"
        if partial_complete
        else "v2 path is missing required baseline/det_u/COR-only evidence"
    )
    return {
        "passed": passed,
        "reason": reason,
        "phase": phase,
        "quality_kind": (
            "real_reconstruction_quality"
            if full_complete
            else "real_reconstruction_quality_partial_cor_only"
        ),
        "primary_metric": (
            "final_fista_last_loss_lt_cor_only_fista_last_loss"
            if full_complete
            else "cor_only_fista_loss_recorded_after_v2_det_u_stage"
        ),
        "required_stages_completed": sorted(
            (FULL_REQUIRED_STAGES if full_complete else PARTIAL_REQUIRED_STAGES) & completed
        ),
        "planned_stages": sorted(planned),
        "final_loss": final_loss,
        "cor_only_loss": cor_loss,
        "loss_improvement_abs": reconstruction.get("loss_improvement_abs"),
        "loss_improvement_rel": reconstruction.get("loss_improvement_rel"),
        "same_volume_shape": bool(reconstruction.get("same_volume_shape")),
        "full_staged_success_deferred": not full_complete,
        "validation_failed": validation_failed,
        "failed_or_skipped_stages": [
            {
                "stage": record.get("stage"),
                "status": record.get("status"),
                "failure_provenance": record.get("failure_provenance"),
                "skip_reason": record.get("skip_reason"),
            }
            for record in failed_or_skipped
        ],
    }


def write_real_lamino_residual_trace(path: Path, records: list[Mapping[str, Any]]) -> Path:
    fields = (
        "label",
        "stage",
        "status",
        "level_factor",
        "iteration",
        "loss_before",
        "loss_after",
        "accepted",
        "active_dofs",
        "elapsed_seconds",
    )
    rows: list[dict[str, Any]] = []
    for record in records:
        summary_rows = record.get("summary_rows", [])
        if not summary_rows:
            loss = record.get("reconstruction_loss") or {}
            rows.append(
                {
                    "label": record.get("label"),
                    "stage": record.get("stage"),
                    "status": record.get("status"),
                    "level_factor": "",
                    "iteration": "",
                    "loss_before": loss.get("first"),
                    "loss_after": loss.get("last"),
                    "accepted": "",
                    "active_dofs": ",".join(str(v) for v in record.get("active_dofs", [])),
                    "elapsed_seconds": record.get("elapsed_seconds"),
                }
            )
            continue
        active_dofs = ",".join(str(v) for v in record.get("active_dofs", []))
        rows.extend(
            {
                "label": record.get("label"),
                "stage": record.get("stage"),
                "status": record.get("status"),
                "level_factor": row.get("level_factor", ""),
                "iteration": row.get("outer_iter", row.get("outer_idx", "")),
                "loss_before": row.get("geometry_loss_before", row.get("loss_before", "")),
                "loss_after": row.get("geometry_loss_after", row.get("loss_after", "")),
                "accepted": row.get("geometry_accepted", row.get("accepted", "")),
                "active_dofs": row.get("active_dofs", active_dofs),
                "elapsed_seconds": row.get("elapsed_seconds", row.get("cumulative_time", "")),
            }
            for row in summary_rows
        )
    _write_csv(path, rows, fields)
    return path


def write_real_lamino_geometry_trace(path: Path, records: list[Mapping[str, Any]]) -> Path:
    _write_json(
        path,
        {
            "schema": "tomojax.real_lamino_geometry_trace.v1",
            "stages": [
                {
                    "label": record.get("label"),
                    "stage": record.get("stage"),
                    "status": record.get("status"),
                    "active_dofs": record.get("active_dofs", []),
                    "bounds": record.get("bounds"),
                    "geometry_calibration_state": record.get("geometry_calibration_state"),
                    "params_summary": record.get("params_summary"),
                    "planned_after": record.get("planned_after"),
                }
                for record in records
            ],
        },
    )
    return path


def real_lamino_finite_fraction(value: Any | None) -> float:
    if value is None:
        return 0.0
    arr = np.asarray(value)
    if arr.size == 0:
        return 0.0
    return float(np.isfinite(arr).mean())


def real_lamino_pose_params_summary(params5: Any) -> dict[str, Any]:
    """Summarize per-view 5-DOF pose parameters for real-laminography reports."""
    arr = np.asarray(params5, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError(f"expected pose params with shape (views, 5), got {arr.shape}")
    names = ("alpha", "beta", "phi", "dx", "dz")
    summary: dict[str, Any] = {}
    for idx, name in enumerate(names):
        col = arr[:, idx]
        summary[name] = {
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
        }
        if name in {"alpha", "beta", "phi"}:
            deg = np.rad2deg(col)
            summary[f"{name}_deg"] = {
                "min": float(np.min(deg)),
                "max": float(np.max(deg)),
                "mean": float(np.mean(deg)),
                "std": float(np.std(deg)),
            }
    return summary


def real_lamino_safe_params_summary(
    params5: Any,
    *,
    summarize: Callable[[np.ndarray], Mapping[str, Any]] = real_lamino_pose_params_summary,
) -> dict[str, Any] | None:
    """Summarize pose parameters only when every value is finite."""
    params = np.asarray(params5, dtype=np.float32)
    if real_lamino_finite_fraction(params) != 1.0:
        return None
    return dict(summarize(params))


def real_lamino_checkpoint_validation_failures(stage_dir: Path) -> list[str]:
    failures: list[str] = []
    checkpoint_dir = stage_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return failures
    for path in sorted(checkpoint_dir.glob("*.npz")):
        try:
            with np.load(path) as payload:
                if "x" not in payload:
                    failures.append(f"{path.name} missing x checkpoint array")
                    continue
                fraction = real_lamino_finite_fraction(payload["x"])
        except Exception as exc:
            failures.append(f"{path.name} could not be read: {type(exc).__name__}: {exc}")
            continue
        if fraction != 1.0:
            failures.append(f"{path.name} x finite fraction is {fraction:.6g}")
    return failures


def real_lamino_stat_validation_failures(
    stats: list[dict[str, Any]],
    *,
    require_data_loss: bool,
) -> list[str]:
    failures: list[str] = []
    for idx, stat in enumerate(stats):
        finite_reported_losses = [
            key
            for key in ("geometry_loss_before", "geometry_loss_after", "loss_before", "loss_after")
            if key in stat and _is_finite_scalar(stat.get(key))
        ]
        failures.extend(
            f"stat[{idx}] {key} is non-finite: {stat.get(key)!r}"
            for key in ("geometry_loss_before", "geometry_loss_after", "loss_before", "loss_after")
            if key in stat and not _is_finite_scalar(stat.get(key))
        )
        if (
            require_data_loss
            and stat.get("data_loss_computed") is False
            and not finite_reported_losses
        ):
            failures.append(
                f"stat[{idx}] data_loss_computed is false and no finite objective loss was reported"
            )
    return failures


def real_lamino_artifact_validation_failures(stage_dir: Path) -> list[str]:
    manifest_path = stage_dir / "stage_manifest.json"
    if not manifest_path.exists():
        return []
    manifest = _read_json(manifest_path)
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        return ["stage artifacts payload is missing or not an object"]
    failures: list[str] = []
    for key, raw_path in artifacts.items():
        path = _resolve_staged_artifact_path(stage_dir, raw_path)
        if not path.exists():
            failures.append(f"artifact {key} is missing: {path}")
        elif path.stat().st_size <= 0:
            failures.append(f"artifact {key} is empty: {path}")
    return failures


def _is_finite_scalar(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def validate_real_lamino_stage_output(
    stage_dir: Path,
    *,
    stage_name: str,
    volume: Any | None,
    params5: Any | None,
    stats: list[dict[str, Any]],
    require_data_loss: bool,
) -> dict[str, Any]:
    failures: list[str] = []
    volume_fraction = real_lamino_finite_fraction(volume)
    if volume_fraction != 1.0:
        failures.append(f"reconstruction volume finite fraction is {volume_fraction:.6g}")
    params_fraction = real_lamino_finite_fraction(params5)
    if params_fraction != 1.0:
        failures.append(f"pose/setup params finite fraction is {params_fraction:.6g}")
    checkpoint_failures = real_lamino_checkpoint_validation_failures(stage_dir)
    failures.extend(checkpoint_failures)
    failures.extend(
        real_lamino_stat_validation_failures(stats, require_data_loss=require_data_loss)
    )
    artifact_failures = real_lamino_artifact_validation_failures(stage_dir)
    failures.extend(artifact_failures)
    return {
        "schema": "tomojax.real_lamino_stage_validation.v1",
        "stage": stage_name,
        "passed": not failures,
        "failures": failures,
        "volume_finite_fraction": volume_fraction,
        "params_finite_fraction": params_fraction,
        "checkpoint_failures": checkpoint_failures,
        "artifact_failures": artifact_failures,
        "require_data_loss": bool(require_data_loss),
    }


def real_lamino_loss_summary(info: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize first/last finite loss values from a staged reconstruction info payload."""
    losses = info.get("loss", [])
    if not isinstance(losses, list) or not losses:
        return {"first": None, "last": None, "iters": 0}
    first = float(losses[0])
    last = float(losses[-1])
    if not np.isfinite(first):
        first = None
    if not np.isfinite(last):
        last = None
    return {
        "first": first,
        "last": last,
        "iters": int(info.get("effective_iters", len(losses))),
    }


def mark_real_lamino_stage_failed(
    stage_dir: Path,
    *,
    stage_name: str,
    validation: Mapping[str, Any],
) -> None:
    """Persist fail-closed stage provenance for a staged real-laminography run."""
    manifest_path = stage_dir / "stage_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {"stage": stage_name}
    manifest["status"] = "failed"
    manifest["failure_provenance"] = dict(validation)
    _write_json(manifest_path, manifest)
    _write_json(stage_dir / "failure_provenance.json", dict(validation))


def write_real_lamino_skipped_stage_manifests(
    root: Path,
    *,
    stages: list[str],
    reason: str,
) -> None:
    """Write skipped-stage manifests without overwriting existing stage evidence."""
    for stage in stages:
        stage_dir = Path(root) / stage
        manifest_path = stage_dir / "stage_manifest.json"
        if manifest_path.exists():
            continue
        _write_json(
            manifest_path,
            {
                "stage": stage,
                "status": "skipped",
                "skip_reason": reason,
                "failure_provenance": {
                    "schema": "tomojax.real_lamino_stage_validation.v1",
                    "stage": stage,
                    "passed": False,
                    "failures": [reason],
                },
            },
        )


def write_real_lamino_planned_stage_manifests(
    root: Path,
    *,
    staged_path: tuple[Mapping[str, Any], ...] = REAL_LAMINO_STAGED_PATH,
    planned_after: str = "v2 COR-only path works",
) -> None:
    """Write planned-stage manifests for stages not yet executed by the staged runner."""
    for spec in staged_path:
        if spec.get("status") != "planned":
            continue
        stage_dir = root / str(spec["stage"])
        manifest_path = stage_dir / "stage_manifest.json"
        if manifest_path.exists():
            continue
        _write_json(
            manifest_path,
            {
                "stage": spec["stage"],
                "label": spec["label"],
                "status": "planned",
                "active_dofs": spec["active_dofs"],
                "planned_after": planned_after,
            },
        )


def build_real_lamino_report(
    run_dir: Path,
    *,
    out_dir: Path | None = None,
    reference_report: Path | None = None,
    require_success: bool = False,
) -> dict[str, Any]:
    """Write and return the real-laminography staged report for ``run_dir``."""
    root = Path(run_dir)
    if out_dir is None:
        out_dir = root / "real_lamino_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = _read_json(root / "run_manifest.json")
    status = _read_json(root / "status.json") if (root / "status.json").exists() else {}
    stage_records = [_staged_report_record(root, spec) for spec in REAL_LAMINO_STAGED_PATH]
    reconstruction = _staged_reconstruction_comparison(root)
    success = real_lamino_success_payload(reconstruction, stage_records)
    if require_success and not bool(success["passed"]):
        raise RuntimeError(str(success["reason"]))

    publication_artifacts = _copy_staged_publication_images(
        root,
        out_dir,
        full_completed=reconstruction["final"]["status"] == "completed"
        and reconstruction["final"]["loss"]["last"] is not None,
    )
    residual_trace = write_real_lamino_residual_trace(
        out_dir / "real_lamino_residual_trace.csv",
        stage_records,
    )
    geometry_trace = write_real_lamino_geometry_trace(
        out_dir / "real_lamino_geometry_trace.json",
        stage_records,
    )
    reference_regression = _write_reference_regression_audit(
        root=root,
        out_dir=out_dir,
        reference_report=reference_report,
        run_manifest=run_manifest,
    )

    summary: dict[str, Any] = {
        "schema": "tomojax.real_lamino_staged_report.v2",
        "contract_compatible_with": "tomojax.real_lamino_staged_report.v2",
        "run_dir": str(root),
        "reference_target_report": str(reference_report) if reference_report else None,
        "reference_case": root.name,
        "success": success,
        "quality_basis": {
            "kind": success["quality_kind"],
            "primary_metric": success["primary_metric"],
            "full_staged_primary_metric": "final_fista_last_loss_lt_cor_only_fista_last_loss",
            "full_staged_success_deferred": success["full_staged_success_deferred"],
            "truth_metrics": "not_applicable_real_data",
            "synthetic_truth_metrics_allowed": False,
        },
        "staged_path": stage_records,
        "reconstruction_comparison": reconstruction,
        "publication_artifacts": publication_artifacts,
        "reference_regression": reference_regression["payload"],
        "provenance": {
            "input": run_manifest.get("input"),
            "binning": run_manifest.get("binning"),
            "backend": run_manifest.get("backend"),
            "devices": run_manifest.get("devices"),
            "final_volume_shape": run_manifest.get("final_volume_shape"),
            "final_setup_estimates": run_manifest.get("final_setup_estimates"),
            "final_pose_summary": run_manifest.get("final_pose_summary"),
            "status_completed_at": status.get("completed_at", run_manifest.get("completed_at")),
        },
        "method_constraints": real_lamino_method_constraints(),
        "artifacts": {
            "summary_json": str((out_dir / "real_lamino_summary.json").resolve()),
            "summary_md": str((out_dir / "real_lamino_summary.md").resolve()),
            "residual_trace_csv": str(residual_trace.resolve()),
            "geometry_trace_json": str(geometry_trace.resolve()),
            "publication_dir": str((out_dir / "publication").resolve()),
            **reference_regression["artifacts"],
        },
    }
    _write_json(out_dir / "real_lamino_summary.json", summary)
    _write_staged_markdown(out_dir / "real_lamino_summary.md", summary)
    return summary


def _staged_report_record(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    stage_dir = root / str(spec["stage"])
    manifest_path = stage_dir / "stage_manifest.json"
    if not manifest_path.exists():
        return {
            "label": spec["label"],
            "stage": spec["stage"],
            "status": "missing",
            "active_dofs": spec.get("active_dofs", []),
        }
    manifest = _read_json(manifest_path)
    return {
        "label": spec["label"],
        "stage": spec["stage"],
        "status": manifest.get("status"),
        "active_dofs": manifest.get("active_dofs", spec.get("active_dofs", [])),
        "bounds": manifest.get("bounds"),
        "levels": manifest.get("levels", []),
        "elapsed_seconds": manifest.get("elapsed_seconds"),
        "stats_count": manifest.get("stats_count"),
        "geometry_calibration_state": manifest.get(
            "geometry_calibration_state",
            manifest.get("setup_state"),
        ),
        "params_summary": manifest.get("params_summary"),
        "reconstruction_loss": _staged_reconstruction_loss(manifest),
        "artifacts": _staged_artifacts(stage_dir, manifest),
        "summary_rows": _read_real_lamino_stage_summary(stage_dir / "stage_summary.csv"),
        "planned_after": manifest.get("planned_after"),
        "skip_reason": manifest.get("skip_reason"),
        "failure_provenance": manifest.get("failure_provenance"),
    }


def _staged_reconstruction_comparison(root: Path) -> dict[str, Any]:
    baseline_manifest = _read_json(root / "00_baseline" / "stage_manifest.json")
    cor_manifest = _read_json(root / REAL_LAMINO_COR_ONLY_STAGE / "stage_manifest.json")
    cor_loss = _staged_loss_summary(cor_manifest.get("fista_info", {}))
    final_path = root / "05_final" / "stage_manifest.json"
    final_manifest = _read_json(final_path) if final_path.exists() else {"status": "missing"}
    final_info = final_manifest.get("recon_info", {})
    if not isinstance(final_info, Mapping):
        final_info = {}
    final_loss = _staged_loss_summary(final_info)
    final_shape = final_manifest.get(
        "volume_shape",
        _read_json(root / "run_manifest.json").get("final_volume_shape"),
    )
    final_completed = final_manifest.get("status") == "completed" and final_loss["last"] is not None
    improvement = None
    relative = None
    if final_completed and cor_loss["last"] is not None:
        improvement = float(cor_loss["last"]) - float(final_loss["last"])
        relative = improvement / max(abs(float(cor_loss["last"])), 1.0e-12)
    return {
        "baseline": {
            "stage": "00_baseline",
            "volume_shape": baseline_manifest.get("volume_shape"),
            "loss": {"first": None, "last": None, "iters": 0},
            "role": "raw FBP visual/reference baseline",
        },
        "cor_only": {
            "stage": REAL_LAMINO_COR_ONLY_STAGE,
            "volume_shape": cor_manifest.get("volume_shape"),
            "loss": cor_loss,
            "regulariser": cor_manifest.get("fista_info", {}).get("regulariser"),
        },
        "final": {
            "stage": "05_final",
            "status": final_manifest.get("status"),
            "loss": final_loss,
            "volume_shape": final_shape,
            "regulariser": final_info.get("regulariser"),
        },
        "same_volume_shape": (
            (final_shape if final_completed else baseline_manifest.get("volume_shape"))
            == cor_manifest.get("volume_shape")
        ),
        "loss_improvement_abs": improvement,
        "loss_improvement_rel": relative,
        "full_staged_vs_cor_only_deferred": not final_completed,
    }


def _copy_staged_publication_images(
    root: Path, out_dir: Path, *, full_completed: bool
) -> dict[str, str]:
    pub_dir = out_dir / "publication"
    pub_dir.mkdir(parents=True, exist_ok=True)
    images = [
        ("before", "00_baseline", "orthos.png"),
        ("before_xy", "00_baseline", "aligned_xy_global_z209.png"),
        ("cor_only", REAL_LAMINO_COR_ONLY_STAGE, "orthos.png"),
        ("cor_only_xy", REAL_LAMINO_COR_ONLY_STAGE, "aligned_xy_global_z209.png"),
    ]
    if full_completed:
        images.extend(
            [
                ("full", "05_final", "orthos.png"),
                ("full_xy", "05_final", "aligned_xy_global_z209.png"),
                ("full_delta_xy", "05_final", "delta_xy_global_z209.png"),
            ]
        )
    copied: dict[str, str] = {}
    for label, stage, filename in images:
        source = root / stage / filename
        if not source.exists():
            raise FileNotFoundError(f"missing publication artifact {source}")
        dest = pub_dir / f"{label}_{filename}"
        shutil.copy2(source, dest)
        copied[label] = str(dest.resolve())
    return copied


def _write_reference_regression_audit(
    *,
    root: Path,
    out_dir: Path,
    reference_report: Path | None,
    run_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    workflow = run_manifest.get("workflow", {})
    enabled = bool(isinstance(workflow, Mapping) and workflow.get("reference_regression"))
    if reference_report is None or not enabled:
        return {
            "payload": {
                "enabled": enabled,
                "status": "skipped",
                "reason": "reference-regression mode and reference report are required",
            },
            "artifacts": {},
        }
    reference_root = Path(reference_report).resolve().parents[1]
    rows = _build_reference_regression_rows(reference_root=reference_root, v2_root=root)
    csv_path = out_dir / "real_lamino_reference_regression_table.csv"
    fields = (
        "stage",
        "level_factor",
        "iteration",
        "reference_loss_before",
        "reference_loss_after",
        "current_loss_before",
        "current_loss_after",
        "loss_scale_ratio_after",
        "status",
        "notes",
    )
    _write_csv(csv_path, rows, fields)
    pose_scale_failures = [
        row
        for row in rows
        if str(row["stage"]).startswith(("02_pose", "03_pose", "04_pose"))
        and row["status"] == "loss_scale_mismatch"
    ]
    row_shape_failures = [
        row for row in rows if row["status"] in {"missing_reference_row", "missing_current_row"}
    ]
    contract = (
        workflow.get("reference_regression_contract", {}) if isinstance(workflow, Mapping) else {}
    )
    payload = {
        "schema": "tomojax.real_lamino_reference_regression.v2",
        "enabled": True,
        "status": "failed" if pose_scale_failures or row_shape_failures else "recorded",
        "source_reference_run": str(reference_root),
        "source_script": "scripts/real_laminography/run_real_lamino_staged.py",
        "contract": contract,
        "pose_loss_scale_failures": pose_scale_failures,
        "row_shape_failures": row_shape_failures,
        "stage_summaries": _reference_regression_stage_summaries(reference_root, root),
        "table_csv": str(csv_path.resolve()),
    }
    json_path = out_dir / "real_lamino_reference_regression.json"
    _write_json(json_path, payload)
    return {
        "payload": payload,
        "artifacts": {
            "reference_regression_table_csv": str(csv_path.resolve()),
            "reference_regression_json": str(json_path.resolve()),
        },
    }


def _build_reference_regression_rows(
    *,
    reference_root: Path,
    v2_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reference_stage, current_stage in REFERENCE_REGRESSION_STAGE_MAP:
        if reference_stage in {"05_final", REAL_LAMINO_COR_ONLY_STAGE}:
            reference_rows = _reconstruction_loss_rows_for_stage(reference_root / reference_stage)
            current_rows = _reconstruction_loss_rows_for_stage(v2_root / current_stage)
        else:
            reference_rows = _loss_rows_for_stage(reference_root / reference_stage)
            current_rows = _loss_rows_for_stage(v2_root / current_stage)
        keys = sorted(set(reference_rows) | set(current_rows), key=_reference_row_sort_key)
        for key in keys:
            reference = reference_rows.get(key, {})
            current = current_rows.get(key, {})
            ratio = _loss_scale_ratio(reference.get("loss_after"), current.get("loss_after"))
            status = "matched"
            notes = ""
            if not reference:
                status = "missing_reference_row"
            elif not current:
                status = "missing_current_row"
            elif (
                reference_stage.startswith(("02_pose", "03_pose", "04_pose"))
                and ratio is not None
                and (ratio > 10.0 or ratio < 0.1)
            ):
                status = "loss_scale_mismatch"
                notes = "pose loss scale differs by more than 10x"
            rows.append(
                {
                    "stage": current_stage,
                    "level_factor": key[0],
                    "iteration": key[1],
                    "reference_loss_before": reference.get("loss_before", ""),
                    "reference_loss_after": reference.get("loss_after", ""),
                    "current_loss_before": current.get("loss_before", ""),
                    "current_loss_after": current.get("loss_after", ""),
                    "loss_scale_ratio_after": "" if ratio is None else ratio,
                    "status": status,
                    "notes": notes,
                }
            )
    return rows


def _loss_rows_for_stage(stage_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    summary_rows = _read_real_lamino_stage_summary(stage_dir / "stage_summary.csv")
    if summary_rows:
        return {
            (
                str(row.get("level_factor", "")),
                str(row.get("outer_iter", row.get("outer_idx", ""))),
            ): {
                "loss_before": row.get("geometry_loss_before", row.get("loss_before", "")),
                "loss_after": row.get("geometry_loss_after", row.get("loss_after", "")),
            }
            for row in summary_rows
        }
    manifest_path = stage_dir / "stage_manifest.json"
    if not manifest_path.exists():
        return {}
    loss = _staged_reconstruction_loss(_read_json(manifest_path))
    if not loss:
        return {}
    return {
        ("final", ""): {
            "loss_before": loss.get("first", ""),
            "loss_after": loss.get("last", ""),
        }
    }


def _reconstruction_loss_rows_for_stage(stage_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    manifest_path = stage_dir / "stage_manifest.json"
    if not manifest_path.exists():
        return {}
    loss = _staged_reconstruction_loss(_read_json(manifest_path))
    if not loss:
        return {}
    return {
        ("final", ""): {
            "loss_before": loss.get("first", ""),
            "loss_after": loss.get("last", ""),
        }
    }


def _reference_row_sort_key(key: tuple[str, str]) -> tuple[int, int, str, str]:
    level, iteration = key
    try:
        level_i = int(level)
    except ValueError:
        level_i = 10**9
    try:
        iter_i = int(iteration)
    except ValueError:
        iter_i = 10**9
    return (level_i, iter_i, level, iteration)


def _loss_scale_ratio(reference_loss: Any, current_loss: Any) -> float | None:
    try:
        reference = float(reference_loss)
        current = float(current_loss)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(reference) and np.isfinite(current)) or abs(reference) <= 1e-12:
        return None
    return float(current / reference)


def _reference_regression_stage_summaries(
    reference_root: Path,
    v2_root: Path,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for reference_stage, current_stage in REFERENCE_REGRESSION_STAGE_MAP:
        reference_manifest = _read_json(reference_root / reference_stage / "stage_manifest.json")
        current_manifest_path = v2_root / current_stage / "stage_manifest.json"
        current_manifest = (
            _read_json(current_manifest_path) if current_manifest_path.exists() else {}
        )
        summaries.append(
            {
                "stage": current_stage,
                "reference": _reference_manifest_summary(reference_manifest),
                "current": _reference_manifest_summary(current_manifest),
            }
        )
    return summaries


def _reference_manifest_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": manifest.get("status"),
        "active_dofs": manifest.get("active_dofs"),
        "bounds": manifest.get("bounds"),
        "levels": manifest.get("levels"),
        "geometry_calibration_state": manifest.get("geometry_calibration_state"),
        "params_summary": manifest.get("params_summary"),
        "reconstruction_loss": _staged_reconstruction_loss(manifest),
    }


def _write_staged_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    success = summary["success"]
    reconstruction = summary["reconstruction_comparison"]
    lines = [
        "# Real Laminography Staged Report",
        "",
        f"- Reference target report: `{summary['reference_target_report']}`",
        f"- Phase complete: `{success['passed']}`",
        f"- Criterion: {success['reason']}",
        f"- Final loss: `{success['final_loss']}`",
        f"- COR-only loss: `{success['cor_only_loss']}`",
        f"- Full staged success deferred: `{success['full_staged_success_deferred']}`",
        "",
        "| Stage | Active DOFs | Status |",
        "|---|---|---|",
    ]
    for record in summary["staged_path"]:
        dofs = ",".join(str(v) for v in record.get("active_dofs", []))
        lines.append(f"| `{record['stage']}` | `{dofs}` | `{record.get('status')}` |")
    lines.extend(
        [
            "",
            "## Reconstruction Comparison",
            "",
            "| Comparator | First FISTA loss | Last FISTA loss | Effective iterations |",
            "|---|---:|---:|---:|",
            "| Baseline FBP |  |  | 0 |",
            "| COR-only | {first} | {last} | {iters} |".format(
                **reconstruction["cor_only"]["loss"]
            ),
            "| Full staged final | {first} | {last} | {iters} |".format(
                **reconstruction["final"]["loss"]
            ),
            "",
            "## Artifacts",
            "",
            f"- Summary JSON: `{summary['artifacts']['summary_json']}`",
            f"- Residual trace CSV: `{summary['artifacts']['residual_trace_csv']}`",
            f"- Geometry trace JSON: `{summary['artifacts']['geometry_trace_json']}`",
            f"- Publication image directory: `{summary['artifacts']['publication_dir']}`",
            "- Reference-regression table CSV: "
            f"`{summary['artifacts'].get('reference_regression_table_csv', '')}`",
            "",
            "Truth metrics are intentionally not used for this real-data gate.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _staged_artifacts(stage_dir: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    raw = manifest.get("artifacts", {})
    if not isinstance(raw, Mapping):
        return {}
    artifacts: dict[str, str] = {}
    for key, value in raw.items():
        artifacts[str(key)] = str(_resolve_staged_artifact_path(stage_dir, value))
    return artifacts


def _resolve_staged_artifact_path(stage_dir: Path, raw_path: object) -> Path:
    candidate = Path(str(raw_path))
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return stage_dir / candidate


def _staged_reconstruction_loss(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    info = manifest.get("recon_info", manifest.get("fista_info"))
    if not isinstance(info, Mapping):
        return None
    return _staged_loss_summary(info)


def _staged_loss_summary(info: Mapping[str, Any]) -> dict[str, Any]:
    return real_lamino_loss_summary(info)


def _collect_real_lamino_stage_records(root: Path) -> list[dict[str, Any]]:
    records = [
        _real_lamino_stage_record(root, label=label, rel=rel, required=True)
        for label, rel in REAL_LAMINO_REPORT_STAGED_PATH
    ]
    records.append(
        _real_lamino_stage_record(
            root,
            label="cor_only_comparator",
            rel=REAL_LAMINO_COR_ONLY_STAGE,
            required=True,
        )
    )
    return records


def _real_lamino_stage_record(
    root: Path,
    *,
    label: str,
    rel: str,
    required: bool,
) -> dict[str, Any]:
    stage_dir = root / rel
    manifest_path = stage_dir / "stage_manifest.json"
    if not manifest_path.exists():
        if required:
            raise FileNotFoundError(f"missing stage manifest: {manifest_path}")
        return {"label": label, "stage": rel, "status": "missing"}
    manifest = _read_json(manifest_path)
    return {
        "label": label,
        "stage": rel,
        "status": manifest.get("status"),
        "active_dofs": manifest.get("active_dofs", []),
        "bounds": manifest.get("bounds"),
        "levels": manifest.get("levels", []),
        "elapsed_seconds": manifest.get("elapsed_seconds"),
        "stats_count": manifest.get("stats_count"),
        "geometry_calibration_state": manifest.get(
            "geometry_calibration_state",
            manifest.get("setup_state"),
        ),
        "params_summary": manifest.get("params_summary"),
        "reconstruction_loss": _real_lamino_stage_reconstruction_loss(manifest),
        "artifacts": _real_lamino_stage_artifacts(root, stage_dir, manifest),
        "summary_rows": _read_real_lamino_stage_summary(stage_dir / "stage_summary.csv"),
    }


def _real_lamino_reconstruction_comparison(root: Path) -> dict[str, Any]:
    run_manifest = _read_json(root / "run_manifest.json")
    final_manifest = _read_json(root / "05_final" / "stage_manifest.json")
    cor_manifest = _read_json(root / REAL_LAMINO_COR_ONLY_STAGE / "stage_manifest.json")
    final_loss = _real_lamino_loss_summary(final_manifest.get("recon_info", {}))
    cor_loss = _real_lamino_loss_summary(cor_manifest.get("fista_info", {}))
    if final_loss["last"] is None or cor_loss["last"] is None:
        raise ValueError("final and COR-only manifests must include FISTA loss traces")
    final_shape = final_manifest.get("volume_shape", run_manifest.get("final_volume_shape"))
    cor_shape = cor_manifest.get("volume_shape")
    improvement = float(cor_loss["last"]) - float(final_loss["last"])
    relative = improvement / max(abs(float(cor_loss["last"])), 1.0e-12)
    return {
        "final": {
            "stage": "05_final",
            "loss": final_loss,
            "volume_shape": final_shape,
            "regulariser": final_manifest.get("recon_info", {}).get("regulariser"),
        },
        "cor_only": {
            "stage": REAL_LAMINO_COR_ONLY_STAGE,
            "loss": cor_loss,
            "volume_shape": cor_shape,
            "regulariser": cor_manifest.get("fista_info", {}).get("regulariser"),
        },
        "loss_improvement_abs": improvement,
        "loss_improvement_rel": relative,
        "same_volume_shape": final_shape == cor_shape,
    }


def _copy_real_lamino_publication_images(root: Path, out_dir: Path) -> dict[str, str]:
    pub_dir = out_dir / "publication"
    pub_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for label, stage, filename in REAL_LAMINO_PUBLICATION_IMAGES:
        source = root / stage / filename
        if not source.exists():
            raise FileNotFoundError(f"missing publication artifact {source}")
        dest = pub_dir / f"{label}_{filename}"
        shutil.copy2(source, dest)
        copied[label] = str(dest.resolve())
    return copied


def _write_real_lamino_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    success = summary["success"]
    reconstruction = summary["reconstruction_comparison"]
    lines = [
        "# Real Laminography Staged Report",
        "",
        f"- Reference case: `{summary['reference_case']}`",
        f"- Success: `{success['passed']}`",
        f"- Criterion: {success['reason']}",
        f"- Final loss: `{success['final_loss']}`",
        f"- COR-only loss: `{success['cor_only_loss']}`",
        f"- Relative improvement: `{success['loss_improvement_rel']}`",
        "",
        "| Stage | Active DOFs | Status |",
        "|---|---|---|",
    ]
    for record in summary["staged_path"]:
        dofs = ",".join(str(v) for v in record.get("active_dofs", []))
        lines.append(f"| `{record['stage']}` | `{dofs}` | `{record.get('status')}` |")
    lines.extend(
        [
            "",
            "## Reconstruction Comparison",
            "",
            "| Comparator | First FISTA loss | Last FISTA loss | Effective iterations |",
            "|---|---:|---:|---:|",
            "| Full staged final | {first} | {last} | {iters} |".format(
                **reconstruction["final"]["loss"]
            ),
            "| COR-only | {first} | {last} | {iters} |".format(
                **reconstruction["cor_only"]["loss"]
            ),
            "",
            "## Artifacts",
            "",
            f"- Summary JSON: `{summary['artifacts']['summary_json']}`",
            f"- Residual trace CSV: `{summary['artifacts']['residual_trace_csv']}`",
            f"- Geometry trace JSON: `{summary['artifacts']['geometry_trace_json']}`",
            f"- Publication image directory: `{summary['artifacts']['publication_dir']}`",
            "",
            "Truth metrics are intentionally not used for this real-data success criterion.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _real_lamino_stage_artifacts(
    root: Path,
    stage_dir: Path,
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    raw = manifest.get("artifacts", {})
    if not isinstance(raw, Mapping):
        return {}
    artifacts: dict[str, str] = {}
    for key, value in raw.items():
        candidate = Path(str(value))
        if not candidate.is_absolute():
            candidate = stage_dir / candidate
        if not candidate.exists():
            repo_relative = root.parent.parent / candidate
            if repo_relative.exists():
                candidate = repo_relative
        artifacts[str(key)] = str(candidate)
    return artifacts


def _real_lamino_stage_reconstruction_loss(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    info = manifest.get("recon_info", manifest.get("fista_info"))
    if not isinstance(info, Mapping):
        return None
    return _real_lamino_loss_summary(info)


def _real_lamino_loss_summary(info: Mapping[str, Any]) -> dict[str, Any]:
    return real_lamino_loss_summary(info)


def _read_real_lamino_stage_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    return dict(read_json_object(path))


def _write_json(path: Path, payload: Any) -> None:
    write_json_object(path, payload)


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


__all__ = [
    "FULL_REQUIRED_STAGES",
    "PARTIAL_REQUIRED_STAGES",
    "REAL_LAMINO_COR_ONLY_STAGE",
    "REAL_LAMINO_PUBLICATION_IMAGES",
    "REAL_LAMINO_REPORT_STAGED_PATH",
    "build_real_lamino_report",
    "mark_real_lamino_stage_failed",
    "real_lamino_artifact_validation_failures",
    "real_lamino_checkpoint_validation_failures",
    "real_lamino_finite_fraction",
    "real_lamino_loss_summary",
    "real_lamino_method_constraints",
    "real_lamino_pose_params_summary",
    "real_lamino_safe_params_summary",
    "real_lamino_stat_validation_failures",
    "real_lamino_success_payload",
    "validate_real_lamino_stage_output",
    "write_real_lamino_geometry_trace",
    "write_real_lamino_planned_stage_manifests",
    "write_real_lamino_residual_trace",
    "write_real_lamino_skipped_stage_manifests",
]
