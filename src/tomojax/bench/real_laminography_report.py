"""Report helpers for real-laminography benchmark runs."""

from __future__ import annotations

from collections.abc import Mapping
import csv
import json
from pathlib import Path
from typing import Any

PARTIAL_REQUIRED_STAGES = frozenset(
    ("00_baseline", "01_setup_geometry/01_cor", "06_cor_only_fista")
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
        "06_cor_only_fista",
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
        str(record.get("stage"))
        for record in records
        if str(record.get("status")) == "completed"
    }
    planned = {
        str(record.get("stage"))
        for record in records
        if str(record.get("status")) == "planned"
    }
    failed_or_skipped = [
        record
        for record in records
        if str(record.get("status")) in {"failed", "skipped"}
    ]
    final_loss = reconstruction["final"]["loss"]["last"]
    cor_loss = reconstruction["cor_only"]["loss"]["last"]
    full_complete = FULL_REQUIRED_STAGES <= completed
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
    partial_complete = PARTIAL_REQUIRED_STAGES <= completed and cor_loss is not None
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    "real_lamino_method_constraints",
    "real_lamino_success_payload",
    "write_real_lamino_geometry_trace",
    "write_real_lamino_residual_trace",
]
