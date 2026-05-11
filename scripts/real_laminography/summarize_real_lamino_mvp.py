#!/usr/bin/env python3
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedCallResult=false
"""Build the real-laminography MVP artifact contract from a staged run."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
import csv
import json
from pathlib import Path
import shutil
from typing import Any

STAGED_PATH: tuple[tuple[str, str], ...] = (
    ("baseline", "00_baseline"),
    ("cor_detu", "01_setup_geometry/01_cor"),
    ("detector_roll", "01_setup_geometry/02_detector_roll"),
    ("axis_direction", "01_setup_geometry/03_axis_direction"),
    ("pose_phi", "02_pose_phi"),
    ("pose_dx_dz", "03_pose_dx_dz"),
    ("pose_5dof_polish", "04_pose_polish"),
    ("final_reconstruction", "05_final"),
)
COR_ONLY_STAGE = "06_cor_only_fista"
PUBLICATION_IMAGES: tuple[tuple[str, str, str], ...] = (
    ("before", "00_baseline", "orthos.png"),
    ("before_xy", "00_baseline", "aligned_xy_global_z209.png"),
    ("cor_only", COR_ONLY_STAGE, "orthos.png"),
    ("cor_only_xy", COR_ONLY_STAGE, "aligned_xy_global_z209.png"),
    ("full", "05_final", "orthos.png"),
    ("full_xy", "05_final", "aligned_xy_global_z209.png"),
    ("full_delta_xy", "05_final", "delta_xy_global_z209.png"),
)


def main(argv: list[str] | None = None) -> int:
    """Run the real-laminography MVP report CLI."""
    args = _parse_args(argv)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "real_mvp_report"
    report = build_real_mvp_report(
        run_dir,
        out_dir=out_dir,
        require_success=bool(args.require_success),
    )
    print(f"mvp_report: {report['artifacts']['summary_json']}")
    print(f"success: {report['success']['passed']}")
    return 0


def build_real_mvp_report(
    run_dir: Path,
    *,
    out_dir: Path | None = None,
    require_success: bool = False,
) -> dict[str, Any]:
    """Write and return the real-laminography MVP report for ``run_dir``."""
    root = Path(run_dir)
    if out_dir is None:
        out_dir = root / "real_mvp_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = _read_json(root / "run_manifest.json")
    status = _read_json(root / "status.json")
    stage_records = _collect_stage_records(root)
    reconstruction = _reconstruction_comparison(root)
    success = _success_payload(reconstruction)
    if require_success and not bool(success["passed"]):
        raise RuntimeError(str(success["reason"]))

    publication_artifacts = _copy_publication_images(root, out_dir)
    residual_trace = _write_residual_trace(out_dir / "real_mvp_residual_trace.csv", stage_records)
    geometry_trace = _write_geometry_trace(out_dir / "real_mvp_geometry_trace.json", stage_records)

    summary: dict[str, Any] = {
        "schema": "tomojax.real_lamino_mvp_report.v1",
        "run_dir": str(root),
        "reference_case": root.name,
        "success": success,
        "quality_basis": {
            "kind": "real_reconstruction_quality",
            "primary_metric": "final_fista_last_loss_lt_cor_only_fista_last_loss",
            "truth_metrics": "not_applicable_real_data",
            "synthetic_truth_metrics_allowed": False,
        },
        "staged_path": stage_records,
        "reconstruction_comparison": reconstruction,
        "publication_artifacts": publication_artifacts,
        "provenance": {
            "input": run_manifest.get("input"),
            "backend": run_manifest.get("backend"),
            "devices": run_manifest.get("devices"),
            "final_volume_shape": run_manifest.get("final_volume_shape"),
            "final_setup_estimates": run_manifest.get("final_setup_estimates"),
            "final_pose_summary": run_manifest.get("final_pose_summary"),
            "status_completed_at": status.get("completed_at"),
        },
        "method_constraints": {
            "cor_grid_search_added": False,
            "sinogram_or_correlation_method_added": False,
            "sharpness_or_autofocus_method_added": False,
            "benchmark_only_knobs_promoted": False,
            "cor_only_role": "comparator_artifact_from_reference_run",
        },
        "artifacts": {
            "summary_json": str((out_dir / "real_mvp_summary.json").resolve()),
            "summary_md": str((out_dir / "real_mvp_summary.md").resolve()),
            "residual_trace_csv": str(residual_trace.resolve()),
            "geometry_trace_json": str(geometry_trace.resolve()),
            "publication_dir": str((out_dir / "publication").resolve()),
        },
    }
    _write_json(out_dir / "real_mvp_summary.json", summary)
    _write_markdown(out_dir / "real_mvp_summary.md", summary)
    return summary


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--require-success", action="store_true")
    return parser.parse_args(argv)


def _collect_stage_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for label, rel in STAGED_PATH:
        records.append(_stage_record(root, label=label, rel=rel, required=True))
    records.append(
        _stage_record(root, label="cor_only_comparator", rel=COR_ONLY_STAGE, required=True)
    )
    return records


def _stage_record(root: Path, *, label: str, rel: str, required: bool) -> dict[str, Any]:
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
        "reconstruction_loss": _stage_reconstruction_loss(manifest),
        "artifacts": _stage_artifacts(root, stage_dir, manifest),
        "summary_rows": _read_stage_summary(stage_dir / "stage_summary.csv"),
    }


def _reconstruction_comparison(root: Path) -> dict[str, Any]:
    run_manifest = _read_json(root / "run_manifest.json")
    final_manifest = _read_json(root / "05_final" / "stage_manifest.json")
    cor_manifest = _read_json(root / COR_ONLY_STAGE / "stage_manifest.json")
    final_loss = _loss_summary(final_manifest.get("recon_info", {}))
    cor_loss = _loss_summary(cor_manifest.get("fista_info", {}))
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
            "stage": COR_ONLY_STAGE,
            "loss": cor_loss,
            "volume_shape": cor_shape,
            "regulariser": cor_manifest.get("fista_info", {}).get("regulariser"),
        },
        "loss_improvement_abs": improvement,
        "loss_improvement_rel": relative,
        "same_volume_shape": final_shape == cor_shape,
    }


def _success_payload(reconstruction: Mapping[str, Any]) -> dict[str, Any]:
    final = reconstruction["final"]["loss"]["last"]
    cor = reconstruction["cor_only"]["loss"]["last"]
    same_shape = bool(reconstruction.get("same_volume_shape"))
    improved = final is not None and cor is not None and float(final) < float(cor)
    passed = bool(improved and same_shape)
    reason = (
        "final reconstruction improves COR-only FISTA loss at matching shape"
        if passed
        else "final reconstruction did not improve COR-only loss at matching shape"
    )
    return {
        "passed": passed,
        "reason": reason,
        "final_loss": final,
        "cor_only_loss": cor,
        "loss_improvement_abs": reconstruction["loss_improvement_abs"],
        "loss_improvement_rel": reconstruction["loss_improvement_rel"],
        "same_volume_shape": same_shape,
    }


def _copy_publication_images(root: Path, out_dir: Path) -> dict[str, str]:
    pub_dir = out_dir / "publication"
    pub_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for label, stage, filename in PUBLICATION_IMAGES:
        source = root / stage / filename
        if not source.exists():
            raise FileNotFoundError(f"missing publication artifact {source}")
        dest = pub_dir / f"{label}_{filename}"
        shutil.copy2(source, dest)
        copied[label] = str(dest.resolve())
    return copied


def _write_residual_trace(path: Path, records: Iterable[Mapping[str, Any]]) -> Path:
    fieldnames = (
        "label",
        "stage",
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
    _write_csv(path, rows, fieldnames)
    return path


def _write_geometry_trace(path: Path, records: Iterable[Mapping[str, Any]]) -> Path:
    trace = {
        "schema": "tomojax.real_lamino_geometry_trace.v1",
        "stages": [
            {
                "label": record.get("label"),
                "stage": record.get("stage"),
                "active_dofs": record.get("active_dofs", []),
                "bounds": record.get("bounds"),
                "geometry_calibration_state": record.get("geometry_calibration_state"),
                "params_summary": record.get("params_summary"),
            }
            for record in records
        ],
    }
    _write_json(path, trace)
    return path


def _write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    success = summary["success"]
    reconstruction = summary["reconstruction_comparison"]
    lines = [
        "# Real Laminography MVP Report",
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


def _stage_artifacts(root: Path, stage_dir: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
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


def _stage_reconstruction_loss(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    info = manifest.get("recon_info", manifest.get("fista_info"))
    if not isinstance(info, Mapping):
        return None
    return _loss_summary(info)


def _loss_summary(info: Mapping[str, Any]) -> dict[str, Any]:
    losses = info.get("loss", [])
    if not isinstance(losses, list) or not losses:
        return {"first": None, "last": None, "iters": 0}
    return {
        "first": float(losses[0]),
        "last": float(losses[-1]),
        "iters": int(info.get("effective_iters", len(losses))),
    }


def _read_stage_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


if __name__ == "__main__":
    raise SystemExit(main())
