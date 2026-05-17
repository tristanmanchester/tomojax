"""Result-quality helpers for article alignment benchmark artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from tomojax.bench.article_alignment_manifest import (
    article_scenario_catalog_payload,
    article_scenario_truth_payload,
)
from tomojax.bench.article_alignment_runs import article_phantom_metadata
from tomojax.bench.article_visuals import resize_for_master, vstack_rgb


@dataclass(frozen=True)
class ArticleScenarioRunArtifacts:
    """Artifact paths emitted for one article alignment scenario."""

    visual_paths: dict[str, str]
    alignment_metadata_path: Path | None = None


@dataclass(frozen=True)
class ArticleScenarioRunResult:
    """CSV row, case manifest, and optional metadata for one article scenario."""

    row: dict[str, Any]
    case_manifest: dict[str, Any]
    alignment_metadata: dict[str, Any] | None = None


def _geometry_status_label(diagnostics: Any) -> str:
    if not isinstance(diagnostics, dict):
        return ""
    overall = diagnostics.get("overall_status")
    if isinstance(overall, str) and overall:
        return overall
    blocks = diagnostics.get("blocks")
    if not isinstance(blocks, list):
        return ""
    statuses = [
        str(block.get("status"))
        for block in blocks
        if isinstance(block, dict) and block.get("status")
    ]
    if not statuses:
        return ""
    if "ill_conditioned" in statuses:
        return "ill_conditioned"
    if "underconverged" in statuses:
        return "underconverged"
    if all(status == "converged" for status in statuses):
        return "converged"
    return ",".join(statuses)


def _last_objective_provenance(outer_stats: Any) -> dict[str, Any]:
    if not isinstance(outer_stats, Sequence):
        return {}
    for stat in reversed(list(outer_stats)):
        if not isinstance(stat, Mapping):
            continue
        provenance = stat.get("objective_provenance")
        if isinstance(provenance, Mapping):
            return dict(provenance)
    return {}


def array_finite_summary(name: str, value: np.ndarray | None) -> dict[str, Any]:
    """Summarize finite/non-finite values for a volume-like array."""
    if value is None:
        return {
            "name": name,
            "present": False,
            "all_finite": False,
            "finite_fraction": 0.0,
        }
    arr = np.asarray(value)
    finite = np.isfinite(arr)
    total = int(arr.size)
    summary: dict[str, Any] = {
        "name": name,
        "present": True,
        "shape": [int(v) for v in arr.shape],
        "dtype": str(arr.dtype),
        "size": total,
        "finite_count": int(finite.sum()),
        "finite_fraction": float(finite.mean()) if total else 1.0,
        "nan_count": int(np.isnan(arr).sum()),
        "posinf_count": int(np.isposinf(arr).sum()),
        "neginf_count": int(np.isneginf(arr).sum()),
        "all_finite": bool(finite.all()),
    }
    if total and not bool(finite.all()):
        first = np.argwhere(~finite)[0]
        summary["first_nonfinite_index"] = [int(v) for v in first]
        summary["first_nonfinite_value"] = repr(arr[tuple(first)])
    if bool(finite.any()):
        vals = arr[finite].astype(np.float64, copy=False)
        summary.update(
            {
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "mean": float(np.mean(vals)),
                "rms": float(np.sqrt(np.mean(vals * vals))),
            }
        )
    return summary


def scalar_finite_summary(name: str, value: Any) -> dict[str, Any]:
    """Summarize finite/non-finite state for a scalar-like value."""
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError):
        return {
            "name": name,
            "present": value is not None,
            "all_finite": False,
            "value": repr(value),
        }
    return {"name": name, "present": True, "all_finite": bool(np.isfinite(scalar)), "value": scalar}


def article_scenario_finite_report(result: Any) -> dict[str, Any]:
    """Build the finite-value report for one article alignment scenario result."""
    volume_summaries = [
        array_finite_summary("naive_fbp", result.naive_fbp),
        array_finite_summary("calibrated_fbp", result.calibrated_fbp),
        array_finite_summary("aligned_tv", result.aligned_tv),
    ]
    metric_summaries = [
        scalar_finite_summary(name, value) for name, value in sorted(result.metrics.items())
    ]
    if result.provenance == "naive_only":
        required = ["naive_fbp"]
    else:
        required = ["naive_fbp", "calibrated_fbp", "aligned_tv"]
    required_by_name = {
        summary["name"]: bool(summary.get("all_finite", False)) for summary in volume_summaries
    }
    all_required_finite = all(required_by_name.get(name, False) for name in required) and all(
        bool(summary.get("all_finite", False)) for summary in metric_summaries
    )
    first_nonfinite = next(
        (
            summary
            for summary in volume_summaries + metric_summaries
            if not bool(summary.get("all_finite", False))
        ),
        None,
    )
    return {
        "schema_version": 1,
        "all_required_finite": bool(all_required_finite),
        "required_arrays": required,
        "first_nonfinite": first_nonfinite,
        "volumes": volume_summaries,
        "metrics": metric_summaries,
    }


def article_alignment_metadata(
    scenario: Any,
    *,
    profile: Any,
    result: Any,
) -> dict[str, Any] | None:
    """Build alignment metadata for one rendered article scenario."""
    if result.provenance == "naive_only":
        return None
    info = result.info
    return {
        "schema_version": 1,
        "scenario": asdict(scenario),
        "scenario_catalog": article_scenario_catalog_payload(scenario),
        "profile": asdict(profile),
        "acquisition": {
            "theta_span_deg": result.theta_span,
            "n_views": int(profile.views),
            "geometry_type": scenario.geometry_type,
        },
        "hidden_truth": article_scenario_truth_payload(scenario),
        "supplied_corrections": result.supplied,
        "estimated_corrections": result.estimates if result.provenance == "estimated" else {},
        "final_calibrated_geometry": result.estimates,
        "parameter_provenance": result.provenance,
        "active_dofs": info.get("active_dofs", list(scenario.geometry_dofs)),
        "active_pose_dofs": info.get("active_pose_dofs", []),
        "active_geometry_dofs": info.get("active_geometry_dofs", list(scenario.geometry_dofs)),
        "schedule_metadata": result.schedule_metadata,
        "executed_stages": result.executed_stages,
        "gauge_decision": info.get("gauge_decision"),
        "loss_kind": info.get("loss_kind"),
        "calibration_state": info.get("geometry_calibration_state"),
        "geometry_calibration_diagnostics": result.diagnostics,
        "geometry_objectives": result.geometry_objectives,
        "objective_provenance": _last_objective_provenance(info.get("outer_stats", [])),
        "solver_metadata": result.solver_metadata,
        "outer_stats": info.get("outer_stats", []),
        "metrics": result.metrics,
        "alignment_info": info,
    }


def build_article_naive_run_result(
    scenario: Any,
    *,
    profile: Any,
    result: Any,
    artifacts: ArticleScenarioRunArtifacts,
    elapsed: float,
) -> ArticleScenarioRunResult:
    """Build the summary row and manifest for a naive-only article scenario."""
    row: dict[str, Any] = {
        "slug": scenario.slug,
        "title": scenario.title,
        "scenario_category": scenario.scenario_category,
        "scenario_family": scenario.scenario_family,
        "expectation": scenario.expectation,
        "headline_eligible": bool(scenario.headline_eligible),
        "phantom_key": scenario.phantom_key,
        "schedule": scenario.schedule,
        "expected_objective": scenario.expected_objective,
        "expected_optimizer": scenario.expected_optimizer,
        "expected_loss": scenario.expected_loss,
        "geometry_type": scenario.geometry_type,
        "geometry_dofs": ",".join(scenario.geometry_dofs),
        "active_dofs": ",".join(scenario.active_dofs or scenario.geometry_dofs),
        "theta_span_deg": result.theta_span,
        "n_views": int(profile.views),
        "parameter_provenance": "naive_only",
        "hidden_truth_json": json.dumps(article_scenario_truth_payload(scenario), sort_keys=True),
        "supplied_corrections_json": "{}",
        "estimates_json": "{}",
        "geometry_diagnostics_json": "{}",
        "geometry_status": "",
        "naive_volume_nmse": result.metrics["naive_volume_nmse"],
        "calibrated_volume_nmse": np.nan,
        "aligned_tv_volume_nmse": np.nan,
        "total_outer_iters": 0,
        "elapsed_sec": elapsed,
        "error": "",
        **artifacts.visual_paths,
    }
    manifest = {
        "schema_version": 1,
        "scenario": asdict(scenario),
        "scenario_catalog": article_scenario_catalog_payload(scenario),
        "phantom": article_phantom_metadata(),
        "profile": asdict(profile),
        "acquisition": {
            "theta_span_deg": result.theta_span,
            "n_views": int(profile.views),
            "geometry_type": scenario.geometry_type,
        },
        "hidden_truth": article_scenario_truth_payload(scenario),
        "parameter_provenance": "naive_only",
        "metrics": {"naive_volume_nmse": row["naive_volume_nmse"]},
        "artifacts": artifacts.visual_paths,
        "elapsed_sec": elapsed,
    }
    return ArticleScenarioRunResult(row=row, case_manifest=manifest)


def build_article_full_run_result(
    scenario: Any,
    *,
    profile: Any,
    result: Any,
    artifacts: ArticleScenarioRunArtifacts,
    alignment_metadata: dict[str, Any],
    elapsed: float,
) -> ArticleScenarioRunResult:
    """Build the summary row and manifest for an aligned article scenario."""
    info = result.info
    row: dict[str, Any] = {
        "slug": scenario.slug,
        "title": scenario.title,
        "scenario_category": scenario.scenario_category,
        "scenario_family": scenario.scenario_family,
        "expectation": scenario.expectation,
        "headline_eligible": bool(scenario.headline_eligible),
        "phantom_key": scenario.phantom_key,
        "schedule": scenario.schedule,
        "schedule_name": str(info.get("schedule_name", "")),
        "schedule_stages_json": json.dumps(result.executed_stages, sort_keys=True),
        "last_schedule_stage_name": str(
            (result.executed_stages[-1].get("stage_name", "") if result.executed_stages else "")
        ),
        "gauge_status": str(
            (info.get("gauge_decision") or {}).get("status", "")
            if isinstance(info.get("gauge_decision"), Mapping)
            else ""
        ),
        "expected_objective": scenario.expected_objective,
        "expected_optimizer": scenario.expected_optimizer,
        "expected_loss": scenario.expected_loss,
        "geometry_type": scenario.geometry_type,
        "geometry_dofs": ",".join(scenario.geometry_dofs),
        "active_dofs": ",".join(str(v) for v in info.get("active_dofs", scenario.geometry_dofs)),
        "active_pose_dofs": ",".join(str(v) for v in info.get("active_pose_dofs", [])),
        "active_geometry_dofs": ",".join(
            str(v) for v in info.get("active_geometry_dofs", scenario.geometry_dofs)
        ),
        "loss_kind": str(info.get("loss_kind", "")),
        "geometry_objectives": ",".join(result.geometry_objectives),
        "objective_provenance_json": json.dumps(
            _last_objective_provenance(info.get("outer_stats", [])),
            sort_keys=True,
        ),
        "solver_metadata_json": json.dumps(result.solver_metadata, sort_keys=True),
        "objective_kind": str(result.solver_metadata.get("objective_kind", "")),
        "optimizer_kind": str(result.solver_metadata.get("optimizer_kind", "")),
        "outer_loss_kind": str(result.solver_metadata.get("outer_loss_kind", "")),
        "recon_sensitivity": str(result.solver_metadata.get("recon_sensitivity", "")),
        "fold_eval_mode": str(result.solver_metadata.get("fold_eval_mode", "")),
        "active_gradient_mode": str(result.solver_metadata.get("active_gradient_mode", "")),
        "theta_span_deg": result.theta_span,
        "n_views": int(profile.views),
        "parameter_provenance": result.provenance,
        "hidden_truth_json": json.dumps(article_scenario_truth_payload(scenario), sort_keys=True),
        "supplied_corrections_json": json.dumps(result.supplied, sort_keys=True),
        "estimates_json": json.dumps(result.estimates, sort_keys=True),
        "geometry_diagnostics_json": json.dumps(result.diagnostics, sort_keys=True),
        "geometry_status": _geometry_status_label(result.diagnostics),
        "naive_volume_nmse": result.metrics["naive_volume_nmse"],
        "calibrated_volume_nmse": result.metrics["calibrated_volume_nmse"],
        "aligned_tv_volume_nmse": result.metrics["aligned_tv_volume_nmse"],
        "total_outer_iters": int(info.get("total_outer_iters", 0)),
        "elapsed_sec": elapsed,
        "error": "",
        **artifacts.visual_paths,
    }
    manifest = {
        "schema_version": 1,
        "scenario": asdict(scenario),
        "scenario_catalog": article_scenario_catalog_payload(scenario),
        "phantom": article_phantom_metadata(),
        "profile": asdict(profile),
        "acquisition": {
            "theta_span_deg": result.theta_span,
            "n_views": int(profile.views),
            "geometry_type": scenario.geometry_type,
        },
        "hidden_truth": article_scenario_truth_payload(scenario),
        "supplied_corrections": result.supplied,
        "estimated_corrections": result.estimates if result.provenance == "estimated" else {},
        "final_calibrated_geometry": result.estimates,
        "parameter_provenance": result.provenance,
        "active_dofs": info.get("active_dofs", list(scenario.geometry_dofs)),
        "active_pose_dofs": info.get("active_pose_dofs", []),
        "active_geometry_dofs": info.get("active_geometry_dofs", list(scenario.geometry_dofs)),
        "schedule_metadata": result.schedule_metadata,
        "executed_stages": result.executed_stages,
        "gauge_decision": info.get("gauge_decision"),
        "loss_kind": info.get("loss_kind"),
        "calibration_state": info.get("geometry_calibration_state"),
        "geometry_calibration_diagnostics": result.diagnostics,
        "geometry_objectives": result.geometry_objectives,
        "objective_provenance": _last_objective_provenance(info.get("outer_stats", [])),
        "outer_stats": info.get("outer_stats", []),
        "metrics": result.metrics,
        "artifacts": artifacts.visual_paths,
        "alignment_metadata": alignment_metadata,
        "elapsed_sec": elapsed,
    }
    return ArticleScenarioRunResult(
        row=row,
        case_manifest=manifest,
        alignment_metadata=alignment_metadata,
    )


def build_article_scenario_run_result(
    scenario: Any,
    *,
    profile: Any,
    result: Any,
    artifacts: ArticleScenarioRunArtifacts,
    alignment_metadata: dict[str, Any] | None,
    elapsed: float,
) -> ArticleScenarioRunResult:
    """Build the summary row and manifest for any article scenario result."""
    if result.provenance == "naive_only":
        return build_article_naive_run_result(
            scenario,
            profile=profile,
            result=result,
            artifacts=artifacts,
            elapsed=elapsed,
        )
    if alignment_metadata is None:
        raise ValueError("full scenario result requires alignment metadata")
    return build_article_full_run_result(
        scenario,
        profile=profile,
        result=result,
        artifacts=artifacts,
        alignment_metadata=alignment_metadata,
        elapsed=elapsed,
    )


def build_article_nonfinite_run_result(
    scenario: Any,
    *,
    profile: Any,
    result: Any,
    alignment_metadata: dict[str, Any] | None,
    finite_report: dict[str, Any],
    elapsed: float,
    out_dir: Path,
) -> ArticleScenarioRunResult:
    """Build the fail-closed result contract when visual inputs are non-finite."""
    report_path = out_dir / "finite_report.json"
    metadata_path = out_dir / "alignment_metadata.pre_visual.json"
    visual_paths = {
        "finite_report_json": str(report_path),
        "alignment_metadata_json": str(metadata_path) if alignment_metadata is not None else "",
    }
    first_nonfinite = finite_report.get("first_nonfinite")
    reason = "nonfinite_visual_inputs"
    if isinstance(first_nonfinite, Mapping):
        reason = f"nonfinite_{first_nonfinite.get('name', 'visual_inputs')}"
    row: dict[str, Any] = {
        "slug": scenario.slug,
        "title": scenario.title,
        "scenario_category": scenario.scenario_category,
        "scenario_family": scenario.scenario_family,
        "expectation": scenario.expectation,
        "headline_eligible": bool(scenario.headline_eligible),
        "phantom_key": scenario.phantom_key,
        "schedule": scenario.schedule,
        "schedule_name": str(result.info.get("schedule_name", "")),
        "schedule_stages_json": json.dumps(result.executed_stages, sort_keys=True),
        "last_schedule_stage_name": str(
            (result.executed_stages[-1].get("stage_name", "") if result.executed_stages else "")
        ),
        "expected_objective": scenario.expected_objective,
        "expected_optimizer": scenario.expected_optimizer,
        "expected_loss": scenario.expected_loss,
        "geometry_type": scenario.geometry_type,
        "geometry_dofs": ",".join(scenario.geometry_dofs),
        "active_dofs": ",".join(
            str(v) for v in result.info.get("active_dofs", scenario.active_dofs)
        ),
        "active_pose_dofs": ",".join(str(v) for v in result.info.get("active_pose_dofs", [])),
        "active_geometry_dofs": ",".join(
            str(v) for v in result.info.get("active_geometry_dofs", scenario.geometry_dofs)
        ),
        "loss_kind": str(result.info.get("loss_kind", "")),
        "geometry_objectives": ",".join(result.geometry_objectives),
        "solver_metadata_json": json.dumps(result.solver_metadata, sort_keys=True),
        "theta_span_deg": result.theta_span,
        "n_views": int(profile.views),
        "parameter_provenance": result.provenance,
        "hidden_truth_json": json.dumps(article_scenario_truth_payload(scenario), sort_keys=True),
        "supplied_corrections_json": json.dumps(result.supplied, sort_keys=True),
        "estimates_json": json.dumps(result.estimates, sort_keys=True),
        "geometry_diagnostics_json": json.dumps(result.diagnostics, sort_keys=True),
        "geometry_status": _geometry_status_label(result.diagnostics),
        "naive_volume_nmse": result.metrics.get("naive_volume_nmse", np.nan),
        "calibrated_volume_nmse": result.metrics.get("calibrated_volume_nmse", np.nan),
        "aligned_tv_volume_nmse": result.metrics.get("aligned_tv_volume_nmse", np.nan),
        "total_outer_iters": int(result.info.get("total_outer_iters", 0)),
        "elapsed_sec": elapsed,
        "error": reason,
        "truth_xy": "",
        "naive_fbp_xy": "",
        "calibrated_fbp_xy": "",
        "aligned_tv_xy": "",
        "before_after_panel": "",
        "inspection_panel": "",
        "loss_panel": "",
        "diagnostics_panel": "",
        "truth_orthos": "",
        "calibrated_orthos": "",
        "absolute_difference_xy": "",
        "status": "nonfinite",
        **visual_paths,
    }
    manifest = {
        "schema_version": 1,
        "scenario": asdict(scenario),
        "scenario_catalog": article_scenario_catalog_payload(scenario),
        "phantom": article_phantom_metadata(),
        "profile": asdict(profile),
        "status": "nonfinite",
        "error": reason,
        "finite_report": finite_report,
        "alignment_metadata": alignment_metadata,
        "metrics": result.metrics,
        "elapsed_sec": elapsed,
    }
    return ArticleScenarioRunResult(
        row=row,
        case_manifest=manifest,
        alignment_metadata=alignment_metadata,
    )


def write_article_summary_csv(rows: list[dict[str, Any]], summary_path: Path) -> None:
    """Write the per-scenario article alignment summary CSV."""
    if not rows:
        return
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_article_master_panel(rows: list[dict[str, Any]], master_path: Path) -> None:
    """Write the stacked article master panel from per-scenario panels."""
    panels: list[np.ndarray] = []
    for row in rows:
        panel_path = row.get("inspection_panel") or row.get("before_after_panel")
        if not isinstance(panel_path, str) or not panel_path.strip():
            continue
        path = Path(panel_path)
        if path.is_file():
            panels.append(resize_for_master(iio.imread(path), width=1200))
    if panels:
        iio.imwrite(master_path, vstack_rgb(panels, pad=10))


__all__ = [
    "ArticleScenarioRunArtifacts",
    "ArticleScenarioRunResult",
    "array_finite_summary",
    "article_alignment_metadata",
    "article_scenario_finite_report",
    "build_article_full_run_result",
    "build_article_naive_run_result",
    "build_article_nonfinite_run_result",
    "build_article_scenario_run_result",
    "scalar_finite_summary",
    "write_article_master_panel",
    "write_article_summary_csv",
]
