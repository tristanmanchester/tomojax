from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Sequence

import jax
import numpy as np

from tomojax.bench.article_alignment_compute import (
    ArticleScenarioComputationResult as ScenarioComputationResult,
    execute_article_scenario_computation as _execute_scenario_computation,
)
from tomojax.bench.article_alignment_manifest import (
    article_scenario_catalog_payload as _scenario_catalog_payload,
    article_scenario_supplied_payload as _scenario_supplied_payload,
    article_scenario_truth_payload as _scenario_truth_payload,
    build_article_run_manifest as build_run_manifest,
)
from tomojax.bench.article_alignment_results import (
    ArticleScenarioRunArtifacts as ScenarioRunArtifacts,
    article_alignment_metadata as _build_alignment_metadata,
    article_scenario_finite_report as _scenario_finite_report,
    build_article_nonfinite_run_result as _build_nonfinite_run_result,
    build_article_scenario_run_result as _build_scenario_run_result,
    write_article_master_panel as _write_master_panel,
    write_article_summary_csv as _write_summary,
)
from tomojax.bench.article_alignment_runs import (
    ArticleRunProfile as RunProfile,
    ArticleScenario as Scenario,
    article_scenario_catalog_for_kind as scenario_catalog_for_kind,
    article_theta_span_deg as _theta_span_deg,
    make_article_phantom as _phantom,
    profile_from_args,
)
from tomojax.bench.article_visuals import (
    alignment_visualization_payload as _visualization_payload,
    naive_visualization_payload as _naive_visualization_payload,
    write_alignment_visuals,
    write_naive_visuals,
)
from tomojax.core.geometry import Detector, Grid
from tomojax.io import normalize_json


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    return normalize_json(value, sort_mapping_keys=True, catch_to_dict_errors=True)


def _write_visuals(payload: Any, *, out_dir: Path) -> dict[str, str]:
    return write_alignment_visuals(payload, out_dir=out_dir)


def _write_naive_visuals(payload: Any, *, out_dir: Path) -> dict[str, str]:
    return write_naive_visuals(payload, out_dir=out_dir)


def _write_scenario_artifacts(
    scenario: Scenario,
    *,
    out_dir: Path,
    profile: RunProfile,
    truth: np.ndarray,
    result: ScenarioComputationResult,
    alignment_metadata: dict[str, Any] | None,
) -> ScenarioRunArtifacts:
    if result.provenance == "naive_only":
        return ScenarioRunArtifacts(
            visual_paths=_write_naive_visuals(
                _naive_visualization_payload(scenario, truth=truth, result=result),
                out_dir=out_dir,
            )
        )
    visual_paths = _write_visuals(
        _visualization_payload(scenario, profile=profile, truth=truth, result=result),
        out_dir=out_dir,
    )
    alignment_metadata_path = out_dir / "alignment_metadata.json"
    if alignment_metadata is not None:
        _write_json(alignment_metadata_path, alignment_metadata)
        visual_paths["alignment_metadata_json"] = str(alignment_metadata_path)
    return ScenarioRunArtifacts(
        visual_paths=visual_paths,
        alignment_metadata_path=alignment_metadata_path,
    )


def _run_scenario(
    scenario: Scenario,
    *,
    out_dir: Path,
    profile: RunProfile,
    grid: Grid,
    detector: Detector,
    truth: np.ndarray,
    naive_only: bool = False,
) -> dict[str, Any]:
    start_time = time.time()
    computation = _execute_scenario_computation(
        scenario,
        profile=profile,
        grid=grid,
        detector=detector,
        truth=truth,
        naive_only=naive_only,
    )
    alignment_metadata = _build_alignment_metadata(
        scenario,
        profile=profile,
        result=computation,
    )
    finite_report = _scenario_finite_report(computation)
    _write_json(out_dir / "finite_report.json", finite_report)
    if alignment_metadata is not None:
        _write_json(out_dir / "alignment_metadata.pre_visual.json", alignment_metadata)
    elapsed_before_visual = time.time() - start_time
    if not bool(finite_report.get("all_required_finite", False)):
        run_result = _build_nonfinite_run_result(
            scenario,
            profile=profile,
            result=computation,
            alignment_metadata=alignment_metadata,
            finite_report=finite_report,
            elapsed=elapsed_before_visual,
            out_dir=out_dir,
        )
        _write_json(out_dir / "case_manifest.json", run_result.case_manifest)
        jax.clear_caches()
        return run_result.row
    artifacts = _write_scenario_artifacts(
        scenario,
        out_dir=out_dir,
        profile=profile,
        truth=truth,
        result=computation,
        alignment_metadata=alignment_metadata,
    )
    elapsed = time.time() - start_time
    run_result = _build_scenario_run_result(
        scenario,
        profile=profile,
        result=computation,
        artifacts=artifacts,
        alignment_metadata=alignment_metadata,
        elapsed=elapsed,
    )
    _write_json(out_dir / "case_manifest.json", run_result.case_manifest)
    jax.clear_caches()
    return run_result.row


def _select_scenarios(args: argparse.Namespace) -> list[Scenario]:
    scenarios = scenario_catalog_for_kind(str(args.scenario_set))
    if args.scenario:
        wanted = set(args.scenario)
        scenarios = [s for s in scenarios if s.slug in wanted]
        missing = sorted(wanted - {s.slug for s in scenarios})
        if missing:
            raise SystemExit(f"Unknown scenario(s): {', '.join(missing)}")
    if args.limit is not None:
        scenarios = scenarios[: int(args.limit)]
    return scenarios


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.out)
    artifacts = out_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    profile = profile_from_args(args)
    scenarios = _select_scenarios(args)
    manifest = build_run_manifest(profile, scenarios, suite_name=str(args.scenario_set))
    manifest["naive_only"] = bool(args.naive_only)
    _write_json(out_root / "run_manifest.json", manifest)
    _write_json(artifacts / "scenario_catalog.json", manifest["scenarios"])

    if args.dry_run:
        _write_json(
            artifacts / "status.json",
            {
                "state": "dry_run_completed",
                "profile": asdict(profile),
                "scenario_count": len(scenarios),
                "scenarios": [s.slug for s in scenarios],
                "run_manifest": str(out_root / "run_manifest.json"),
            },
        )
        return

    grid = Grid(profile.size, profile.size, profile.size, 1.0, 1.0, 1.0)
    detector = Detector(profile.size, profile.size, 1.0, 1.0, det_center=(0.0, 0.0))
    truth = _phantom(profile.size)
    rows: list[dict[str, Any]] = []
    summary_path = artifacts / "summary.csv"
    master_path = artifacts / "alignment_before_after_master.png"

    for index, scenario in enumerate(scenarios, start=1):
        _write_json(
            artifacts / "status.json",
            {
                "state": "running",
                "scenario": scenario.slug,
                "index": index,
                "total": len(scenarios),
                "profile": asdict(profile),
                "summary_csv": str(summary_path),
                "master_panel": str(master_path),
            },
        )
        try:
            row = _run_scenario(
                scenario,
                out_dir=artifacts / scenario.slug,
                profile=profile,
                grid=grid,
                detector=detector,
                truth=truth,
                naive_only=bool(args.naive_only),
            )
            row.setdefault("status", "completed")
        except Exception as exc:
            row = {
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
                "theta_span_deg": _theta_span_deg(scenario),
                "n_views": int(profile.views),
                "parameter_provenance": "failed",
                "hidden_truth_json": json.dumps(_scenario_truth_payload(scenario), sort_keys=True),
                "supplied_corrections_json": json.dumps(
                    _scenario_supplied_payload(scenario), sort_keys=True
                ),
                "estimates_json": "{}",
                "geometry_diagnostics_json": "{}",
                "geometry_status": "",
                "naive_volume_nmse": np.nan,
                "calibrated_volume_nmse": np.nan,
                "aligned_tv_volume_nmse": np.nan,
                "total_outer_iters": 0,
                "elapsed_sec": 0.0,
                "truth_xy": "",
                "naive_fbp_xy": "",
                "calibrated_fbp_xy": "",
                "aligned_tv_xy": "",
                "before_after_panel": "",
                "truth_orthos": "",
                "calibrated_orthos": "",
                "absolute_difference_xy": "",
                "status": "failed",
                "error": repr(exc),
            }
            _write_json(
                artifacts / scenario.slug / "case_manifest.json",
                {
                    "schema_version": 1,
                    "scenario": asdict(scenario),
                    "scenario_catalog": _scenario_catalog_payload(scenario),
                    "profile": asdict(profile),
                    "status": "failed",
                    "error": repr(exc),
                },
            )
            if not args.continue_on_error:
                rows.append(row)
                _write_summary(rows, summary_path)
                raise
        rows.append(row)
        _write_summary(rows, summary_path)
        _write_master_panel(rows, master_path)

    _write_json(
        artifacts / "status.json",
        {
            "state": "completed",
            "profile": asdict(profile),
            "scenarios": [row["slug"] for row in rows],
            "summary_csv": str(summary_path),
            "master_panel": str(master_path),
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--profile",
        default="docs",
        help="Run profile: docs or diagnostic.",
    )
    parser.add_argument(
        "--scenario-set",
        default="default",
        help="Scenario set, for example default, diagnostic, diagnostic_128, or comprehensive_128.",
    )
    parser.add_argument("--naive-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--views", type=int, default=None)
    parser.add_argument("--views-per-batch", type=int, default=None)
    parser.add_argument("--levels", type=int, nargs="+", default=None)
    parser.add_argument("--outer-iters", type=int, default=None)
    parser.add_argument("--recon-iters", type=int, default=None)
    parser.add_argument("--tv-prox-iters", type=int, default=None)
    early_stop = parser.add_mutually_exclusive_group()
    early_stop.add_argument("--early-stop", dest="early_stop", action="store_true", default=None)
    early_stop.add_argument("--no-early-stop", dest="early_stop", action="store_false")
    parser.add_argument("--early-stop-rel-impr", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--gather-dtype", default=None, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    args.profile = _normalize_profile_name(str(args.profile), parser)
    args.scenario_set = _normalize_scenario_set(str(args.scenario_set))
    return args


def _normalize_profile_name(name: str, parser: argparse.ArgumentParser) -> str:
    normalized = {"smoke": "diagnostic"}.get(name, name)
    if normalized not in {"docs", "diagnostic"}:
        parser.error("--profile must be one of: docs, diagnostic")
    return normalized


def _normalize_scenario_set(name: str) -> str:
    return {
        "diagnostic_64": "smoke_64",
        "pose_reference": "pose_parity",
        "pose_reference_128": "pose_parity_128",
    }.get(name, name)


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
