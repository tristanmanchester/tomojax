from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.align import AlignConfig, align_multires
from tomojax.align.api import (
    GeometryCalibrationState,
    geometry_with_axis_state,
    level_detector_grid,
    normalize_geometry_dofs,
    schedule_preset,
    summarize_geometry_calibration_stats,
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
from tomojax.core.geometry import Detector, Geometry, Grid, LaminographyGeometry, ParallelGeometry
from tomojax.core.projector import forward_project_view
from tomojax.io import normalize_json
from tomojax.recon.fbp import fbp
from tomojax.recon.fista_tv import FistaConfig, fista_tv


@dataclass(frozen=True)
class ScenarioComputationResult:
    theta_span: float
    naive_fbp: np.ndarray
    calibrated_fbp: np.ndarray | None
    aligned_tv: np.ndarray | None
    provenance: str
    supplied: dict[str, float]
    estimates: dict[str, Any]
    metrics: dict[str, float]
    info: Mapping[str, Any]
    diagnostics: Any
    geometry_objectives: list[str]
    schedule_metadata: Any
    executed_stages: Any
    solver_metadata: dict[str, Any]


def _build_geometry(
    *,
    grid: Grid,
    detector: Detector,
    thetas: np.ndarray,
    geometry_type: str,
    tilt_deg: float,
) -> Geometry:
    if geometry_type == "lamino":
        return LaminographyGeometry(
            grid=grid,
            detector=detector,
            thetas_deg=thetas,
            tilt_deg=float(tilt_deg),
            tilt_about="x",
        )
    return ParallelGeometry(grid=grid, detector=detector, thetas_deg=thetas)


def _state_from_values(
    geometry: Geometry,
    *,
    active_geometry_dofs: Sequence[str],
    det_u_px: float = 0.0,
    det_v_px: float = 0.0,
    detector_roll_deg: float = 0.0,
    axis_rot_x_deg: float = 0.0,
    axis_rot_y_deg: float = 0.0,
) -> GeometryCalibrationState:
    active = normalize_geometry_dofs(active_geometry_dofs, geometry=geometry)
    state = GeometryCalibrationState.from_geometry(geometry, active_geometry_dofs=active)
    return state.replace_values(
        (
            "det_u_px",
            "det_v_px",
            "detector_roll_deg",
            "axis_rot_x_deg",
            "axis_rot_y_deg",
        ),
        jnp.asarray(
            [det_u_px, det_v_px, detector_roll_deg, axis_rot_x_deg, axis_rot_y_deg],
            dtype=jnp.float32,
        ),
    )


def _hidden_state(scenario: Scenario, geometry: Geometry) -> GeometryCalibrationState:
    return _state_from_values(
        geometry,
        active_geometry_dofs=scenario.geometry_dofs,
        det_u_px=float(scenario.hidden_det_u_px),
        det_v_px=float(scenario.hidden_det_v_px),
        detector_roll_deg=float(scenario.hidden_detector_roll_deg),
        axis_rot_x_deg=float(scenario.hidden_axis_rot_x_deg),
        axis_rot_y_deg=float(scenario.hidden_axis_rot_y_deg),
    )


def _supplied_state(scenario: Scenario, geometry: Geometry) -> GeometryCalibrationState:
    return _state_from_values(
        geometry,
        active_geometry_dofs=(),
        det_u_px=float(scenario.supplied_det_u_px or 0.0),
        det_v_px=float(scenario.supplied_det_v_px or 0.0),
        detector_roll_deg=float(scenario.supplied_detector_roll_deg or 0.0),
        axis_rot_x_deg=float(scenario.supplied_axis_rot_x_deg or 0.0),
        axis_rot_y_deg=float(scenario.supplied_axis_rot_y_deg or 0.0),
    )


def _state_values(state: GeometryCalibrationState) -> dict[str, Any]:
    return {
        "det_u_px": float(state.det_u_px),
        "det_v_px": float(state.det_v_px),
        "detector_roll_deg": float(state.detector_roll_deg),
        "axis_rot_x_deg": float(state.axis_rot_x_deg),
        "axis_rot_y_deg": float(state.axis_rot_y_deg),
        "axis_unit_lab": [float(v) for v in state.axis_unit_lab()],
    }


def _simulate(
    scenario: Scenario,
    *,
    nominal_geometry: Geometry,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    views_per_batch: int,
    gather_dtype: str,
) -> jnp.ndarray:
    true_state = _hidden_state(scenario, nominal_geometry)
    true_geometry = geometry_with_axis_state(nominal_geometry, grid, detector, true_state)
    true_det_grid = level_detector_grid(detector, state=true_state, factor=1)
    chunks = []
    n_views = len(getattr(nominal_geometry, "thetas_deg"))
    for start in range(0, n_views, max(1, int(views_per_batch))):
        stop = min(start + max(1, int(views_per_batch)), n_views)
        chunk = [
            forward_project_view(
                true_geometry,
                grid,
                detector,
                volume,
                i,
                gather_dtype=gather_dtype,
                det_grid=true_det_grid,
            )
            for i in range(start, stop)
        ]
        chunks.append(jnp.stack(chunk, axis=0))
    return jnp.concatenate(chunks, axis=0)


def _run_fbp(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    views_per_batch: int,
    gather_dtype: str,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> np.ndarray:
    x = fbp(
        geometry,
        grid,
        detector,
        projections,
        views_per_batch=max(1, int(views_per_batch)),
        gather_dtype=gather_dtype,
        checkpoint_projector=True,
        det_grid=det_grid,
    )
    return np.asarray(x, dtype=np.float32)


def _run_tv_recon(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    profile: RunProfile,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> np.ndarray:
    x, _ = fista_tv(
        geometry,
        grid,
        detector,
        projections,
        init_x=None,
        config=FistaConfig(
            iters=int(profile.recon_iters),
            lambda_tv=0.0015,
            tv_prox_iters=int(profile.tv_prox_iters),
            views_per_batch=max(1, int(profile.views_per_batch)),
            projector_unroll=1,
            checkpoint_projector=True,
            gather_dtype=str(profile.gather_dtype),
        ),
        det_grid=det_grid,
    )
    return np.asarray(x, dtype=np.float32)


def _run_geometry_alignment(
    scenario: Scenario,
    *,
    nominal_geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    profile: RunProfile,
) -> tuple[np.ndarray, GeometryCalibrationState, dict[str, Any]]:
    geometry_dofs = normalize_geometry_dofs(scenario.geometry_dofs, geometry=nominal_geometry)
    active_dofs = tuple(scenario.active_dofs or geometry_dofs)
    all_known_dofs = {
        "det_u_px",
        "det_v_px",
        "detector_roll_deg",
        "axis_rot_x_deg",
        "axis_rot_y_deg",
        "alpha",
        "beta",
        "phi",
        "dx",
        "dz",
    }
    freeze_dofs: tuple[str, ...] = ()
    if scenario.schedule == "expert_coupled":
        schedule: object | None = schedule_preset(
            "expert_coupled",
            active_dofs=active_dofs,
            gauge_policy="prior_required",
        )
        optimise_dofs: tuple[str, ...] | None = None
    elif scenario.schedule:
        schedule = scenario.schedule
        optimise_dofs = None
        freeze_dofs = tuple(sorted(all_known_dofs.difference(active_dofs)))
    else:
        schedule = None
        optimise_dofs = active_dofs
    x_aligned, _, info = align_multires(
        nominal_geometry,
        grid,
        detector,
        projections,
        factors=profile.levels,
        cfg=AlignConfig(
            outer_iters=int(profile.outer_iters),
            recon_iters=int(profile.recon_iters),
            lambda_tv=0.0015,
            tv_prox_iters=int(profile.tv_prox_iters),
            schedule=schedule,
            optimise_dofs=optimise_dofs,
            freeze_dofs=freeze_dofs,
            early_stop=bool(profile.early_stop),
            early_stop_rel_impr=float(profile.early_stop_rel_impr),
            early_stop_patience=int(profile.early_stop_patience),
            gather_dtype=profile.gather_dtype,
            checkpoint_projector=True,
            views_per_batch=max(1, int(profile.views_per_batch)),
            projector_unroll=1,
            gn_damping=1e-3,
        ),
    )
    state = GeometryCalibrationState.from_checkpoint(
        info.get("geometry_calibration_state"),
        nominal_geometry,
        active_geometry_dofs=geometry_dofs,
    )
    return np.asarray(x_aligned, dtype=np.float32), state, dict(info)


def _volume_nmse(candidate: np.ndarray, truth: np.ndarray) -> float:
    denom = float(np.mean(np.square(truth))) + 1e-6
    return float(np.mean(np.square(candidate - truth)) / denom)


def _last_solver_metadata(outer_stats: Any) -> dict[str, Any]:
    if not isinstance(outer_stats, Sequence):
        return {}
    keys = (
        "objective_kind",
        "optimizer_kind",
        "outer_loss_kind",
        "recon_sensitivity",
        "fold_eval_mode",
        "active_gradient_mode",
        "views_per_batch",
        "n_folds",
        "validation_projection_chunked",
        "recon_projection_chunked",
        "train_reconstruction_gradient",
        "schedule_name",
        "schedule_stage_index",
        "schedule_stage_name",
        "schedule_stage_active_dofs",
        "gauge_policy",
        "gauge_status",
    )
    for stat in reversed(list(outer_stats)):
        if not isinstance(stat, Mapping):
            continue
        payload = {key: stat.get(key) for key in keys if key in stat}
        if payload:
            return payload
    return {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    return normalize_json(value, sort_mapping_keys=True, catch_to_dict_errors=True)


def _execute_scenario_computation(
    scenario: Scenario,
    *,
    profile: RunProfile,
    grid: Grid,
    detector: Detector,
    truth: np.ndarray,
    naive_only: bool,
) -> ScenarioComputationResult:
    volume = jnp.asarray(truth, dtype=jnp.float32)
    theta_span = _theta_span_deg(scenario)
    thetas = np.linspace(0.0, theta_span, int(profile.views), endpoint=False, dtype=np.float32)
    nominal_geometry = _build_geometry(
        grid=grid,
        detector=detector,
        thetas=thetas,
        geometry_type=scenario.geometry_type,
        tilt_deg=scenario.nominal_tilt_deg,
    )
    projections = _simulate(
        scenario,
        nominal_geometry=nominal_geometry,
        grid=grid,
        detector=detector,
        volume=volume,
        views_per_batch=profile.views_per_batch,
        gather_dtype=profile.gather_dtype,
    )
    projections.block_until_ready()
    naive_fbp = _run_fbp(
        nominal_geometry,
        grid,
        detector,
        projections,
        views_per_batch=profile.views_per_batch,
        gather_dtype=profile.gather_dtype,
    )
    if naive_only:
        return ScenarioComputationResult(
            theta_span=theta_span,
            naive_fbp=naive_fbp,
            calibrated_fbp=None,
            aligned_tv=None,
            provenance="naive_only",
            supplied={},
            estimates={},
            metrics={"naive_volume_nmse": _volume_nmse(naive_fbp, truth)},
            info={},
            diagnostics={},
            geometry_objectives=[],
            schedule_metadata={},
            executed_stages=[],
            solver_metadata={},
        )

    supplied = _scenario_supplied_payload(scenario)
    if scenario.active_dofs or scenario.geometry_dofs:
        aligned_tv, state, info = _run_geometry_alignment(
            scenario,
            nominal_geometry=nominal_geometry,
            grid=grid,
            detector=detector,
            projections=projections,
            profile=profile,
        )
        provenance = "estimated"
    elif supplied:
        state = _supplied_state(scenario, nominal_geometry)
        calibrated_geometry_for_tv = geometry_with_axis_state(
            nominal_geometry,
            grid,
            detector,
            state,
        )
        supplied_grid = level_detector_grid(detector, state=state, factor=1)
        aligned_tv = _run_tv_recon(
            calibrated_geometry_for_tv,
            grid,
            detector,
            projections,
            profile=profile,
            det_grid=supplied_grid,
        )
        info = {
            "geometry_calibration_state": state.to_calibration_state().to_dict(),
            "geometry_calibration_diagnostics": {},
            "outer_stats": [],
            "total_outer_iters": 0,
            "control": "supplied_known_correction",
        }
        provenance = "supplied"
    else:
        aligned_tv, state, info = _run_geometry_alignment(
            scenario,
            nominal_geometry=nominal_geometry,
            grid=grid,
            detector=detector,
            projections=projections,
            profile=profile,
        )
        provenance = "frozen"

    if "geometry_calibration_diagnostics" not in info:
        info["geometry_calibration_diagnostics"] = summarize_geometry_calibration_stats(
            info.get("outer_stats", [])
        )
    diagnostics = info.get("geometry_calibration_diagnostics", {})
    geometry_objectives = sorted(
        {
            str(stat.get("geometry_objective"))
            for stat in info.get("outer_stats", [])
            if isinstance(stat, Mapping) and stat.get("geometry_objective")
        }
    )
    calibrated_geometry = geometry_with_axis_state(nominal_geometry, grid, detector, state)
    calibrated_det_grid = level_detector_grid(detector, state=state, factor=1)
    calibrated_fbp = _run_fbp(
        calibrated_geometry,
        grid,
        detector,
        projections,
        views_per_batch=profile.views_per_batch,
        gather_dtype=profile.gather_dtype,
        det_grid=calibrated_det_grid,
    )
    estimates = _state_values(state)
    metrics = {
        "naive_volume_nmse": _volume_nmse(naive_fbp, truth),
        "calibrated_volume_nmse": _volume_nmse(calibrated_fbp, truth),
        "aligned_tv_volume_nmse": _volume_nmse(aligned_tv, truth),
    }
    return ScenarioComputationResult(
        theta_span=theta_span,
        naive_fbp=naive_fbp,
        calibrated_fbp=calibrated_fbp,
        aligned_tv=aligned_tv,
        provenance=provenance,
        supplied=supplied,
        estimates=estimates,
        metrics=metrics,
        info=info,
        diagnostics=diagnostics,
        geometry_objectives=geometry_objectives,
        schedule_metadata=info.get("schedule", {}),
        executed_stages=info.get("schedule_stages", []),
        solver_metadata=_last_solver_metadata(info.get("outer_stats", [])),
    )


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
