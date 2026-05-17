"""Computation helpers for article alignment benchmark scenarios."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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
from tomojax.bench.article_alignment_manifest import article_scenario_supplied_payload
from tomojax.bench.article_alignment_runs import (
    ArticleRunProfile,
    ArticleScenario,
    article_theta_span_deg,
)
from tomojax.core.geometry import Detector, Geometry, Grid, LaminographyGeometry, ParallelGeometry
from tomojax.core.projector import forward_project_view
from tomojax.recon.fbp import fbp
from tomojax.recon.fista_tv import FistaConfig, fista_tv


@dataclass(frozen=True)
class ArticleScenarioComputationResult:
    """Computed volumes and geometry metadata for one article scenario."""

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


def _hidden_state(scenario: ArticleScenario, geometry: Geometry) -> GeometryCalibrationState:
    return _state_from_values(
        geometry,
        active_geometry_dofs=scenario.geometry_dofs,
        det_u_px=float(scenario.hidden_det_u_px),
        det_v_px=float(scenario.hidden_det_v_px),
        detector_roll_deg=float(scenario.hidden_detector_roll_deg),
        axis_rot_x_deg=float(scenario.hidden_axis_rot_x_deg),
        axis_rot_y_deg=float(scenario.hidden_axis_rot_y_deg),
    )


def _supplied_state(scenario: ArticleScenario, geometry: Geometry) -> GeometryCalibrationState:
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
    scenario: ArticleScenario,
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
    profile: ArticleRunProfile,
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
    scenario: ArticleScenario,
    *,
    nominal_geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    profile: ArticleRunProfile,
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


def execute_article_scenario_computation(
    scenario: ArticleScenario,
    *,
    profile: ArticleRunProfile,
    grid: Grid,
    detector: Detector,
    truth: np.ndarray,
    naive_only: bool,
) -> ArticleScenarioComputationResult:
    """Simulate projections, reconstruct, and optionally align one article scenario."""
    volume = jnp.asarray(truth, dtype=jnp.float32)
    theta_span = article_theta_span_deg(scenario)
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
        return ArticleScenarioComputationResult(
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

    supplied = article_scenario_supplied_payload(scenario)
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
    return ArticleScenarioComputationResult(
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


__all__ = [
    "ArticleScenarioComputationResult",
    "execute_article_scenario_computation",
]
