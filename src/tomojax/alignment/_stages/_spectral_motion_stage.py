"""Projection-only motion initialization for an explicitly requested stage."""

from __future__ import annotations

from dataclasses import replace
import time
from typing import cast

import jax.numpy as jnp
import numpy as np

from tomojax.alignment._config import AlignConfig
from tomojax.alignment._geometry.common_lines import prepare_common_lines
from tomojax.alignment._geometry.geometry_applier import (
    materialize_setup_geometry,
    setup_moved_detector,
)
from tomojax.alignment._geometry.spectral_motion import refine_common_line_motion
from tomojax.alignment._model.schedules import ResolvedAlignmentStage
from tomojax.alignment._model.state import AlignmentState, PoseState
from tomojax.alignment._observer import OuterStat
from tomojax.core.geometry.base import Detector, Geometry, Grid
from tomojax.geometry import LaminographyGeometry, RotationAxisGeometry, stack_view_poses

from ._stage_types import StageLoopState, StageRuntime


def _seed_motion(geometry, detector, projections):
    axis = np.asarray(geometry.axis_unit_lab)
    angles = np.asarray(geometry.angles)
    lines = prepare_common_lines(projections, angles, detector)
    if lines is None:
        return None, {"motion_seed_reason": "insufficient_evidence"}
    old = lines.motion(axis, center_u=detector.center[0])
    order = np.argsort(lines.view_indices)
    initial = np.zeros((len(angles), 2))
    if old.shifts is not None:
        initial = np.column_stack(
            [
                np.interp(np.arange(len(angles)), lines.view_indices[order], old.shifts[order, k])
                for k in range(2)
            ]
        )
    fitted = refine_common_line_motion(
        projections,
        angles,
        detector,
        axis,
        center_u=detector.center[0],
        initial_shifts=initial,
    )
    shifts = fitted.shifts
    method = "bounded_spectral_knots"
    if shifts is None and old.shifts is not None:
        shifts, method = initial, "common_line_phase_fallback"
    diagnostics = {
        "motion_seed_method": method,
        "motion_seed_reason": fitted.status if shifts is None else "projection_evidence",
        "motion_seed_refinement_status": fitted.status,
        "motion_seed_coherence_before": fitted.coherence_before,
        "motion_seed_coherence_after": fitted.coherence_after,
        "motion_seed_iterations": fitted.iterations,
    }
    if shifts is None:
        return None, diagnostics
    # One rigid object translation is a common-line phase null direction.
    basis = np.asarray(stack_view_poses(geometry, len(angles)))[:, [0, 2], :3].reshape(-1, 3)
    shifts = shifts - (basis @ np.linalg.lstsq(basis, shifts.ravel(), rcond=1e-8)[0]).reshape(-1, 2)
    return shifts, diagnostics


def _within_bounds(shifts, bounds):
    for name, lo, hi in bounds:
        if name in {"dx", "dz"}:
            values = shifts[:, 0 if name == "dx" else 1]
            if np.any((values < lo) | (values > hi)):
                return False
    return True


def _candidate(state, stage, geometry, grid, detector, projections, cfg, level_factor):
    reason = None
    if cfg.pose_translation_frame != "detector":
        raise ValueError("spectral motion initialization requires detector-frame translations")
    if set(stage.active_pose_dofs) != {"dx", "dz"}:
        reason = "requires_both_detector_translations"
    elif np.any(np.asarray(state.pose_params)):
        reason = "nonzero_pose_corrections"
    elif not isinstance(geometry, LaminographyGeometry | RotationAxisGeometry):
        reason = "unsupported_geometry"
    setup_state = cast("AlignmentState", state.setup_alignment_state)
    moved = setup_moved_detector(detector, setup_state.setup, level_factor=level_factor)
    if moved is None:
        reason = "rolled_detector"
    if reason is not None:
        return state.pose_params, {
            "motion_seed_status": "seed_declined",
            "motion_seed_reason": reason,
        }
    effective = materialize_setup_geometry(
        geometry, grid, cast("Detector", moved), setup_state.setup
    )
    shifts, diagnostics = _seed_motion(effective, moved, projections)
    if shifts is not None and not _within_bounds(shifts, cfg.bounds):
        shifts = None
        diagnostics["motion_seed_reason"] = "explicit_bounds"
    diagnostics["motion_seed_status"] = (
        "unvalidated_seed" if shifts is not None else "seed_declined"
    )
    if shifts is None:
        return state.pose_params, diagnostics
    return state.pose_params.at[:, 3:5].set(jnp.asarray(shifts)), diagnostics


def run_spectral_motion_stage(
    state: StageLoopState,
    stage: ResolvedAlignmentStage,
    *,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    cfg: AlignConfig,
    level_factor: int,
    stage_runtime: StageRuntime,
    stage_global_start: int,
    level_stats: list[OuterStat],
) -> StageLoopState:
    """Update only zero-initialized detector translations, with no scored volume loss."""
    start = time.perf_counter()
    params, diagnostics = _candidate(
        state, stage, geometry, grid, detector, projections, cfg, level_factor
    )
    seconds = time.perf_counter() - start
    stat = cast(
        "OuterStat",
        {
            **diagnostics,
            "outer_idx": 1,
            "motion_validation_performed": False,
            "num_train_reconstructions": 0,
            "validation_residual_count": 0,
            "wall_time": seconds,
        },
    )
    level_stats.extend(
        stage_runtime.enrich_stats([stat], stage=stage, global_start=stage_global_start)
    )
    info = dict(state.info)
    info.update(
        wall_time_total=state.level_wall_time + seconds, completed_outer_iterations=len(level_stats)
    )
    aligned = cast("AlignmentState", state.setup_alignment_state).replace(
        pose=PoseState(params, translation_frame=cfg.pose_translation_frame),
        volume=state.x_lvl,
    )
    return replace(
        state,
        pose_params=params,
        setup_alignment_state=aligned,
        info=info,
        level_wall_time=state.level_wall_time + seconds,
    )
