from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
import gc
import logging
import os
from pathlib import Path
from typing import cast
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.alignment._config import AlignConfig
from tomojax.alignment._gauge import Gauge, least_motion_estimate
from tomojax.alignment._geometry.geometry_applier import BaseGeometryArrays, pose_stack_for_setup
from tomojax.alignment._model.dofs import POSE_WIDTH
from tomojax.alignment._model.state import AlignmentState
from tomojax.alignment._objectives.loss_specs import loss_spec_name, resolve_loss_for_level
from tomojax.alignment._observer import ObserverAction, ObserverCallback, OuterStat
from tomojax.alignment._pose._coupled_objective import AlignmentMemoryError
from tomojax.alignment._prealign import seeded_translation_params
from tomojax.alignment._results import (
    AlignMultiresCheckpointCallback,
    AlignMultiresInfo,
    AlignMultiresResumeState,
)
from tomojax.core.geometry.base import Detector, Geometry, Grid
from tomojax.core.geometry.cone import is_cone_beam
from tomojax.geometry import stack_view_poses

from ._stage_runners import _run_multires_level_stages
from ._stage_state import (
    _build_multires_context,
    _build_stage_runtime,
    _emit_level_completion_checkpoint,
    _emit_run_completion_checkpoint,
    _final_align_multires_info,
    _final_multires_volume,
    _initial_multires_run_state,
    _level_initial_volume,
    _levels_to_run,
    _multires_run_is_complete,
    _prepare_multires_level_state,
    _state_after_multires_level,
)
from ._stage_types import MultiresContext, MultiresLevel, MultiresRunState

LOG = logging.getLogger(__name__)
# Warnings name the first caller outside the package, whichever entry point it used.
_PACKAGE = (str(Path(__file__).parents[2]) + os.sep,)


def _run_one_multires_level(
    *,
    geometry: Geometry,
    context: MultiresContext,
    resume_state: AlignMultiresResumeState | None,
    checkpoint_callback: AlignMultiresCheckpointCallback | None,
    level_index: int,
    level: MultiresLevel,
    state: MultiresRunState,
) -> MultiresRunState:
    level_factor = int(level["factor"])
    grid = level["grid"]
    detector = level["detector"]
    projections = level["projections"]
    active_loss_spec = resolve_loss_for_level(context.cfg.loss, level_factor)
    active_loss_name = loss_spec_name(active_loss_spec)
    logging.info(
        "Alignment level %d/%d factor=%d using loss=%s",
        level_index + 1,
        len(context.levels),
        level_factor,
        active_loss_name,
    )
    resuming_this_level = (
        resume_state is not None
        and not resume_state.level_complete
        and int(resume_state.level_index) == level_index
    )
    x0 = _level_initial_volume(
        level=level,
        x_init=state.x_init,
        prev_factor=state.prev_factor,
        resume_state=resume_state,
        resuming_this_level=resuming_this_level,
    )
    params0 = (
        resume_state.pose_params
        if resuming_this_level and resume_state is not None
        else state.pose_params
    )
    if params0 is None and level_index == 0:
        # The first level starts from explicit zero poses, so seed it here.
        params0 = seeded_translation_params(geometry, grid, detector, projections, context.cfg)
    level_run = _prepare_multires_level_state(
        resume_state=resume_state,
        level_index=level_index,
        loss_hist=state.loss_hist,
        global_outer_stats=state.global_outer_stats,
        executed_outer_iterations=state.executed_outer_iterations,
    )
    level_stats: list[OuterStat] = [dict(stat) for stat in level_run.preserved_level_stats]
    level_losses: list[float] = [float(value) for value in level_run.preserved_level_losses]
    stage_result = _run_multires_level_stages(
        geometry=geometry,
        grid=grid,
        detector=detector,
        projections=projections,
        cfg=context.cfg,
        resolved_schedule=context.resolved_schedule,
        active_loss_spec=active_loss_spec,
        active_loss_name=active_loss_name,
        setup_alignment_state=state.setup_alignment_state,
        active_geometry_dofs=context.active_geometry_dofs,
        level_factor=level_factor,
        stage_runtime=_build_stage_runtime(
            context=context,
            level_index=level_index,
            level_factor=level_factor,
            level_run=level_run,
            state=state,
            level_stats=level_stats,
            level_losses=level_losses,
            active_loss_name=active_loss_name,
            checkpoint_callback=checkpoint_callback,
        ),
        resume_state=resume_state,
        resuming_this_level=resuming_this_level,
        level_resume=level_run,
        global_elapsed_offset=state.global_elapsed_offset,
        x_lvl=x0 if x0 is not None else jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32),
        pose_params=params0
        if params0 is not None
        else jnp.zeros((projections.shape[0], POSE_WIDTH), dtype=jnp.float32),
        level_stats=level_stats,
        level_losses=level_losses,
        final_gauge_fix=state.final_gauge_fix,
        final_gauge_fix_dofs=state.final_gauge_fix_dofs,
        final_gauge_fix_stats=state.final_gauge_fix_stats or {},
    )
    next_state, info = _state_after_multires_level(
        state=state,
        level=level,
        level_index=level_index,
        level_run=level_run,
        stage_result=stage_result,
    )
    _emit_level_completion_checkpoint(
        checkpoint_callback=checkpoint_callback,
        x_lvl=stage_result.x_lvl,
        pose_params=stage_result.pose_params,
        info=info,
        level_index=level_index,
        level_factor=level_factor,
        level_completed_after=len(stage_result.level_stats),
        global_outer_idx=int(next_state.global_outer_idx),
        prev_factor=level["factor"],
        loss_hist=next_state.loss_hist,
        global_outer_stats=next_state.global_outer_stats,
        global_elapsed_offset=next_state.global_elapsed_offset,
        level_complete=_level_is_complete(
            context, len(stage_result.level_stats), info, stage_result.level_action
        ),
        setup_alignment_state=stage_result.setup_alignment_state,
        active_geometry_dofs=context.active_geometry_dofs,
        resolved_schedule=context.resolved_schedule,
        level_stats=stage_result.level_stats,
        ray_integrator=context.cfg.ray_integrator,
    )
    return next_state


def _level_is_complete(
    context: MultiresContext,
    level_completed_after: int,
    info: Mapping[str, object],
    level_action: ObserverAction,
) -> bool:
    return (
        level_completed_after
        >= sum(int(stage.maxiter) for stage in context.resolved_schedule.stages)
        or level_action == "advance_level"
        or not bool(info.get("stopped_by_observer", False))
    )


def _run_multires_levels(
    *,
    geometry: Geometry,
    context: MultiresContext,
    resume_state: AlignMultiresResumeState | None,
    checkpoint_callback: AlignMultiresCheckpointCallback | None,
) -> MultiresRunState:
    state = _initial_multires_run_state(context=context, resume_state=resume_state)
    if resume_state is not None and resume_state.run_complete:
        # The completed run's level_index is the last level that ran.
        skipped = context.factors_list[int(resume_state.level_index) + 1 :]
        return replace(
            state,
            x_init=resume_state.x,
            pose_params=resume_state.pose_params,
            prev_factor=1,
            factors_skipped=tuple(skipped),
        )
    for level_index, level in _levels_to_run(context.levels, resume_state):
        try:
            state = _run_one_multires_level(
                geometry=geometry,
                context=context,
                resume_state=resume_state,
                checkpoint_callback=checkpoint_callback,
                level_index=int(level_index),
                level=level,
                state=state,
            )
        except AlignmentMemoryError as error:
            if state.prev_factor is None:  # nothing coarser to keep
                raise
            skipped = tuple(int(lv["factor"]) for lv in context.levels[int(level_index) :])
            warnings.warn(
                f"alignment stops at factor {state.prev_factor}, skipping factors "
                f"{list(skipped)}: {error}",
                skip_file_prefixes=_PACKAGE,
            )
            state = replace(state, factors_skipped=skipped)
            break
        if state.final_observer_action == "stop_run":
            break
        _release_completed_level_accelerator_state(state)
    return state


def _release_completed_level_accelerator_state(state: MultiresRunState) -> None:
    """Drop per-level JAX executable/cache state before compiling the next level."""
    if state.x_init is not None:
        jax.block_until_ready(state.x_init)
    if state.pose_params is not None:
        jax.block_until_ready(state.pose_params)
    gc.collect()
    clear_caches = getattr(jax, "clear_caches", None)
    if callable(clear_caches):
        clear_caches()


def _setup_dofs_requested(cfg: AlignConfig | None) -> bool:
    from tomojax.alignment._config import _resolved_schedule_for_cfg

    return bool(_resolved_schedule_for_cfg(cfg or AlignConfig()).active_geometry_dofs)


def _fix_gauge(
    state: MultiresRunState,
    x_final: jnp.ndarray,
    *,
    context: MultiresContext,
    geometry: Geometry,
    detector: Detector,
    grid: Grid,
) -> tuple[MultiresRunState, jnp.ndarray, Gauge | None, tuple[str, ...]]:
    """Report the least-motion estimate among those predicting the same data.

    A rigid motion of the object, undone by every pose, leaves the data
    unchanged, so the solve may end anywhere along it (see
    :mod:`tomojax.alignment._gauge`). In a parallel beam with detector-frame
    poses, a schedule reporting the detector centre (``cor_then_pose``, or one
    estimating ``det_u_px``) also takes the poses' constant u shift into it.
    """
    dofs = context.active_geometry_dofs
    if state.pose_params is None:
        return state, x_final, None, dofs
    frame = context.cfg.pose_translation_frame
    pose_dofs = context.resolved_schedule.active_pose_dofs
    cone_beam = is_cone_beam(geometry)
    reports_centre = context.resolved_schedule.name == "cor_then_pose" or "det_u_px" in dofs
    offset = not cone_beam and reports_centre and "dx" in pose_dofs and frame == "detector"
    setup_state = cast("AlignmentState | None", state.setup_alignment_state)
    if setup_state is None:
        nominal = np.asarray(stack_view_poses(geometry, int(state.pose_params.shape[0])))
    else:
        base = BaseGeometryArrays.from_geometry(geometry, detector)
        nominal = np.asarray(pose_stack_for_setup(base, setup_state.setup))
    volume, params, gauge = least_motion_estimate(
        np.asarray(x_final),
        np.asarray(state.pose_params),
        nominal=nominal,
        grid=grid,
        translation_frame=frame,
        active=pose_dofs,
        cone_beam=cone_beam,
        detector_offset=offset,
    )
    if gauge is None:
        LOG.info("The object reaches the grid edge, which fixes its position: poses unchanged")
        return state, x_final, None, dofs
    LOG.info(
        "Least-motion estimate: removed a common rotation of %s deg and shift of %s",
        [round(v, 4) for v in gauge.rotation_deg],
        [round(float(v), 4) for v in gauge.shift],
    )
    if setup_state is not None:
        setup = setup_state.setup
        if offset:
            det_u_px = float(setup.det_u_px) + gauge.detector_offset / float(detector.du)
            LOG.info("Detector-u (centre-of-rotation) offset %.3f px", det_u_px)
            setup = setup.replace(det_u_px=det_u_px)
        setup_state = setup_state.replace(
            setup=setup, pose=setup_state.pose.replace(pose_params=jnp.asarray(params))
        )
    fixed = replace(
        state, pose_params=jnp.asarray(params, jnp.float32), setup_alignment_state=setup_state
    )
    if offset and "det_u_px" not in dofs:
        dofs = (*dofs, "det_u_px")
    return fixed, jnp.asarray(volume), gauge, dofs


def align_multires(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    factors: Iterable[int] = (2, 1),
    cfg: AlignConfig | None = None,
    observer: ObserverCallback | None = None,
    resume_state: AlignMultiresResumeState | None = None,
    checkpoint_callback: AlignMultiresCheckpointCallback | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, AlignMultiresInfo]:
    """Coarse-to-fine alignment using simple binning for speed and robustness.

    Carries alignment parameters across levels and downsamples/upsamples volume.
    """
    if is_cone_beam(geometry) and _setup_dofs_requested(cfg):
        raise ValueError(
            "align_multires: setup stages (detector centre, roll, axis) model parallel "
            "beams; for cone-beam geometry calibrate the axis offset and detector roll with "
            "tomojax.recon.calibrate_cone_axis, then run pose stages (which estimate dy along "
            "the beam)"
        )
    context = _build_multires_context(
        geometry,
        grid,
        detector,
        projections,
        cfg,
        observer,
        resume_state,
        factors,
    )
    state = _run_multires_levels(
        geometry=geometry,
        context=context,
        resume_state=resume_state,
        checkpoint_callback=checkpoint_callback,
    )
    x_final = _final_multires_volume(
        x_init=state.x_init,
        prev_factor=state.prev_factor,
        grid=grid,
    )
    state, x_final, gauge, active_geometry_dofs = _fix_gauge(
        state, x_final, context=context, geometry=geometry, detector=detector, grid=grid
    )
    run_complete = _multires_run_is_complete(
        pose_params=state.pose_params,
        stopped_by_observer=state.stopped_by_observer,
        resume_state=resume_state,
        last_level_index_processed=state.last_level_index_processed,
        level_count=len(context.levels),
        levels_skipped=bool(state.factors_skipped),
    )
    _emit_run_completion_checkpoint(
        checkpoint_callback=checkpoint_callback,
        pose_params=state.pose_params,
        run_complete=run_complete,
        x_final=x_final,
        levels_run=len(context.levels) - len(state.factors_skipped),
        executed_outer_iterations=state.executed_outer_iterations,
        loss_hist=state.loss_hist,
        global_outer_stats=state.global_outer_stats,
        global_elapsed_offset=state.global_elapsed_offset,
        setup_alignment_state=state.setup_alignment_state,
        active_geometry_dofs=active_geometry_dofs,
        resolved_schedule=context.resolved_schedule,
        ray_integrator=context.cfg.ray_integrator,
    )

    return (
        x_final,
        state.pose_params
        if state.pose_params is not None
        else jnp.zeros((projections.shape[0], POSE_WIDTH), jnp.float32),
        _final_align_multires_info(
            loss_hist=state.loss_hist,
            factors_list=context.factors_list[
                : len(context.factors_list) - len(state.factors_skipped)
            ],
            factors_skipped=state.factors_skipped,
            final_loss_kind=state.final_loss_kind,
            cfg=context.cfg,
            stopped_by_observer=state.stopped_by_observer,
            final_observer_action=state.final_observer_action,
            executed_outer_iterations=state.executed_outer_iterations,
            global_elapsed_offset=state.global_elapsed_offset,
            global_outer_stats=state.global_outer_stats,
            resolved_schedule=context.resolved_schedule,
            geometry=geometry,
            final_pose_model_variables=state.final_pose_model_variables,
            final_per_view_variables=state.final_per_view_variables,
            final_pose_model_basis_shape=state.final_pose_model_basis_shape,
            active_geometry_dofs=active_geometry_dofs,
            final_gauge_fix=state.final_gauge_fix,
            final_gauge_fix_dofs=state.final_gauge_fix_dofs,
            final_gauge_fix_stats=state.final_gauge_fix_stats,
            setup_alignment_state=state.setup_alignment_state,
            gauge=None if gauge is None else gauge.to_dict(),
        ),
    )
