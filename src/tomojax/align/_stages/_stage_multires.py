from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
import gc
import logging

import jax
import jax.numpy as jnp

from tomojax.align._config import AlignConfig
from tomojax.align._objectives.loss_specs import loss_spec_name, resolve_loss_for_level
from tomojax.align._observer import ObserverAction, ObserverCallback, OuterStat
from tomojax.align._results import (
    AlignMultiresCheckpointCallback,
    AlignMultiresInfo,
    AlignMultiresResumeState,
)
from tomojax.core.geometry.base import Detector, Geometry, Grid

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
    _params_for_multires_level,
    _prepare_multires_level_state,
    _state_after_multires_level,
)
from ._stage_types import MultiresContext, MultiresLevel, MultiresRunState


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
        resume_state.params5 if resuming_this_level and resume_state is not None else state.params5
    )
    params0 = _params_for_multires_level(
        level_index=level_index,
        geometry=geometry,
        context=context,
        level=level,
        x0=x0,
        params0=params0,
    )
    level_run = _prepare_multires_level_state(
        resume_state=resume_state,
        level_index=level_index,
        loss_hist=state.loss_hist,
        global_outer_stats=state.global_outer_stats,
        executed_outer_iters=state.executed_outer_iters,
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
        params5=params0
        if params0 is not None
        else jnp.zeros((projections.shape[0], 5), dtype=jnp.float32),
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
        params5=stage_result.params5,
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
        return replace(state, x_init=resume_state.x, params5=resume_state.params5, prev_factor=1)
    for level_index, level in _levels_to_run(context.levels, resume_state):
        state = _run_one_multires_level(
            geometry=geometry,
            context=context,
            resume_state=resume_state,
            checkpoint_callback=checkpoint_callback,
            level_index=int(level_index),
            level=level,
            state=state,
        )
        if state.final_observer_action == "stop_run":
            break
        _release_completed_level_accelerator_state(state)
    return state


def _release_completed_level_accelerator_state(state: MultiresRunState) -> None:
    """Drop per-level JAX executable/cache state before compiling the next level."""
    if state.x_init is not None:
        jax.block_until_ready(state.x_init)
    if state.params5 is not None:
        jax.block_until_ready(state.params5)
    gc.collect()
    clear_caches = getattr(jax, "clear_caches", None)
    if callable(clear_caches):
        clear_caches()


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
    run_complete = _multires_run_is_complete(
        params5=state.params5,
        stopped_by_observer=state.stopped_by_observer,
        resume_state=resume_state,
        last_level_index_processed=state.last_level_index_processed,
        level_count=len(context.levels),
    )
    _emit_run_completion_checkpoint(
        checkpoint_callback=checkpoint_callback,
        params5=state.params5,
        run_complete=run_complete,
        x_final=x_final,
        level_count=len(context.levels),
        executed_outer_iters=state.executed_outer_iters,
        loss_hist=state.loss_hist,
        global_outer_stats=state.global_outer_stats,
        global_elapsed_offset=state.global_elapsed_offset,
        setup_alignment_state=state.setup_alignment_state,
        active_geometry_dofs=context.active_geometry_dofs,
        resolved_schedule=context.resolved_schedule,
    )

    return (
        x_final,
        state.params5
        if state.params5 is not None
        else jnp.zeros((projections.shape[0], 5), jnp.float32),
        _final_align_multires_info(
            loss_hist=state.loss_hist,
            factors_list=context.factors_list,
            final_loss_kind=state.final_loss_kind,
            cfg=context.cfg,
            stopped_by_observer=state.stopped_by_observer,
            final_observer_action=state.final_observer_action,
            executed_outer_iters=state.executed_outer_iters,
            global_elapsed_offset=state.global_elapsed_offset,
            global_outer_stats=state.global_outer_stats,
            resolved_schedule=context.resolved_schedule,
            geometry=geometry,
            final_pose_model_variables=state.final_pose_model_variables,
            final_per_view_variables=state.final_per_view_variables,
            final_pose_model_basis_shape=state.final_pose_model_basis_shape,
            active_geometry_dofs=context.active_geometry_dofs,
            final_gauge_fix=state.final_gauge_fix,
            final_gauge_fix_dofs=state.final_gauge_fix_dofs,
            final_gauge_fix_stats=state.final_gauge_fix_stats,
            setup_alignment_state=state.setup_alignment_state,
        ),
    )
