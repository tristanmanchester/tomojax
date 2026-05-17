from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace

import jax
import jax.numpy as jnp

from tomojax.core.geometry.base import Detector, Geometry, Grid
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.multires import (
    bin_projections,
    scale_detector,
    scale_grid,
    upsample_volume,
    validate_scale_factor,
)
from tomojax.core.projector import forward_project_view_T
from tomojax.core.validation import validate_grid, validate_projection_stack
from tomojax.motion import phase_corr_shift
from tomojax.recon.fista_tv import FistaConfig, fista_tv

from ._config import AlignConfig, _resolved_schedule_for_cfg
from ._geometry.geometry_applier import BaseGeometryArrays
from ._geometry.geometry_blocks import (
    add_geometry_acquisition_diagnostics,
    summarize_geometry_calibration_stats,
)
from ._model.gauge import active_gauge_dofs, normalize_gauge_fix
from ._model.state import alignment_state_from_checkpoint
from ._objectives.loss_specs import validate_loss_schedule_levels
from ._observer import ObserverAction, ObserverCallback, OuterStat, adapt_observer_callback
from ._profiles import profile_policy_from_config
from ._results import AlignMultiresCheckpointCallback, AlignMultiresInfo, AlignMultiresResumeState
from ._setup_stage import _geometry_calibration_payload
from ._stage_types import (
    LevelResumePlan,
    MultiresContext,
    MultiresLevel,
    MultiresRunState,
    StageRunResult,
    StageRuntime,
    _build_multires_checkpoint_state,
    _stage_index_or_none,
)


def _prepare_multires_level_state(
    *,
    resume_state: AlignMultiresResumeState | None,
    level_index: int,
    loss_hist: list[float],
    global_outer_stats: list[OuterStat],
    executed_outer_iters: int,
) -> LevelResumePlan:
    resuming = (
        resume_state is not None
        and not resume_state.level_complete
        and int(resume_state.level_index) == int(level_index)
    )
    level_completed_before = int(resume_state.completed_outer_iters_in_level) if resuming else 0
    current_level_stats = [
        dict(stat) for stat in global_outer_stats if stat.get("level_index") == int(level_index)
    ]
    stats_before_level = [
        dict(stat) for stat in global_outer_stats if stat.get("level_index") != int(level_index)
    ]
    loss_before_level = (
        list(loss_hist[:-level_completed_before])
        if resuming and level_completed_before > 0
        else list(loss_hist)
    )
    global_before_level = (
        int(executed_outer_iters) - level_completed_before
        if resuming
        else int(executed_outer_iters)
    )
    resume_stage_index = int(resume_state.stage_index) if resuming else 0
    resume_stage_completed = bool(resume_state.stage_completed) if resuming else False
    resume_stage_iters = int(resume_state.completed_outer_iters_in_stage) if resuming else 0

    if not resuming:
        return LevelResumePlan(
            resuming=False,
            level_completed_before=0,
            stats_before_level=stats_before_level,
            loss_before_level=loss_before_level,
            global_before_level=global_before_level,
            resume_stage_index=0,
            resume_stage_completed=False,
            resume_stage_iters=0,
            preserved_level_stats=[],
            resume_stage_stats=[],
            preserved_level_losses=[],
            resume_stage_losses=[],
        )

    preserved_level_stats = [
        dict(stat)
        for stat in current_level_stats
        if (
            (stage_idx := _stage_index_or_none(stat)) is not None
            and (
                stage_idx < resume_stage_index
                or (stage_idx == resume_stage_index and resume_stage_completed)
            )
        )
    ]
    resume_stage_stats = [
        dict(stat)
        for stat in current_level_stats
        if _stage_index_or_none(stat) == resume_stage_index and not resume_stage_completed
    ]
    level_history = list(loss_hist[-level_completed_before:]) if level_completed_before > 0 else []
    preserved_loss_count = min(len(preserved_level_stats), len(level_history))
    preserved_level_losses = level_history[:preserved_loss_count]
    resume_stage_losses = level_history[preserved_loss_count:]
    if len(resume_stage_losses) > resume_stage_iters:
        resume_stage_losses = resume_stage_losses[-resume_stage_iters:]

    return LevelResumePlan(
        resuming=True,
        level_completed_before=level_completed_before,
        stats_before_level=stats_before_level,
        loss_before_level=loss_before_level,
        global_before_level=global_before_level,
        resume_stage_index=resume_stage_index,
        resume_stage_completed=resume_stage_completed,
        resume_stage_iters=resume_stage_iters,
        preserved_level_stats=preserved_level_stats,
        resume_stage_stats=resume_stage_stats,
        preserved_level_losses=[float(value) for value in preserved_level_losses],
        resume_stage_losses=[float(value) for value in resume_stage_losses],
    )


def _final_multires_volume(
    *,
    x_init: jnp.ndarray | None,
    prev_factor: int | None,
    grid: Grid,
) -> jnp.ndarray:
    if x_init is None:
        return jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    if prev_factor is not None and prev_factor != 1:
        return upsample_volume(x_init, prev_factor, (grid.nx, grid.ny, grid.nz))
    return x_init


def _multires_run_is_complete(
    *,
    params5: jnp.ndarray | None,
    stopped_by_observer: bool,
    resume_state: AlignMultiresResumeState | None,
    last_level_index_processed: int,
    level_count: int,
) -> bool:
    return (
        params5 is not None
        and not stopped_by_observer
        and (
            (resume_state is not None and resume_state.run_complete)
            or last_level_index_processed == level_count - 1
            or level_count == 0
        )
    )


def _emit_level_completion_checkpoint(
    *,
    checkpoint_callback: AlignMultiresCheckpointCallback | None,
    x_lvl: jnp.ndarray,
    params5: jnp.ndarray,
    info: Mapping[str, object],
    level_index: int,
    level_factor: int,
    level_completed_after: int,
    global_outer_idx: int,
    prev_factor: int | None,
    loss_hist: list[float],
    global_outer_stats: list[OuterStat],
    global_elapsed_offset: float,
    level_complete: bool,
    setup_alignment_state: object,
    active_geometry_dofs: tuple[str, ...],
    resolved_schedule: object,
    level_stats: list[OuterStat],
) -> None:
    if checkpoint_callback is None:
        return
    last_stage = resolved_schedule.stages[-1]
    last_stage_iters = sum(
        1 for stat in level_stats if stat.get("schedule_stage_index") == int(last_stage.index)
    )
    checkpoint_callback(
        _build_multires_checkpoint_state(
            x=x_lvl,
            params5=params5,
            motion_coeffs=info.get("motion_coeffs"),
            level_index=int(level_index),
            level_factor=int(level_factor),
            completed_outer_iters_in_level=level_completed_after,
            global_outer_iters_completed=int(global_outer_idx),
            prev_factor=prev_factor,
            loss=list(loss_hist),
            outer_stats=[dict(stat) for stat in global_outer_stats],
            L=info.get("L"),
            small_impr_streak=int(info.get("small_impr_streak", 0)),
            elapsed_offset=float(global_elapsed_offset),
            level_complete=level_complete,
            run_complete=False,
            setup_alignment_state=setup_alignment_state,
            active_geometry_dofs=active_geometry_dofs,
            stage=last_stage,
            stage_completed=level_complete,
            completed_outer_iters_in_stage=last_stage_iters,
        )
    )


def _emit_run_completion_checkpoint(
    *,
    checkpoint_callback: AlignMultiresCheckpointCallback | None,
    params5: jnp.ndarray | None,
    run_complete: bool,
    x_final: jnp.ndarray,
    level_count: int,
    executed_outer_iters: int,
    loss_hist: list[float],
    global_outer_stats: list[OuterStat],
    global_elapsed_offset: float,
    setup_alignment_state: object,
    active_geometry_dofs: tuple[str, ...],
    resolved_schedule: object,
) -> None:
    if checkpoint_callback is None or params5 is None or not run_complete:
        return
    final_stage = resolved_schedule.stages[-1]
    checkpoint_callback(
        _build_multires_checkpoint_state(
            x=x_final,
            params5=params5,
            motion_coeffs=None,
            level_index=max(0, level_count - 1),
            level_factor=1,
            completed_outer_iters_in_level=0,
            global_outer_iters_completed=int(executed_outer_iters),
            prev_factor=1,
            loss=list(loss_hist),
            outer_stats=[dict(stat) for stat in global_outer_stats],
            L=None,
            small_impr_streak=0,
            elapsed_offset=float(global_elapsed_offset),
            level_complete=True,
            run_complete=True,
            setup_alignment_state=setup_alignment_state,
            active_geometry_dofs=active_geometry_dofs,
            stage=final_stage,
            stage_completed=True,
            completed_outer_iters_in_stage=0,
        )
    )


def _final_align_multires_info(
    *,
    loss_hist: list[float],
    factors_list: list[int],
    final_loss_kind: str | None,
    cfg: AlignConfig,
    stopped_by_observer: bool,
    final_observer_action: ObserverAction,
    executed_outer_iters: int,
    global_elapsed_offset: float,
    global_outer_stats: list[OuterStat],
    resolved_schedule: object,
    geometry: Geometry,
    final_pose_model_variables: int | None,
    final_per_view_variables: int | None,
    final_pose_model_basis_shape: list[int] | None,
    active_geometry_dofs: tuple[str, ...],
    final_gauge_fix: str,
    final_gauge_fix_dofs: list[str],
    final_gauge_fix_stats: dict[str, float | str | list[str]] | None,
    setup_alignment_state: object,
) -> AlignMultiresInfo:
    geometry_calibration_diagnostics = add_geometry_acquisition_diagnostics(
        summarize_geometry_calibration_stats(global_outer_stats),
        geometry,
        active_geometry_dofs,
    )
    objective_kinds = [
        str(stat.get("objective_kind") or stat.get("geometry_objective"))
        for stat in global_outer_stats
        if isinstance(stat, Mapping)
        and (stat.get("objective_kind") is not None or stat.get("geometry_objective") is not None)
    ]
    objective_provenance = next(
        (
            dict(stat["objective_provenance"])
            for stat in reversed(global_outer_stats)
            if isinstance(stat, Mapping) and isinstance(stat.get("objective_provenance"), Mapping)
        ),
        None,
    )
    backend_provenance = next(
        (
            dict(stat["backend_provenance"])
            for stat in reversed(global_outer_stats)
            if isinstance(stat, Mapping) and isinstance(stat.get("backend_provenance"), Mapping)
        ),
        None,
    )

    return {
        "loss": loss_hist,
        "factors": factors_list,
        "loss_kind": final_loss_kind,
        "recon_algo": str(cfg.recon_algo),
        "align_profile": str(cfg.align_profile),
        "profile_policy": profile_policy_from_config(cfg).to_dict(),
        "quality_tier": str(cfg.quality_tier),
        "fallback_policy": str(cfg.fallback_policy),
        "stopped_by_observer": stopped_by_observer,
        "observer_action": final_observer_action,
        "total_outer_iters": int(executed_outer_iters),
        "wall_time_total": float(global_elapsed_offset),
        "outer_stats": global_outer_stats,
        "schedule": resolved_schedule.to_dict(),
        "schedule_name": resolved_schedule.name,
        "schedule_stages": [stage.to_dict() for stage in resolved_schedule.stages],
        "gauge_policy": cfg.gauge_policy,
        "gauge_decision": resolved_schedule.gauge_decision.to_dict(),
        "objective_kind": objective_kinds[-1] if objective_kinds else None,
        "objective_kinds": objective_kinds,
        "objective_provenance": objective_provenance,
        "backend_provenance": backend_provenance,
        "pose_model": str(cfg.pose_model),
        "pose_model_variables": final_pose_model_variables,
        "per_view_variables": final_per_view_variables,
        "pose_model_basis_shape": final_pose_model_basis_shape,
        "active_dofs": list(resolved_schedule.active_dofs),
        "active_pose_dofs": list(resolved_schedule.active_pose_dofs),
        "active_geometry_dofs": list(active_geometry_dofs),
        "gauge_fix": final_gauge_fix,
        "gauge_fix_dofs": final_gauge_fix_dofs,
        "gauge_fix_final": final_gauge_fix_stats,
        "geometry_dofs": list(active_geometry_dofs),
        "geometry_calibration_state": (
            _geometry_calibration_payload(setup_alignment_state, active_geometry_dofs)
            if active_geometry_dofs
            else None
        ),
        "geometry_calibration_diagnostics": geometry_calibration_diagnostics,
    }


def _build_multires_context(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    cfg: AlignConfig | None,
    observer: ObserverCallback | None,
    resume_state: AlignMultiresResumeState | None,
    factors: Iterable[int],
) -> MultiresContext:
    cfg = cfg if cfg is not None else AlignConfig()
    observer_fn = adapt_observer_callback(observer) if observer is not None else None
    resolved_schedule = _resolved_schedule_for_cfg(cfg, geometry=geometry)
    setup_base = BaseGeometryArrays.from_geometry(geometry, detector)
    setup_alignment_state = alignment_state_from_checkpoint(
        resume_state.geometry_calibration_state if resume_state is not None else None,
        n_views=int(projections.shape[0]),
        volume=resume_state.x if resume_state is not None else None,
    )
    setup_alignment_state = setup_alignment_state.replace(
        setup=setup_alignment_state.setup.replace(
            nominal_axis_unit=setup_base.nominal_axis_unit,
        )
    )
    validate_grid(grid, "align_multires grid")
    validate_projection_stack(
        projections,
        detector,
        geometry=geometry,
        context="align_multires projections",
    )
    factors_list = [validate_scale_factor(f) for f in factors]
    validate_loss_schedule_levels(cfg.loss, factors_list)
    return MultiresContext(
        cfg=cfg,
        observer_fn=observer_fn,
        resolved_schedule=resolved_schedule,
        active_mask_tuple=resolved_schedule.pose_mask,
        setup_alignment_state=setup_alignment_state,
        active_geometry_dofs=resolved_schedule.active_geometry_dofs,
        factors_list=factors_list,
        levels=_build_multires_levels(geometry, grid, detector, projections, factors_list),
    )


def _build_multires_levels(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    factors_list: list[int],
) -> list[MultiresLevel]:
    levels: list[MultiresLevel] = []
    for factor in factors_list:
        level_grid = scale_grid(grid, factor)
        level_detector = scale_detector(detector, factor)
        level_projections = bin_projections(projections, factor)
        validate_projection_stack(
            level_projections,
            level_detector,
            geometry=geometry,
            context=f"align_multires level factor {factor} projections",
        )
        levels.append(
            {
                "factor": factor,
                "grid": level_grid,
                "detector": level_detector,
                "projections": level_projections,
            }
        )
    return levels


def _initial_multires_run_state(
    *,
    context: MultiresContext,
    resume_state: AlignMultiresResumeState | None,
) -> MultiresRunState:
    level_complete = resume_state is not None and resume_state.level_complete
    x_init = resume_state.x if level_complete else None
    params5 = resume_state.params5 if level_complete else None
    prev_factor = int(resume_state.level_factor) if level_complete else None
    final_gauge_fix = normalize_gauge_fix(context.cfg.gauge_fix)
    return MultiresRunState(
        x_init=x_init,
        params5=params5,
        prev_factor=prev_factor,
        loss_hist=list(resume_state.loss) if resume_state is not None else [],
        global_outer_stats=(
            [dict(stat) for stat in resume_state.outer_stats] if resume_state is not None else []
        ),
        stopped_by_observer=False,
        final_observer_action="continue",
        global_outer_idx=(
            int(resume_state.global_outer_iters_completed) if resume_state is not None else 0
        ),
        global_elapsed_offset=(
            float(resume_state.elapsed_offset) if resume_state is not None else 0.0
        ),
        executed_outer_iters=(
            int(resume_state.global_outer_iters_completed) if resume_state is not None else 0
        ),
        final_pose_model_variables=None,
        final_per_view_variables=None,
        final_pose_model_basis_shape=None,
        final_loss_kind=None,
        final_gauge_fix=final_gauge_fix,
        final_gauge_fix_dofs=list(
            active_gauge_dofs(
                mode=final_gauge_fix,
                active_mask=context.active_mask_tuple,
            )
        ),
        final_gauge_fix_stats=None,
        last_level_index_processed=None,
        setup_alignment_state=context.setup_alignment_state,
    )


def _levels_to_run(
    levels: list[MultiresLevel],
    resume_state: AlignMultiresResumeState | None,
) -> list[tuple[int, MultiresLevel]]:
    if resume_state is not None and resume_state.run_complete:
        return []
    start_level = 0
    if resume_state is not None:
        start_level = int(resume_state.level_index) + (1 if resume_state.level_complete else 0)
    return list(enumerate(levels))[start_level:]


def _level_initial_volume(
    *,
    level: MultiresLevel,
    x_init: jnp.ndarray | None,
    prev_factor: int | None,
    resume_state: AlignMultiresResumeState | None,
    resuming_this_level: bool,
) -> jnp.ndarray | None:
    if resuming_this_level and resume_state is not None:
        return resume_state.x
    if x_init is None or prev_factor is None:
        return None
    level_grid = level["grid"]
    return upsample_volume(
        x_init,
        prev_factor // level["factor"],
        (level_grid.nx, level_grid.ny, level_grid.nz),
    )


def _seed_translation_params(
    *,
    geometry: Geometry,
    cfg: AlignConfig,
    active_mask_tuple: tuple[bool, ...],
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    x0: jnp.ndarray | None,
    params0: jnp.ndarray | None,
) -> jnp.ndarray | None:
    seed_cfg = FistaConfig(
        iters=max(3, cfg.recon_iters // 2),
        lambda_tv=cfg.lambda_tv,
        regulariser=cfg.regulariser,
        huber_delta=cfg.huber_delta,
        projector_unroll=int(cfg.projector_unroll),
        checkpoint_projector=cfg.checkpoint_projector,
        gather_dtype=cfg.gather_dtype,
        recon_rel_tol=cfg.recon_rel_tol,
        recon_patience=(int(cfg.recon_patience) if cfg.recon_patience is not None else 0),
    )
    x_seed, _ = fista_tv(geometry, grid, detector, projections, init_x=x0, config=seed_cfg)
    transforms = stack_view_poses(geometry, projections.shape[0])

    def project_seed_view(transform: jnp.ndarray) -> jnp.ndarray:
        return forward_project_view_T(
            transform,
            grid,
            detector,
            x_seed,
            use_checkpoint=cfg.checkpoint_projector,
            gather_dtype=cfg.gather_dtype,
        )

    preds = jax.vmap(project_seed_view, in_axes=0)(transforms)
    shift_uv = jax.vmap(phase_corr_shift)(preds, projections)
    shifts = jnp.stack(shift_uv, axis=1).astype(jnp.float32)
    seed_params = (
        jnp.zeros((projections.shape[0], 5), dtype=jnp.float32)
        if params0 is None
        else jnp.asarray(params0, dtype=jnp.float32)
    )
    if active_mask_tuple[3]:
        seed_params = seed_params.at[:, 3].set(shifts[:, 0] * jnp.float32(detector.du))
    if active_mask_tuple[4]:
        seed_params = seed_params.at[:, 4].set(shifts[:, 1] * jnp.float32(detector.dv))
    return seed_params


def _params_for_multires_level(
    *,
    level_index: int,
    geometry: Geometry,
    context: MultiresContext,
    level: MultiresLevel,
    x0: jnp.ndarray | None,
    params0: jnp.ndarray | None,
) -> jnp.ndarray | None:
    if level_index != 0 or not context.cfg.seed_translations:
        return params0
    return _seed_translation_params(
        geometry=geometry,
        cfg=context.cfg,
        active_mask_tuple=context.active_mask_tuple,
        grid=level["grid"],
        detector=level["detector"],
        projections=level["projections"],
        x0=x0,
        params0=params0,
    )


def _build_stage_runtime(
    *,
    context: MultiresContext,
    level_index: int,
    level_factor: int,
    level_run: LevelResumePlan,
    state: MultiresRunState,
    level_stats: list[OuterStat],
    level_losses: list[float],
    active_loss_name: str,
    checkpoint_callback: AlignMultiresCheckpointCallback | None,
) -> StageRuntime:
    return StageRuntime(
        level_index=level_index,
        level_factor=level_factor,
        global_before_level=level_run.global_before_level,
        global_elapsed_offset=state.global_elapsed_offset,
        active_loss_name=active_loss_name,
        schedule_name=context.resolved_schedule.name,
        stats_before_level=level_run.stats_before_level,
        loss_before_level=level_run.loss_before_level,
        level_stats=level_stats,
        level_losses=level_losses,
        prev_factor=state.prev_factor,
        observer_fn=context.observer_fn,
        checkpoint_callback=checkpoint_callback,
    )


def _state_after_multires_level(
    *,
    state: MultiresRunState,
    level: MultiresLevel,
    level_index: int,
    level_run: LevelResumePlan,
    stage_result: StageRunResult,
) -> tuple[MultiresRunState, dict[str, object]]:
    info = stage_result.info
    info["loss"] = stage_result.level_losses
    info["outer_stats"] = [
        dict(stat)
        for stat in stage_result.level_stats
        if not str(stat.get("geometry_block") or "").startswith("setup_")
    ]
    info["completed_outer_iters"] = len(stage_result.level_stats)
    info["wall_time_total"] = float(stage_result.level_wall_time)
    info["observer_action"] = stage_result.level_action
    level_completed_after = len(stage_result.level_stats)
    return (
        replace(
            state,
            x_init=stage_result.x_lvl,
            params5=stage_result.params5,
            prev_factor=level["factor"],
            loss_hist=level_run.loss_before_level + stage_result.level_losses,
            global_outer_stats=level_run.stats_before_level + stage_result.level_stats,
            stopped_by_observer=stage_result.level_action == "stop_run",
            final_observer_action=stage_result.level_action,
            global_outer_idx=level_run.global_before_level + level_completed_after,
            executed_outer_iters=level_run.global_before_level + level_completed_after,
            global_elapsed_offset=state.global_elapsed_offset + float(stage_result.level_wall_time),
            final_pose_model_variables=int(info.get("pose_model_variables") or 0),
            final_per_view_variables=int(info.get("per_view_variables") or 0),
            final_pose_model_basis_shape=list(info.get("pose_model_basis_shape") or []),
            final_loss_kind=str(info.get("loss_kind", state.final_loss_kind or "")) or None,
            final_gauge_fix=str(info.get("gauge_fix", stage_result.final_gauge_fix)),
            final_gauge_fix_dofs=list(
                info.get("gauge_fix_dofs", stage_result.final_gauge_fix_dofs)
            ),
            final_gauge_fix_stats=dict(info.get("gauge_fix_final", {}) or {}),
            last_level_index_processed=level_index,
            setup_alignment_state=stage_result.setup_alignment_state,
        ),
        info,
    )
