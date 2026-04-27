from __future__ import annotations

from dataclasses import replace
import logging
import time
from typing import Iterable, Mapping, TypedDict

import jax
import jax.numpy as jnp

from ..core.geometry.base import Geometry, Grid, Detector
from ..core.geometry.views import stack_view_poses
from ..core.projector import forward_project_view_T
from ..core.validation import (
    validate_grid,
    validate_projection_stack,
)
from ..recon.fista_tv import FistaConfig, fista_tv
from ._loss_specs import (
    loss_spec_name,
    resolve_loss_for_level,
    validate_loss_schedule_levels,
)
from .gauge import (
    active_gauge_dofs,
    normalize_gauge_fix,
)
from .geometry_applier import BaseGeometryArrays, apply_setup_to_detector_grid
from .schedules import (
    ResolvedAlignmentStage,
)
from .state import PoseState, alignment_state_from_checkpoint
from .geometry_blocks import (
    add_geometry_acquisition_diagnostics,
    summarize_geometry_calibration_stats,
)
from ._observer import (
    ObserverAction,
    ObserverCallback,
    OuterStat,
    _normalize_observer_action,
    adapt_legacy_observer,
)
from ._results import (
    AlignMultiresInfo,
    AlignMultiresCheckpointCallback,
    AlignMultiresResumeState,
    AlignResumeState,
    enrich_multires_stage_stat as _enrich_multires_stage_stat,
)
from ._pose_stage import align
from ._setup_stage import (
    _geometry_calibration_payload,
    _geometry_with_setup_state,
    _optimize_setup_geometry_bilevel_for_level,
)
from ._config import (
    AlignConfig,
    _resolved_schedule_for_cfg,
)


class MultiresLevel(TypedDict):
    factor: int
    grid: Grid
    detector: Detector
    projections: jnp.ndarray







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
    from ..core.multires import (
        bin_projections,
        scale_detector,
        scale_grid,
        upsample_volume,
        validate_scale_factor,
    )

    if cfg is None:
        cfg = AlignConfig()
    observer_fn = adapt_legacy_observer(observer) if observer is not None else None
    resolved_schedule = _resolved_schedule_for_cfg(cfg, geometry=geometry)
    active_mask_tuple = resolved_schedule.pose_mask
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
    active_geometry_dofs = resolved_schedule.active_geometry_dofs

    validate_grid(grid, "align_multires grid")
    validate_projection_stack(
        projections,
        detector,
        geometry=geometry,
        context="align_multires projections",
    )

    factors_list = [validate_scale_factor(f) for f in factors]
    validate_loss_schedule_levels(cfg.loss, factors_list)
    levels: list[MultiresLevel] = []
    for f in factors_list:
        g = scale_grid(grid, f)
        d = scale_detector(detector, f)
        y = bin_projections(projections, f)
        validate_projection_stack(
            y,
            d,
            geometry=geometry,
            context=f"align_multires level factor {f} projections",
        )
        levels.append(
            {
                "factor": f,
                "grid": g,
                "detector": d,
                "projections": y,
            }
        )

    x_init = resume_state.x if resume_state is not None and resume_state.level_complete else None
    params5 = (
        resume_state.params5 if resume_state is not None and resume_state.level_complete else None
    )
    prev_factor: int | None = (
        int(resume_state.level_factor)
        if resume_state is not None and resume_state.level_complete
        else None
    )
    loss_hist: list[float] = list(resume_state.loss) if resume_state is not None else []
    global_outer_stats: list[OuterStat] = (
        [dict(stat) for stat in resume_state.outer_stats] if resume_state is not None else []
    )
    stopped_by_observer = False
    final_observer_action: ObserverAction = "continue"
    global_outer_idx = (
        int(resume_state.global_outer_iters_completed) if resume_state is not None else 0
    )
    global_elapsed_offset = float(resume_state.elapsed_offset) if resume_state is not None else 0.0
    executed_outer_iters = int(global_outer_idx)
    final_pose_model_variables: int | None = None
    final_per_view_variables: int | None = None
    final_pose_model_basis_shape: list[int] | None = None
    final_loss_kind: str | None = None
    final_gauge_fix = normalize_gauge_fix(cfg.gauge_fix)
    final_gauge_fix_dofs = list(
        active_gauge_dofs(mode=final_gauge_fix, active_mask=active_mask_tuple)
    )
    final_gauge_fix_stats: dict[str, float | str | list[str]] | None = None
    last_level_index_processed: int | None = None

    if resume_state is not None and resume_state.run_complete:
        levels_to_run: list[tuple[int, MultiresLevel]] = []
        x_init = resume_state.x
        params5 = resume_state.params5
        prev_factor = 1
    else:
        start_level = 0
        if resume_state is not None:
            start_level = int(resume_state.level_index) + (1 if resume_state.level_complete else 0)
        levels_to_run = list(enumerate(levels))[start_level:]

    for li, lvl in levels_to_run:
        g = lvl["grid"]
        d = lvl["detector"]
        y = lvl["projections"]
        active_loss_spec = resolve_loss_for_level(cfg.loss, int(lvl["factor"]))
        active_loss_name = loss_spec_name(active_loss_spec)
        final_loss_kind = active_loss_name
        logging.info(
            "Alignment level %d/%d factor=%d using loss=%s",
            int(li) + 1,
            len(levels),
            int(lvl["factor"]),
            active_loss_name,
        )
        resuming_this_level = (
            resume_state is not None
            and not resume_state.level_complete
            and int(resume_state.level_index) == int(li)
        )
        if resuming_this_level:
            x0 = resume_state.x
        elif x_init is not None and prev_factor is not None:
            # Upsample previous x to current level as init
            f_up = prev_factor // lvl["factor"]
            x0 = upsample_volume(x_init, f_up, (g.nx, g.ny, g.nz))
        else:
            x0 = None

        # Optional translation seeding at the coarsest level via phase correlation
        params0 = resume_state.params5 if resuming_this_level else params5
        if li == 0 and cfg.seed_translations:
            # quick seed recon to project nominal poses
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
            x_seed, _ = fista_tv(
                geometry,
                g,
                d,
                y,
                init_x=x0,
                config=seed_cfg,
            )
            T_nom = stack_view_poses(geometry, y.shape[0])
            from ..utils.phasecorr import phase_corr_shift

            vm_pred = jax.vmap(
                lambda T: forward_project_view_T(
                    T,
                    g,
                    d,
                    x_seed,
                    use_checkpoint=cfg.checkpoint_projector,
                    gather_dtype=cfg.gather_dtype,
                ),
                in_axes=0,
            )
            preds = vm_pred(T_nom)
            shift_uv = jax.vmap(phase_corr_shift)(preds, y)  # returns (du, dv)
            shifts = jnp.stack(shift_uv, axis=1).astype(jnp.float32)  # (n, 2)
            # Convert pixel shifts to world units using detector spacing
            dx = shifts[:, 0] * jnp.float32(d.du)
            dz = shifts[:, 1] * jnp.float32(d.dv)
            seed_params = (
                jnp.zeros((y.shape[0], 5), dtype=jnp.float32)
                if params0 is None
                else jnp.asarray(params0, dtype=jnp.float32)
            )
            if active_mask_tuple[3]:
                seed_params = seed_params.at[:, 3].set(dx)
            if active_mask_tuple[4]:
                seed_params = seed_params.at[:, 4].set(dz)
            params0 = seed_params

        level_completed_before = (
            int(resume_state.completed_outer_iters_in_level) if resuming_this_level else 0
        )
        current_level_stats = [
            dict(stat) for stat in global_outer_stats if stat.get("level_index") == int(li)
        ]
        stats_before_level = [
            dict(stat) for stat in global_outer_stats if stat.get("level_index") != int(li)
        ]
        loss_before_level = (
            list(loss_hist[:-level_completed_before])
            if resuming_this_level and level_completed_before > 0
            else list(loss_hist)
        )
        global_before_level = (
            int(executed_outer_iters) - level_completed_before
            if resuming_this_level
            else int(executed_outer_iters)
        )
        resume_stage_index = int(resume_state.stage_index) if resuming_this_level else 0
        resume_stage_completed = (
            bool(resume_state.stage_completed) if resuming_this_level else False
        )
        resume_stage_iters = (
            int(resume_state.completed_outer_iters_in_stage) if resuming_this_level else 0
        )

        def _stat_stage_index(stat: Mapping[str, object]) -> int | None:
            try:
                return int(stat["schedule_stage_index"])
            except Exception:
                return None

        if resuming_this_level:
            preserved_level_stats = [
                dict(stat)
                for stat in current_level_stats
                if (
                    (stage_idx := _stat_stage_index(stat)) is not None
                    and (
                        stage_idx < resume_stage_index
                        or (stage_idx == resume_stage_index and resume_stage_completed)
                    )
                )
            ]
            resume_stage_stats = [
                dict(stat)
                for stat in current_level_stats
                if _stat_stage_index(stat) == resume_stage_index and not resume_stage_completed
            ]
            level_history = (
                list(loss_hist[-level_completed_before:])
                if level_completed_before > 0
                else []
            )
            preserved_loss_count = min(len(preserved_level_stats), len(level_history))
            preserved_level_losses = level_history[:preserved_loss_count]
            resume_stage_losses = level_history[preserved_loss_count:]
            if len(resume_stage_losses) > resume_stage_iters:
                resume_stage_losses = resume_stage_losses[-resume_stage_iters:]
        else:
            preserved_level_stats = []
            resume_stage_stats = []
            preserved_level_losses = []
            resume_stage_losses = []
        level_stats: list[OuterStat] = [dict(stat) for stat in preserved_level_stats]
        level_losses: list[float] = [float(value) for value in preserved_level_losses]
        level_wall_time = 0.0
        level_action: ObserverAction = "continue"
        x_lvl = x0 if x0 is not None else jnp.zeros((g.nx, g.ny, g.nz), dtype=jnp.float32)
        params5 = (
            params0
            if params0 is not None
            else jnp.zeros((y.shape[0], 5), dtype=jnp.float32)
        )
        info: dict[str, object] = {
            "loss": [],
            "loss_kind": active_loss_name,
            "recon_algo": str(cfg.recon_algo),
            "L": None,
            "outer_stats": [],
            "stopped_by_observer": False,
            "observer_action": "continue",
            "wall_time_total": 0.0,
            "pose_model": str(cfg.pose_model),
            "pose_model_variables": 0,
            "per_view_variables": 0,
            "pose_model_basis_shape": [],
            "active_dofs": [],
            "completed_outer_iters": 0,
            "small_impr_streak": 0,
            "motion_coeffs": None,
            "gauge_fix": final_gauge_fix,
            "gauge_fix_dofs": final_gauge_fix_dofs,
            "gauge_fix_final": final_gauge_fix_stats or {},
        }

        def _enrich_level_stats(
            local_stats: list[OuterStat],
            *,
            stage: ResolvedAlignmentStage | None = None,
            global_start: int | None = None,
        ) -> list[OuterStat]:
            enriched_stats: list[OuterStat] = []
            for idx, stat in enumerate(local_stats, start=1):
                enriched = _enrich_multires_stage_stat(
                    stat,
                    level_factor=int(lvl["factor"]),
                    level_index=int(li),
                    global_outer_idx=int(
                        (global_before_level if global_start is None else global_start) + idx
                    ),
                    elapsed_offset=float(global_elapsed_offset),
                    loss_name=active_loss_name,
                    schedule_name=resolved_schedule.name,
                    stage=stage,
                )
                enriched_stats.append(enriched)
            return enriched_stats

        def _emit_multires_checkpoint(
            state: AlignResumeState,
            *,
            level_complete: bool,
            stage: ResolvedAlignmentStage,
            global_start: int,
        ) -> None:
            if checkpoint_callback is None:
                return
            enriched_stats = _enrich_level_stats(
                [dict(stat) for stat in state.outer_stats],
                stage=stage,
                global_start=global_start,
            )
            checkpoint_callback(
                AlignMultiresResumeState(
                    x=state.x,
                    params5=state.params5,
                    motion_coeffs=state.motion_coeffs,
                    level_index=int(li),
                    level_factor=int(lvl["factor"]),
                    completed_outer_iters_in_level=int(
                        len(level_stats) + state.start_outer_iter
                    ),
                    global_outer_iters_completed=int(
                        global_before_level + len(level_stats) + state.start_outer_iter
                    ),
                    prev_factor=prev_factor,
                    loss=loss_before_level + level_losses + list(state.loss),
                    outer_stats=stats_before_level + level_stats + enriched_stats,
                    L=state.L,
                    small_impr_streak=int(state.small_impr_streak),
                    elapsed_offset=float(global_elapsed_offset + state.elapsed_offset),
                    level_complete=bool(level_complete),
                    run_complete=False,
                    geometry_calibration_state=_geometry_calibration_payload(
                        setup_alignment_state,
                        active_geometry_dofs,
                    ),
                    stage_index=int(stage.index),
                    stage_name=stage.name,
                    stage_completed=bool(
                        level_complete or int(state.start_outer_iter) >= int(stage.maxiter)
                    ),
                    completed_outer_iters_in_stage=int(state.start_outer_iter),
                )
            )

        def _stage_observer(
            stage: ResolvedAlignmentStage,
            global_start: int,
            x_obs,
            params_obs,
            stat_obs,
        ):
            nonlocal stopped_by_observer
            enriched = _enrich_multires_stage_stat(
                stat_obs,
                level_factor=int(lvl["factor"]),
                level_index=int(li),
                global_outer_idx=int(global_start + int(stat_obs["outer_idx"])),
                elapsed_offset=float(global_elapsed_offset),
                loss_name=active_loss_name,
                schedule_name=resolved_schedule.name,
                stage=stage,
            )
            if observer_fn is None:
                return "continue"
            return observer_fn(x_obs, params_obs, enriched)

        stage_resume_consumed = False
        for stage in resolved_schedule.stages:
            if resuming_this_level:
                if int(stage.index) < resume_stage_index:
                    continue
                if int(stage.index) == resume_stage_index and resume_stage_completed:
                    continue
            stage_global_start = global_before_level + len(level_stats)
            if stage.active_geometry_dofs:
                cfg_stage = replace(
                    cfg,
                    schedule=None,
                    optimise_dofs=stage.active_geometry_dofs,
                    geometry_dofs=(),
                    outer_iters=int(stage.maxiter),
                    early_stop=bool(stage.early_stop),
                )
                geometry_start = time.perf_counter()
                x_lvl, setup_alignment_state, raw_geometry_stats = (
                    _optimize_setup_geometry_bilevel_for_level(
                        geometry=geometry,
                        grid=g,
                        detector=d,
                        projections=y,
                        init_x=x_lvl,
                        init_params5=params5,
                        state=setup_alignment_state,
                        active_geometry_dofs=stage.active_geometry_dofs,
                        factor=int(lvl["factor"]),
                        cfg=cfg_stage,
                        loss_spec=active_loss_spec,
                        loss_name=active_loss_name,
                        schedule_name=resolved_schedule.name,
                        stage=stage,
                    )
                )
                stage_wall = time.perf_counter() - geometry_start
                level_wall_time += stage_wall
                enriched = _enrich_level_stats(
                    [dict(stat) for stat in raw_geometry_stats],
                    stage=stage,
                    global_start=stage_global_start,
                )
                level_stats.extend(enriched)
                level_losses.extend(
                    float(stat["geometry_loss_after"])
                    for stat in enriched
                    if stat.get("geometry_loss_after") is not None
                )
                info["wall_time_total"] = float(level_wall_time)
                info["completed_outer_iters"] = len(level_stats)
                continue

            if stage.active_pose_dofs:
                pose_optimizer = (
                    stage.optimizer_kind
                    if stage.optimizer_kind in {"gd", "gn", "lbfgs"}
                    else cfg.opt_method
                )
                pose_gauge_fix = (
                    "mean_translation"
                    if stage.gauge_policy == "anchor_mean"
                    else cfg.gauge_fix
                )
                cfg_stage = replace(
                    cfg,
                    schedule=None,
                    optimise_dofs=stage.active_pose_dofs,
                    geometry_dofs=(),
                    opt_method=str(pose_optimizer),
                    outer_iters=int(stage.maxiter),
                    early_stop=bool(stage.early_stop),
                    recon_L=None,
                    loss=active_loss_spec,
                    gauge_fix=pose_gauge_fix,
                )
                align_kwargs = {}
                if active_geometry_dofs:
                    geometry_for_align = _geometry_with_setup_state(
                        geometry,
                        g,
                        d,
                        setup_alignment_state.setup,
                    )
                    det_grid_for_align = apply_setup_to_detector_grid(
                        d,
                        setup_alignment_state.setup,
                        level_factor=int(lvl["factor"]),
                    )
                    align_kwargs["det_grid_override"] = det_grid_for_align
                else:
                    geometry_for_align = geometry
                align_resume_state = None
                if (
                    resuming_this_level
                    and int(stage.index) == resume_stage_index
                    and not resume_stage_completed
                    and not stage_resume_consumed
                ):
                    align_resume_state = AlignResumeState(
                        x=resume_state.x,
                        params5=resume_state.params5,
                        motion_coeffs=resume_state.motion_coeffs,
                        start_outer_iter=resume_stage_iters,
                        loss=list(resume_stage_losses),
                        outer_stats=[dict(stat) for stat in resume_stage_stats],
                        L=resume_state.L,
                        small_impr_streak=int(resume_state.small_impr_streak),
                        elapsed_offset=float(
                            resume_state.elapsed_offset - global_elapsed_offset
                        ),
                    )
                    stage_resume_consumed = True
                x_lvl, params5, info = align(
                    geometry_for_align,
                    g,
                    d,
                    y,
                    cfg=cfg_stage,
                    init_x=x_lvl,
                    init_params5=params5,
                    observer=(
                        (lambda x_obs, params_obs, stat_obs, _stage=stage, _start=stage_global_start: _stage_observer(_stage, _start, x_obs, params_obs, stat_obs))
                        if observer_fn is not None
                        else None
                    ),
                    resume_state=align_resume_state,
                    checkpoint_callback=(
                        (
                            lambda state, _stage=stage, _start=stage_global_start: _emit_multires_checkpoint(
                                state,
                                level_complete=False,
                                stage=_stage,
                                global_start=_start,
                            )
                        )
                        if checkpoint_callback is not None
                        else None
                    ),
                    **align_kwargs,
                )
                setup_alignment_state = setup_alignment_state.replace(
                    pose=PoseState(
                        params5,
                        info.get("motion_coeffs"),  # type: ignore[arg-type]
                    ),
                    volume=x_lvl,
                )
                enriched = _enrich_level_stats(
                    [dict(stat) for stat in info.get("outer_stats", [])],
                    stage=stage,
                    global_start=stage_global_start,
                )
                level_stats.extend(enriched)
                level_losses.extend(float(v) for v in info.get("loss", []))
                try:
                    level_wall_time += float(info.get("wall_time_total") or 0.0)
                except Exception:
                    pass
                level_action = _normalize_observer_action(info.get("observer_action"))
                final_gauge_fix = str(info.get("gauge_fix", final_gauge_fix))
                final_gauge_fix_dofs = list(info.get("gauge_fix_dofs", final_gauge_fix_dofs))
                final_gauge_fix_stats = dict(info.get("gauge_fix_final", {}) or {})
                if level_action != "continue":
                    break

        info["loss"] = level_losses
        info["outer_stats"] = [
            dict(stat)
            for stat in level_stats
            if not str(stat.get("geometry_block") or "").startswith("setup_")
        ]
        info["completed_outer_iters"] = len(level_stats)
        info["wall_time_total"] = float(level_wall_time)
        info["observer_action"] = level_action
        level_completed_after = len(level_stats)
        level_complete = (
            level_completed_after >= sum(int(stage.maxiter) for stage in resolved_schedule.stages)
            or level_action == "advance_level"
            or not bool(info.get("stopped_by_observer", False))
        )
        loss_hist = loss_before_level + level_losses
        global_outer_stats = stats_before_level + level_stats
        global_outer_idx = global_before_level + level_completed_after
        executed_outer_iters = int(global_outer_idx)
        final_pose_model_variables = int(info.get("pose_model_variables") or 0)
        final_per_view_variables = int(info.get("per_view_variables") or 0)
        final_pose_model_basis_shape = list(info.get("pose_model_basis_shape") or [])
        final_gauge_fix = str(info.get("gauge_fix", final_gauge_fix))
        final_gauge_fix_dofs = list(info.get("gauge_fix_dofs", final_gauge_fix_dofs))
        final_gauge_fix_stats = dict(info.get("gauge_fix_final", {}) or {})
        x_init = x_lvl
        prev_factor = lvl["factor"]
        last_level_index_processed = int(li)
        global_elapsed_offset += float(level_wall_time)
        if checkpoint_callback is not None:
            last_stage = resolved_schedule.stages[-1]
            last_stage_iters = sum(
                1
                for stat in level_stats
                if stat.get("schedule_stage_index") == int(last_stage.index)
            )
            checkpoint_callback(
                AlignMultiresResumeState(
                    x=x_lvl,
                    params5=params5,
                    motion_coeffs=info.get("motion_coeffs"),
                    level_index=int(li),
                    level_factor=int(lvl["factor"]),
                    completed_outer_iters_in_level=level_completed_after,
                    global_outer_iters_completed=int(global_outer_idx),
                    prev_factor=prev_factor,
                    loss=list(loss_hist),
                    outer_stats=[dict(stat) for stat in global_outer_stats],
                    L=info.get("L"),
                    small_impr_streak=int(info.get("small_impr_streak", 0)),
                    elapsed_offset=float(global_elapsed_offset),
                    level_complete=bool(level_complete),
                    run_complete=False,
                    geometry_calibration_state=_geometry_calibration_payload(
                        setup_alignment_state,
                        active_geometry_dofs,
                    ),
                    stage_index=int(last_stage.index),
                    stage_name=last_stage.name,
                    stage_completed=bool(level_complete),
                    completed_outer_iters_in_stage=last_stage_iters,
                )
            )
        final_observer_action = level_action
        if level_action == "stop_run":
            stopped_by_observer = True
            break
        if level_action == "advance_level":
            continue

    # Always return a full-resolution-compatible final volume.
    if x_init is None:
        x_final = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    elif prev_factor is not None and prev_factor != 1:
        x_final = upsample_volume(x_init, prev_factor, (grid.nx, grid.ny, grid.nz))
    else:
        x_final = x_init

    run_complete = (
        params5 is not None
        and not stopped_by_observer
        and (
            (resume_state is not None and resume_state.run_complete)
            or last_level_index_processed == len(levels) - 1
            or not levels
        )
    )
    if checkpoint_callback is not None and params5 is not None and run_complete:
        checkpoint_callback(
            AlignMultiresResumeState(
                x=x_final,
                params5=params5,
                motion_coeffs=None,
                level_index=max(0, len(levels) - 1),
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
                geometry_calibration_state=_geometry_calibration_payload(
                    setup_alignment_state,
                    active_geometry_dofs,
                ),
                stage_index=int(resolved_schedule.stages[-1].index),
                stage_name=resolved_schedule.stages[-1].name,
                stage_completed=True,
                completed_outer_iters_in_stage=0,
            )
        )

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

    return (
        x_final,
        params5 if params5 is not None else jnp.zeros((projections.shape[0], 5), jnp.float32),
        {
            "loss": loss_hist,
            "factors": factors_list,
            "loss_kind": final_loss_kind,
            "recon_algo": str(cfg.recon_algo),
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
        },
    )
