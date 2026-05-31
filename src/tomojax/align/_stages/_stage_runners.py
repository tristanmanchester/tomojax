from __future__ import annotations

from dataclasses import replace
import time

import jax.numpy as jnp

from tomojax.align._config import AlignConfig
from tomojax.align._geometry.geometry_applier import (
    BaseGeometryArrays,
    apply_alignment_state,
    apply_setup_to_detector_grid,
)
from tomojax.align._model.dofs import DOF_INDEX
from tomojax.align._model.schedules import ResolvedAlignmentStage
from tomojax.align._model.state import PoseState
from tomojax.align._objectives.loss_adapters import build_loss_adapter
from tomojax.align._observer import OuterStat, _normalize_observer_action
from tomojax.align._pose._pose_loop import align
from tomojax.align._profiles import profile_policy_from_config
from tomojax.align._results import AlignMultiresResumeState, AlignResumeState
from tomojax.align.proposals import ProposalCandidate, score_pose_stack_candidates
from tomojax.core.geometry.base import Detector, Geometry, Grid

from ._setup_stage import _geometry_with_setup_state, _optimize_setup_geometry_bilevel_for_level
from ._stage_types import (
    _PROPOSAL_STEP_BY_DOF,
    LevelResumePlan,
    StageLoopState,
    StageRunResult,
    StageRuntime,
    _accumulate_stage_wall_time,
    _pose_stage_optimizer_or_raise,
)


def _proposal_candidates_for_pose_stage(
    *,
    stage: ResolvedAlignmentStage,
    base: BaseGeometryArrays,
    setup_alignment_state: object,
    params5: jnp.ndarray,
    volume: jnp.ndarray,
) -> tuple[ProposalCandidate, ...]:
    state = setup_alignment_state.replace(
        pose=PoseState(params5),
        volume=volume,
    )
    baseline = apply_alignment_state(base, state).pose_stack
    candidates: list[ProposalCandidate] = [
        ProposalCandidate(
            "baseline",
            baseline,
            {
                "kind": "baseline",
                "active_pose_dofs": list(stage.active_pose_dofs),
            },
        )
    ]
    for dof in stage.active_pose_dofs:
        col = DOF_INDEX.get(dof)
        if col is None:
            continue
        step = float(_PROPOSAL_STEP_BY_DOF.get(dof, 0.0))
        if step <= 0.0:
            continue
        for sign, label in ((-1.0, "minus"), (1.0, "plus")):
            candidate_params = params5.at[:, col].add(jnp.float32(sign * step))
            candidate_state = setup_alignment_state.replace(
                pose=PoseState(candidate_params),
                volume=volume,
            )
            candidates.append(
                ProposalCandidate(
                    f"{dof}_{label}",
                    apply_alignment_state(base, candidate_state).pose_stack,
                    {
                        "kind": "uniform_pose_perturbation",
                        "dof": dof,
                        "delta": float(sign * step),
                    },
                )
            )
    return tuple(candidates)


def _apply_pose_proposal_stage(
    *,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    cfg: AlignConfig,
    stage: ResolvedAlignmentStage,
    active_loss_spec: object,
    setup_alignment_state: object,
    level_factor: int,
    x_lvl: jnp.ndarray,
    params5: jnp.ndarray,
) -> tuple[jnp.ndarray, dict[str, object]]:
    if not stage.active_pose_dofs:
        return params5, {
            "proposal_status": "skipped",
            "proposal_reason": "proposal stage has no active pose DOFs",
        }
    base = BaseGeometryArrays.from_geometry(
        geometry,
        detector,
        level_factor=int(level_factor),
    )
    effective = apply_alignment_state(
        base,
        setup_alignment_state.replace(pose=PoseState(params5), volume=x_lvl),
    )
    adapter = build_loss_adapter(active_loss_spec, projections)
    candidates = _proposal_candidates_for_pose_stage(
        stage=stage,
        base=base,
        setup_alignment_state=setup_alignment_state,
        params5=params5,
        volume=x_lvl,
    )
    result = score_pose_stack_candidates(
        candidates=candidates,
        grid=grid,
        detector=detector,
        volume=x_lvl,
        det_grid=effective.det_grid,
        targets=projections,
        loss_adapter=adapter,
        projector_backend=cfg.projector_backend,
        gather_dtype=cfg.gather_dtype,
        views_per_batch=cfg.views_per_batch,
        projector_unroll=cfg.projector_unroll,
        checkpoint_projector=cfg.checkpoint_projector,
    )
    best_params = params5
    if result.improved:
        best_meta = result.candidate_metadata[result.best_index]
        dof = best_meta.get("dof")
        delta = best_meta.get("delta")
        if isinstance(dof, str) and dof in DOF_INDEX and delta is not None:
            best_params = params5.at[:, DOF_INDEX[dof]].add(jnp.float32(float(delta)))
    return best_params, result.to_dict()


def _run_proposal_stage(
    *,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    cfg: AlignConfig,
    stage: ResolvedAlignmentStage,
    active_loss_spec: object,
    setup_alignment_state: object,
    level_factor: int,
    stage_runtime: StageRuntime,
    stage_global_start: int,
    level_stats: list[OuterStat],
    level_losses: list[float],
    state: StageLoopState,
) -> StageLoopState:
    proposal_start = time.perf_counter()
    loss_before = float(level_losses[-1]) if level_losses else None
    params5, proposal_info = _apply_pose_proposal_stage(
        geometry=geometry,
        grid=grid,
        detector=detector,
        projections=projections,
        cfg=cfg,
        stage=stage,
        active_loss_spec=active_loss_spec,
        setup_alignment_state=setup_alignment_state,
        level_factor=int(level_factor),
        x_lvl=state.x_lvl,
        params5=state.params5,
    )
    setup_alignment_state = setup_alignment_state.replace(
        pose=PoseState(params5),
        volume=state.x_lvl,
    )
    proposal_wall_time = time.perf_counter() - proposal_start
    level_wall_time = state.level_wall_time + proposal_wall_time
    proposal_loss = proposal_info.get("best_value")
    if proposal_loss is not None:
        level_losses.append(float(proposal_loss))
    stat: OuterStat = {
        "outer_idx": 1,
        "loss": float(proposal_loss) if proposal_loss is not None else None,
        "loss_before": loss_before,
        "loss_after": float(proposal_loss) if proposal_loss is not None else None,
        "proposal_status": str(proposal_info.get("proposal_status") or "scored"),
        "proposal_best_name": str(proposal_info.get("best_name") or ""),
        "proposal_best_index": int(proposal_info.get("best_index") or 0),
        "proposal_improved": bool(proposal_info.get("improved") or False),
        "proposal_candidate_count": len(proposal_info.get("values") or []),
        "proposal_backend_provenance": dict(proposal_info.get("backend_provenance") or {}),
        "wall_time": float(proposal_wall_time),
    }
    level_stats.extend(
        stage_runtime.enrich_stats([stat], stage=stage, global_start=stage_global_start)
    )
    info = dict(state.info)
    info["wall_time_total"] = float(level_wall_time)
    info["completed_outer_iters"] = len(level_stats)
    info["proposal"] = dict(proposal_info)
    return replace(
        state,
        params5=params5,
        info=info,
        setup_alignment_state=setup_alignment_state,
        level_wall_time=level_wall_time,
    )


def _run_setup_geometry_stage(
    *,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    cfg: AlignConfig,
    stage: ResolvedAlignmentStage,
    resolved_schedule: object,
    active_loss_spec: object,
    active_loss_name: str,
    level_factor: int,
    stage_runtime: StageRuntime,
    stage_global_start: int,
    level_stats: list[OuterStat],
    level_losses: list[float],
    state: StageLoopState,
) -> StageLoopState:
    cfg_stage = replace(
        cfg,
        schedule=None,
        optimise_dofs=stage.active_geometry_dofs,
        outer_iters=int(stage.maxiter),
        early_stop=bool(stage.early_stop),
    )
    geometry_start = time.perf_counter()
    setup_result = _optimize_setup_geometry_bilevel_for_level(
        geometry=geometry,
        grid=grid,
        detector=detector,
        projections=projections,
        init_x=state.x_lvl,
        init_params5=state.params5,
        state=state.setup_alignment_state,
        active_geometry_dofs=stage.active_geometry_dofs,
        factor=int(level_factor),
        cfg=cfg_stage,
        loss_spec=active_loss_spec,
        loss_name=active_loss_name,
        schedule_name=resolved_schedule.name,
        stage=stage,
    )
    level_wall_time = state.level_wall_time + (time.perf_counter() - geometry_start)
    enriched = stage_runtime.enrich_stats(
        [dict(stat) for stat in setup_result.checkpoint_outer_stats],
        stage=stage,
        global_start=stage_global_start,
    )
    level_stats.extend(enriched)
    level_losses.extend(setup_result.losses)
    info = dict(state.info)
    info["wall_time_total"] = float(level_wall_time)
    info["completed_outer_iters"] = len(level_stats)
    return replace(
        state,
        x_lvl=setup_result.x,
        info=info,
        setup_alignment_state=setup_result.state,
        level_wall_time=level_wall_time,
    )


def _pose_stage_geometry_context(
    *,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    setup_alignment_state: object,
    active_geometry_dofs: tuple[str, ...],
    level_factor: int,
) -> tuple[Geometry, dict[str, object]]:
    if not active_geometry_dofs:
        return geometry, {}
    return (
        _geometry_with_setup_state(
            geometry,
            grid,
            detector,
            setup_alignment_state.setup,
        ),
        {
            "det_grid_override": apply_setup_to_detector_grid(
                detector,
                setup_alignment_state.setup,
                level_factor=int(level_factor),
            )
        },
    )


def _pose_stage_resume_state(
    *,
    stage: ResolvedAlignmentStage,
    resume_state: AlignMultiresResumeState | None,
    resuming_this_level: bool,
    level_resume: LevelResumePlan,
    global_elapsed_offset: float,
    stage_resume_consumed: bool,
) -> tuple[AlignResumeState | None, bool]:
    should_resume = (
        resuming_this_level
        and resume_state is not None
        and int(stage.index) == level_resume.resume_stage_index
        and not level_resume.resume_stage_completed
        and not stage_resume_consumed
    )
    if not should_resume or resume_state is None:
        return None, stage_resume_consumed
    return (
        AlignResumeState(
            x=resume_state.x,
            params5=resume_state.params5,
            motion_coeffs=resume_state.motion_coeffs,
            start_outer_iter=level_resume.resume_stage_iters,
            loss=list(level_resume.resume_stage_losses),
            outer_stats=[dict(stat) for stat in level_resume.resume_stage_stats],
            L=resume_state.L,
            small_impr_streak=int(resume_state.small_impr_streak),
            elapsed_offset=float(resume_state.elapsed_offset - global_elapsed_offset),
        ),
        True,
    )


def _run_pose_alignment_stage(
    *,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    cfg: AlignConfig,
    stage: ResolvedAlignmentStage,
    active_loss_spec: object,
    active_geometry_dofs: tuple[str, ...],
    level_factor: int,
    stage_runtime: StageRuntime,
    stage_global_start: int,
    resume_state: AlignMultiresResumeState | None,
    resuming_this_level: bool,
    level_resume: LevelResumePlan,
    global_elapsed_offset: float,
    level_stats: list[OuterStat],
    level_losses: list[float],
    state: StageLoopState,
) -> StageLoopState:
    pose_optimizer = _pose_stage_optimizer_or_raise(stage)
    pose_gauge_fix = "mean_translation" if stage.gauge_policy == "anchor_mean" else cfg.gauge_fix
    cfg_stage = replace(
        cfg,
        schedule=None,
        optimise_dofs=stage.active_pose_dofs,
        opt_method=str(pose_optimizer),
        outer_iters=int(stage.maxiter),
        recon_iters=int(cfg.recon_iters),
        early_stop=bool(stage.early_stop),
        recon_L=None,
        loss=active_loss_spec,
        gauge_fix=pose_gauge_fix,
    )
    geometry_for_align, align_kwargs = _pose_stage_geometry_context(
        geometry=geometry,
        grid=grid,
        detector=detector,
        setup_alignment_state=state.setup_alignment_state,
        active_geometry_dofs=active_geometry_dofs,
        level_factor=level_factor,
    )
    align_resume_state, stage_resume_consumed = _pose_stage_resume_state(
        stage=stage,
        resume_state=resume_state,
        resuming_this_level=resuming_this_level,
        level_resume=level_resume,
        global_elapsed_offset=global_elapsed_offset,
        stage_resume_consumed=state.stage_resume_consumed,
    )
    x_lvl, params5, info = align(
        geometry_for_align,
        grid,
        detector,
        projections,
        cfg=cfg_stage,
        init_x=state.x_lvl,
        init_params5=state.params5,
        observer=stage_runtime.observer_for_stage(stage, stage_global_start),
        resume_state=align_resume_state,
        checkpoint_callback=stage_runtime.checkpoint_for_stage(
            stage=stage,
            global_start=stage_global_start,
            setup_alignment_state=state.setup_alignment_state,
            active_geometry_dofs=active_geometry_dofs,
        ),
        **align_kwargs,
    )
    setup_alignment_state = state.setup_alignment_state.replace(
        pose=PoseState(
            params5,
            info.get("motion_coeffs"),  # type: ignore[arg-type]
        ),
        volume=x_lvl,
    )
    level_stats.extend(
        stage_runtime.enrich_stats(
            [dict(stat) for stat in info.get("outer_stats", [])],
            stage=stage,
            global_start=stage_global_start,
        )
    )
    level_losses.extend(float(value) for value in info.get("loss", []))
    level_wall_time = _accumulate_stage_wall_time(state.level_wall_time, info)
    level_action = _normalize_observer_action(info.get("observer_action"))
    final_gauge_fix = str(info.get("gauge_fix", state.final_gauge_fix))
    final_gauge_fix_dofs = list(info.get("gauge_fix_dofs", state.final_gauge_fix_dofs))
    final_gauge_fix_stats = dict(info.get("gauge_fix_final", {}) or {})
    return replace(
        state,
        x_lvl=x_lvl,
        params5=params5,
        info=info,
        setup_alignment_state=setup_alignment_state,
        level_wall_time=level_wall_time,
        level_action=level_action,
        final_gauge_fix=final_gauge_fix,
        final_gauge_fix_dofs=final_gauge_fix_dofs,
        final_gauge_fix_stats=final_gauge_fix_stats,
        stage_resume_consumed=stage_resume_consumed,
    )


def _run_multires_level_stages(
    *,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    cfg: AlignConfig,
    resolved_schedule: object,
    active_loss_spec: object,
    active_loss_name: str,
    setup_alignment_state: object,
    active_geometry_dofs: tuple[str, ...],
    level_factor: int,
    stage_runtime: StageRuntime,
    resume_state: AlignMultiresResumeState | None,
    resuming_this_level: bool,
    level_resume: LevelResumePlan,
    global_elapsed_offset: float,
    x_lvl: jnp.ndarray,
    params5: jnp.ndarray,
    level_stats: list[OuterStat],
    level_losses: list[float],
    final_gauge_fix: str,
    final_gauge_fix_dofs: list[str],
    final_gauge_fix_stats: dict[str, object],
) -> StageRunResult:
    info: dict[str, object] = {
        "loss": [],
        "loss_kind": active_loss_name,
        "recon_algo": str(cfg.recon_algo),
        "L": None,
        "outer_stats": [],
        "stopped_by_observer": False,
        "observer_action": "continue",
        "wall_time_total": 0.0,
        "align_profile": str(cfg.align_profile),
        "profile_policy": profile_policy_from_config(cfg).to_dict(),
        "quality_tier": str(cfg.quality_tier),
        "fallback_policy": str(cfg.fallback_policy),
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
    state = StageLoopState(
        x_lvl=x_lvl,
        params5=params5,
        info=info,
        setup_alignment_state=setup_alignment_state,
        level_wall_time=0.0,
        level_action="continue",
        final_gauge_fix=final_gauge_fix,
        final_gauge_fix_dofs=final_gauge_fix_dofs,
        final_gauge_fix_stats=final_gauge_fix_stats or {},
    )

    for stage in resolved_schedule.stages:
        if resuming_this_level:
            if int(stage.index) < level_resume.resume_stage_index:
                continue
            if (
                int(stage.index) == level_resume.resume_stage_index
                and level_resume.resume_stage_completed
            ):
                continue
        stage_global_start = level_resume.global_before_level + len(level_stats)
        if stage.stage_role == "proposal":
            state = _run_proposal_stage(
                geometry=geometry,
                grid=grid,
                detector=detector,
                projections=projections,
                cfg=cfg,
                stage=stage,
                active_loss_spec=active_loss_spec,
                setup_alignment_state=state.setup_alignment_state,
                level_factor=int(level_factor),
                stage_runtime=stage_runtime,
                stage_global_start=stage_global_start,
                level_stats=level_stats,
                level_losses=level_losses,
                state=state,
            )
            continue

        if stage.active_geometry_dofs:
            state = _run_setup_geometry_stage(
                geometry=geometry,
                grid=grid,
                detector=detector,
                projections=projections,
                cfg=cfg,
                stage=stage,
                resolved_schedule=resolved_schedule,
                active_loss_spec=active_loss_spec,
                active_loss_name=active_loss_name,
                level_factor=level_factor,
                stage_runtime=stage_runtime,
                stage_global_start=stage_global_start,
                level_stats=level_stats,
                level_losses=level_losses,
                state=state,
            )
            continue

        if not stage.active_pose_dofs:
            continue

        state = _run_pose_alignment_stage(
            geometry=geometry,
            grid=grid,
            detector=detector,
            projections=projections,
            cfg=cfg,
            stage=stage,
            active_loss_spec=active_loss_spec,
            active_geometry_dofs=active_geometry_dofs,
            level_factor=level_factor,
            stage_runtime=stage_runtime,
            stage_global_start=stage_global_start,
            resume_state=resume_state,
            resuming_this_level=resuming_this_level,
            level_resume=level_resume,
            global_elapsed_offset=global_elapsed_offset,
            level_stats=level_stats,
            level_losses=level_losses,
            state=state,
        )
        if state.level_action != "continue":
            break

    return StageRunResult(
        x_lvl=state.x_lvl,
        params5=state.params5,
        info=state.info,
        setup_alignment_state=state.setup_alignment_state,
        level_stats=level_stats,
        level_losses=level_losses,
        level_wall_time=state.level_wall_time,
        level_action=state.level_action,
        final_gauge_fix=state.final_gauge_fix,
        final_gauge_fix_dofs=state.final_gauge_fix_dofs,
        final_gauge_fix_stats=state.final_gauge_fix_stats,
    )
