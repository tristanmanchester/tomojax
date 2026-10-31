from __future__ import annotations

from dataclasses import replace
import time
from typing import cast

import jax.numpy as jnp

from tomojax.alignment._geometry.geometry_applier import (
    BaseGeometryArrays,
    apply_alignment_state,
    apply_setup_to_detector_grid,
    materialize_setup_geometry,
    pose_stack_for_setup,
    setup_moved_detector,
)
from tomojax.alignment._geometry.parametrizations import apply_pose_updates
from tomojax.alignment._model.dofs import DOF_INDEX
from tomojax.alignment._model.schedules import ResolvedAlignmentStage
from tomojax.alignment._model.state import AlignmentState, PoseState
from tomojax.alignment._objectives.loss_adapters import build_loss_adapter
from tomojax.alignment._observer import OuterStat, _normalize_observer_action
from tomojax.alignment._pose._pose_loop import align
from tomojax.alignment._prealign import seeded_translation_params
from tomojax.alignment._profiles import profile_policy_from_config
from tomojax.alignment.proposals import ProposalCandidate, score_pose_stack_candidates
from tomojax.core.cone import cone_model

from ._setup_stage import (
    _optimize_setup_geometry_bilevel_for_level,
    initialize_setup_geometry_for_level,
)
from ._spectral_motion_stage import run_spectral_motion_stage
from ._stage_types import (
    _PROPOSAL_STEP_BY_DOF,
    LevelContext,
    StageLoopState,
    StageRunResult,
    _accumulate_stage_wall_time,
)


def _proposal_candidates_for_pose_stage(
    *,
    stage: ResolvedAlignmentStage,
    base: BaseGeometryArrays,
    setup_alignment_state: object,
    pose_params: jnp.ndarray,
) -> tuple[ProposalCandidate, ...]:
    nominal = pose_stack_for_setup(base, setup_alignment_state.setup)
    frame = setup_alignment_state.pose.translation_frame
    baseline = apply_pose_updates(nominal, pose_params, translation_frame=frame)
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
            candidate_params = pose_params.at[:, col].add(jnp.float32(sign * step))
            candidates.append(
                ProposalCandidate(
                    f"{dof}_{label}",
                    apply_pose_updates(nominal, candidate_params, translation_frame=frame),
                    {
                        "kind": "uniform_pose_perturbation",
                        "dof": dof,
                        "delta": float(sign * step),
                    },
                )
            )
    return tuple(candidates)


def _apply_pose_proposal_stage(
    level: LevelContext, stage: ResolvedAlignmentStage, state: StageLoopState
) -> tuple[jnp.ndarray, dict[str, object]]:
    runtime = level.runtime
    setup_alignment_state, x_lvl, pose_params = (
        state.setup_alignment_state,
        state.x_lvl,
        state.pose_params,
    )
    if not stage.active_pose_dofs:
        return pose_params, {
            "proposal_status": "skipped",
            "proposal_reason": "proposal stage has no active pose DOFs",
        }
    if not bool(jnp.any(x_lvl)):
        return pose_params, {
            "proposal_status": "skipped",
            "proposal_reason": "zero volume has no pose evidence",
        }
    base = BaseGeometryArrays.from_geometry(
        level.geometry,
        level.detector,
        level_factor=int(runtime.level_factor),
    )
    effective = apply_alignment_state(
        base,
        setup_alignment_state.replace(
            pose=PoseState(pose_params, translation_frame=level.cfg.pose_translation_frame),
            volume=x_lvl,
        ),
    )
    adapter = build_loss_adapter(level.loss, level.projections)
    candidates = _proposal_candidates_for_pose_stage(
        stage=stage,
        base=base,
        setup_alignment_state=setup_alignment_state,
        pose_params=pose_params,
    )
    cone = cone_model(level.geometry, level.detector)
    result = score_pose_stack_candidates(
        candidates=candidates,
        grid=level.grid,
        detector=level.detector,
        volume=x_lvl,
        det_grid=effective.det_grid,
        targets=level.projections,
        loss_adapter=adapter,
        backend=level.cfg.backend,
        gather_dtype=level.cfg.gather_dtype,
        views_per_batch=level.cfg.views_per_batch,
        projector_unroll=1,
        checkpoint_projector=level.cfg.checkpoint_projector,
        ray_integrator=level.cfg.ray_integrator,
        frames=None if cone is None else jnp.asarray(cone.frames(level.projections.shape[0])),
    )
    best_params = pose_params
    if result.improved:
        best_meta = result.candidate_metadata[result.best_index]
        dof = best_meta.get("dof")
        delta = best_meta.get("delta")
        if isinstance(dof, str) and dof in DOF_INDEX and delta is not None:
            best_params = pose_params.at[:, DOF_INDEX[dof]].add(jnp.float32(float(delta)))
    return best_params, result.to_dict()


def _run_proposal_stage(
    level: LevelContext, stage: ResolvedAlignmentStage, state: StageLoopState
) -> StageLoopState:
    runtime = level.runtime
    stage_global_start = runtime.global_before_level + len(runtime.level_stats)
    setup_alignment_state = state.setup_alignment_state
    proposal_start = time.perf_counter()
    loss_before = float(runtime.level_losses[-1]) if runtime.level_losses else None
    pose_params, proposal_info = _apply_pose_proposal_stage(level, stage, state)
    setup_alignment_state = setup_alignment_state.replace(
        pose=PoseState(pose_params, translation_frame=level.cfg.pose_translation_frame),
        volume=state.x_lvl,
    )
    proposal_wall_time = time.perf_counter() - proposal_start
    level_wall_time = state.level_wall_time + proposal_wall_time
    proposal_loss = proposal_info.get("best_value")
    if proposal_loss is not None:
        runtime.level_losses.append(float(proposal_loss))
    stat: OuterStat = {
        "outer_idx": 1,
        "loss": float(proposal_loss) if proposal_loss is not None else None,
        "loss_before": loss_before,
        "loss_after": float(proposal_loss) if proposal_loss is not None else None,
        "proposal_status": str(proposal_info.get("proposal_status") or "scored"),
        "proposal_reason": str(proposal_info.get("proposal_reason") or ""),
        "proposal_best_name": str(proposal_info.get("best_name") or ""),
        "proposal_best_index": int(proposal_info.get("best_index") or 0),
        "proposal_improved": bool(proposal_info.get("improved") or False),
        "proposal_candidate_count": len(proposal_info.get("values") or []),
        "proposal_backend_provenance": dict(proposal_info.get("backend_provenance") or {}),
        "wall_time": float(proposal_wall_time),
    }
    runtime.level_stats.extend(
        runtime.enrich_stats([stat], stage=stage, global_start=stage_global_start)
    )
    info = dict(state.info)
    info["wall_time_total"] = float(level_wall_time)
    info["completed_outer_iterations"] = len(runtime.level_stats)
    info["proposal"] = dict(proposal_info)
    return replace(
        state,
        pose_params=pose_params,
        info=info,
        setup_alignment_state=setup_alignment_state,
        level_wall_time=level_wall_time,
    )


def _run_pose_alignment_stage(
    level: LevelContext, stage: ResolvedAlignmentStage, state: StageLoopState
) -> StageLoopState:
    runtime = level.runtime
    stage_global_start = runtime.global_before_level + len(runtime.level_stats)
    if stage.objective_kind == "projection_consistency":
        return run_spectral_motion_stage(
            state,
            stage,
            geometry=level.geometry,
            grid=level.grid,
            detector=level.detector,
            projections=level.projections,
            cfg=level.cfg,
            level_factor=runtime.level_factor,
            stage_runtime=runtime,
            stage_global_start=stage_global_start,
            level_stats=runtime.level_stats,
        )
    anchored = stage.gauge_policy == "anchor_mean" and level.cfg.pose_translation_frame == "object"
    cfg_stage = replace(
        level.cfg,
        schedule=None,
        optimise_dofs=stage.active_pose_dofs,
        outer_iterations=int(stage.maxiter),
        # A fixed-volume objective holds the volume fixed during each pose step;
        # alternation still refreshes it between steps. Skipping reconstruction
        # left pose-only schedules optimising against an all-zero volume.
        iterations=int(level.cfg.iterations),
        gn_coupling="joint" if stage.objective_kind == "joint_volume_pose" else "fixed_volume",
        early_stop=bool(stage.early_stop),
        # A supplied Lipschitz constant describes the full-resolution grid only.
        lipschitz=level.cfg.lipschitz if runtime.level_factor == 1 else None,
        loss=level.loss,
    )
    cfg_stage.stage_quality_tier = stage.quality_tier
    geometry_for_align, align_kwargs = level.geometry, {}
    if level.active_geometry_dofs:
        setup = state.setup_alignment_state.setup
        geometry_for_align = materialize_setup_geometry(
            level.geometry, level.grid, level.detector, setup
        )
        align_kwargs["det_grid_override"] = apply_setup_to_detector_grid(
            level.detector, setup, level_factor=int(runtime.level_factor)
        )
    align_resume_state = None
    stage_resume_consumed = state.stage_resume_consumed
    if (
        level.resume is not None
        and int(stage.index) == level.start_stage
        and not stage_resume_consumed
    ):
        align_resume_state = level.resume
        stage_resume_consumed = True
    x_lvl, pose_params, info = align(
        geometry_for_align,
        level.grid,
        level.detector,
        level.projections,
        cfg=cfg_stage,
        init_x=state.x_lvl,
        init_pose_params=state.pose_params,
        observer=runtime.observer_for_stage(stage, stage_global_start),
        resume_state=align_resume_state,
        checkpoint_callback=runtime.checkpoint_for_stage(
            stage=stage,
            global_start=stage_global_start,
            setup_alignment_state=state.setup_alignment_state,
            active_geometry_dofs=level.active_geometry_dofs,
        ),
        anchor_translations=anchored,
        **align_kwargs,
    )
    setup_alignment_state = state.setup_alignment_state.replace(
        pose=PoseState(
            pose_params,
            info.get("motion_coeffs"),  # type: ignore[arg-type]
            translation_frame=cfg_stage.pose_translation_frame,
        ),
        volume=x_lvl,
    )
    runtime.level_stats.extend(
        runtime.enrich_stats(
            [dict(stat) for stat in info.get("outer_stats", [])],
            stage=stage,
            global_start=stage_global_start,
        )
    )
    runtime.level_losses.extend(float(value) for value in info.get("loss", []))
    level_wall_time = _accumulate_stage_wall_time(state.level_wall_time, info)
    level_action = _normalize_observer_action(info.get("observer_action"))
    final_gauge_fix = str(info.get("gauge_fix", state.final_gauge_fix))
    final_gauge_fix_dofs = list(info.get("gauge_fix_dofs", state.final_gauge_fix_dofs))
    final_gauge_fix_stats = dict(info.get("gauge_fix_final", {}) or {})
    return replace(
        state,
        x_lvl=x_lvl,
        pose_params=pose_params,
        info=info,
        setup_alignment_state=setup_alignment_state,
        level_wall_time=level_wall_time,
        level_action=level_action,
        final_gauge_fix=final_gauge_fix,
        final_gauge_fix_dofs=final_gauge_fix_dofs,
        final_gauge_fix_stats=final_gauge_fix_stats,
        stage_resume_consumed=stage_resume_consumed,
    )


def _seed_stage_translations(
    level: LevelContext, stage: ResolvedAlignmentStage, state: StageLoopState
) -> StageLoopState:
    """Seed the first pose stage against the geometry calibrated before it."""
    runtime = level.runtime
    geometry, detector = level.geometry, level.detector
    if not level.cfg.seed_translations or stage.objective_kind == "projection_consistency":
        return state
    start = time.perf_counter()
    if level.active_geometry_dofs:
        setup = cast("AlignmentState", state.setup_alignment_state).setup
        moved = setup_moved_detector(detector, setup, level_factor=runtime.level_factor)
        if moved is None:
            return state  # The image-domain seed does not support a rolled ray grid.
        detector = moved
        geometry = materialize_setup_geometry(geometry, level.grid, detector, setup)
    seed = seeded_translation_params(geometry, level.grid, detector, level.projections, level.cfg)
    return replace(
        state,
        pose_params=state.pose_params if seed is None else seed,
        level_wall_time=state.level_wall_time + time.perf_counter() - start,
    )


def _run_multires_level_stages(
    level: LevelContext,
    *,
    setup_alignment_state: object,
    x_lvl: jnp.ndarray,
    pose_params: jnp.ndarray,
    seed_translations: bool,
    final_gauge_fix: str,
    final_gauge_fix_dofs: list[str],
    final_gauge_fix_stats: dict[str, object],
) -> StageRunResult:
    runtime = level.runtime
    info = {
        "loss": [],
        "loss_kind": runtime.active_loss_name,
        "reconstruction": str(level.cfg.reconstruction),
        "lipschitz": None,
        "outer_stats": [],
        "stopped_by_observer": False,
        "observer_action": "continue",
        "wall_time_total": 0.0,
        "quality": str(level.cfg.quality),
        "profile_policy": profile_policy_from_config(level.cfg).to_dict(),
        "pose_model": str(level.cfg.pose_model),
        "pose_translation_frame": level.cfg.pose_translation_frame,
        "ray_integrator": level.cfg.ray_integrator,
        "pose_model_variables": 0,
        "per_view_variables": 0,
        "pose_model_basis_shape": [],
        "active_dofs": [],
        "completed_outer_iterations": 0,
        "small_impr_streak": 0,
        "motion_coeffs": None,
        "gauge_fix": final_gauge_fix,
        "gauge_fix_dofs": final_gauge_fix_dofs,
        "gauge_fix_final": final_gauge_fix_stats or {},
    }
    state = StageLoopState(
        x_lvl=x_lvl,
        pose_params=pose_params,
        info=info,
        setup_alignment_state=setup_alignment_state,
        level_wall_time=0.0,
        level_action="continue",
        final_gauge_fix=final_gauge_fix,
        final_gauge_fix_dofs=final_gauge_fix_dofs,
        final_gauge_fix_stats=final_gauge_fix_stats or {},
    )

    # Start after the checkpointed stage only if that stage finished.
    for stage in level.schedule.stages:
        if int(stage.index) < level.start_stage:
            continue
        if seed_translations and stage.active_pose_dofs:
            state = _seed_stage_translations(level, stage, state)
            seed_translations = False
        stage_global_start = runtime.global_before_level + len(runtime.level_stats)
        if stage.stage_role == "proposal":
            state = _run_proposal_stage(level, stage, state)
            continue

        if stage.active_geometry_dofs:
            cfg_stage = replace(
                level.cfg,
                schedule=None,
                optimise_dofs=stage.active_geometry_dofs,
                outer_iterations=int(stage.maxiter),
                early_stop=bool(stage.early_stop),
            )
            cfg_stage.stage_quality_tier = stage.quality_tier
            geometry_start = time.perf_counter()
            if stage.objective_kind == "projection_consistency":
                setup_result = initialize_setup_geometry_for_level(
                    geometry=level.geometry,
                    detector=level.detector,
                    projections=level.projections,
                    state=cast("AlignmentState", state.setup_alignment_state).replace(
                        pose=PoseState(
                            state.pose_params, translation_frame=level.cfg.pose_translation_frame
                        ),
                        volume=state.x_lvl,
                    ),
                    active_geometry_dofs=stage.active_geometry_dofs,
                    factor=int(runtime.level_factor),
                    cfg=cfg_stage,
                )
            else:
                setup_result = _optimize_setup_geometry_bilevel_for_level(
                    geometry=level.geometry,
                    grid=level.grid,
                    detector=level.detector,
                    projections=level.projections,
                    init_x=state.x_lvl,
                    init_pose_params=state.pose_params,
                    state=state.setup_alignment_state,
                    active_geometry_dofs=stage.active_geometry_dofs,
                    factor=int(runtime.level_factor),
                    cfg=cfg_stage,
                    loss_spec=level.loss,
                    loss_name=runtime.active_loss_name,
                    schedule_name=level.schedule.name,
                    stage=stage,
                )
            level_wall_time = state.level_wall_time + (time.perf_counter() - geometry_start)
            enriched = runtime.enrich_stats(
                [dict(stat) for stat in setup_result.checkpoint_outer_stats],
                stage=stage,
                global_start=stage_global_start,
            )
            runtime.level_stats.extend(enriched)
            runtime.level_losses.extend(setup_result.losses)
            info = dict(state.info)
            info["wall_time_total"] = float(level_wall_time)
            info["completed_outer_iterations"] = len(runtime.level_stats)
            state = replace(
                state,
                x_lvl=setup_result.x,
                info=info,
                setup_alignment_state=setup_result.state,
                level_wall_time=level_wall_time,
            )
            continue

        if not stage.active_pose_dofs:
            continue

        state = _run_pose_alignment_stage(level, stage, state)
        if state.level_action != "continue":
            break

    return StageRunResult(
        x_lvl=state.x_lvl,
        pose_params=state.pose_params,
        info=state.info,
        setup_alignment_state=state.setup_alignment_state,
        level_stats=runtime.level_stats,
        level_losses=runtime.level_losses,
        level_wall_time=state.level_wall_time,
        level_action=state.level_action,
        final_gauge_fix=state.final_gauge_fix,
        final_gauge_fix_dofs=state.final_gauge_fix_dofs,
        final_gauge_fix_stats=state.final_gauge_fix_stats,
    )
