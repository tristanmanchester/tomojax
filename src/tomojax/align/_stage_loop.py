from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
import logging
import time
from typing import TYPE_CHECKING, TypedDict

import jax
import jax.numpy as jnp

from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.multires import (
    bin_projections,
    scale_detector,
    scale_grid,
    upsample_volume,
    validate_scale_factor,
)
from tomojax.core.projector import forward_project_view_T
from tomojax.core.validation import (
    validate_grid,
    validate_projection_stack,
)
from tomojax.motion import phase_corr_shift
from tomojax.recon.fista_tv import FistaConfig, fista_tv

from ._config import (
    AlignConfig,
    _resolved_schedule_for_cfg,
)
from ._observer import (
    _normalize_observer_action,
    adapt_observer_callback,
)
from ._pose_stage import align
from ._profiles import profile_policy_from_config
from ._results import (
    AlignMultiresCheckpointCallback,
    AlignMultiresInfo,
    AlignMultiresResumeState,
    AlignResumeState,
    _record_stat_conversion_error,
    enrich_multires_stage_stat as _enrich_multires_stage_stat,
)
from ._setup_stage import (
    _geometry_calibration_payload,
    _geometry_with_setup_state,
    _optimize_setup_geometry_bilevel_for_level,
)
from .geometry.geometry_applier import (
    BaseGeometryArrays,
    apply_alignment_state,
    apply_setup_to_detector_grid,
)
from .geometry.geometry_blocks import (
    add_geometry_acquisition_diagnostics,
    summarize_geometry_calibration_stats,
)
from .model.dofs import DOF_INDEX
from .model.gauge import (
    active_gauge_dofs,
    normalize_gauge_fix,
)
from .model.state import PoseState, alignment_state_from_checkpoint
from .objectives.loss_adapters import build_loss_adapter
from .objectives.loss_specs import (
    loss_spec_name,
    resolve_loss_for_level,
    validate_loss_schedule_levels,
)
from .proposals import ProposalCandidate, score_pose_stack_candidates

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Geometry, Grid

    from ._observer import ObserverAction, ObserverCallback, OuterStat
    from .model.schedules import ResolvedAlignmentSchedule, ResolvedAlignmentStage

_SUPPORTED_POSE_STAGE_OPTIMIZERS = frozenset({"gd", "gn", "lbfgs"})

_PROPOSAL_STEP_BY_DOF = {
    "alpha": 0.01,
    "beta": 0.01,
    "phi": 0.02,
    "dx": 1.0,
    "dz": 1.0,
}


class MultiresLevel(TypedDict):
    factor: int
    grid: Grid
    detector: Detector
    projections: jnp.ndarray


@dataclass(frozen=True)
class LevelResumePlan:
    resuming: bool
    level_completed_before: int
    stats_before_level: list[OuterStat]
    loss_before_level: list[float]
    global_before_level: int
    resume_stage_index: int
    resume_stage_completed: bool
    resume_stage_iters: int
    preserved_level_stats: list[OuterStat]
    resume_stage_stats: list[OuterStat]
    preserved_level_losses: list[float]
    resume_stage_losses: list[float]


@dataclass(frozen=True)
class StageRuntime:
    level_index: int
    level_factor: int
    global_before_level: int
    global_elapsed_offset: float
    active_loss_name: str
    schedule_name: str
    stats_before_level: list[OuterStat]
    loss_before_level: list[float]
    level_stats: list[OuterStat]
    level_losses: list[float]
    prev_factor: int | None
    observer_fn: ObserverCallback | None
    checkpoint_callback: AlignMultiresCheckpointCallback | None

    def enrich_stats(
        self,
        local_stats: list[OuterStat],
        *,
        stage: ResolvedAlignmentStage | None = None,
        global_start: int | None = None,
    ) -> list[OuterStat]:
        enriched_stats: list[OuterStat] = []
        start = self.global_before_level if global_start is None else int(global_start)
        for idx, stat in enumerate(local_stats, start=1):
            enriched_stats.append(
                _enrich_multires_stage_stat(
                    stat,
                    level_factor=self.level_factor,
                    level_index=self.level_index,
                    global_outer_idx=int(start + idx),
                    elapsed_offset=float(self.global_elapsed_offset),
                    loss_name=self.active_loss_name,
                    schedule_name=self.schedule_name,
                    stage=stage,
                )
            )
        return enriched_stats

    def observer_for_stage(
        self,
        stage: ResolvedAlignmentStage,
        global_start: int,
    ) -> ObserverCallback | None:
        if self.observer_fn is None:
            return None

        def stage_observer(
            x_obs: jnp.ndarray,
            params_obs: jnp.ndarray,
            stat_obs: OuterStat,
        ) -> ObserverAction | None:
            enriched = _enrich_multires_stage_stat(
                stat_obs,
                level_factor=self.level_factor,
                level_index=self.level_index,
                global_outer_idx=int(global_start + int(stat_obs["outer_idx"])),
                elapsed_offset=float(self.global_elapsed_offset),
                loss_name=self.active_loss_name,
                schedule_name=self.schedule_name,
                stage=stage,
            )
            return self.observer_fn(x_obs, params_obs, enriched)

        return stage_observer

    def checkpoint_for_stage(
        self,
        *,
        stage: ResolvedAlignmentStage,
        global_start: int,
        setup_alignment_state: object,
        active_geometry_dofs: tuple[str, ...],
    ) -> AlignMultiresCheckpointCallback | None:
        if self.checkpoint_callback is None:
            return None

        def emit_multires_checkpoint(state: AlignResumeState) -> None:
            enriched_stats = self.enrich_stats(
                [dict(stat) for stat in state.outer_stats],
                stage=stage,
                global_start=global_start,
            )
            self.checkpoint_callback(
                _build_multires_checkpoint_state(
                    x=state.x,
                    params5=state.params5,
                    motion_coeffs=state.motion_coeffs,
                    level_index=self.level_index,
                    level_factor=self.level_factor,
                    completed_outer_iters_in_level=(len(self.level_stats) + state.start_outer_iter),
                    global_outer_iters_completed=(
                        self.global_before_level + len(self.level_stats) + state.start_outer_iter
                    ),
                    prev_factor=self.prev_factor,
                    loss=self.loss_before_level + self.level_losses + list(state.loss),
                    outer_stats=self.stats_before_level + self.level_stats + enriched_stats,
                    L=state.L,
                    small_impr_streak=int(state.small_impr_streak),
                    elapsed_offset=float(self.global_elapsed_offset + state.elapsed_offset),
                    level_complete=False,
                    run_complete=False,
                    setup_alignment_state=setup_alignment_state,
                    active_geometry_dofs=active_geometry_dofs,
                    stage=stage,
                    stage_completed=int(state.start_outer_iter) >= int(stage.maxiter),
                    completed_outer_iters_in_stage=int(state.start_outer_iter),
                )
            )

        return emit_multires_checkpoint


@dataclass(frozen=True)
class StageRunResult:
    x_lvl: jnp.ndarray
    params5: jnp.ndarray
    info: dict[str, object]
    setup_alignment_state: object
    level_stats: list[OuterStat]
    level_losses: list[float]
    level_wall_time: float
    level_action: ObserverAction
    final_gauge_fix: str
    final_gauge_fix_dofs: list[str]
    final_gauge_fix_stats: dict[str, object]


@dataclass(frozen=True)
class StageLoopState:
    x_lvl: jnp.ndarray
    params5: jnp.ndarray
    info: dict[str, object]
    setup_alignment_state: object
    level_wall_time: float
    level_action: ObserverAction
    final_gauge_fix: str
    final_gauge_fix_dofs: list[str]
    final_gauge_fix_stats: dict[str, object]
    stage_resume_consumed: bool = False


@dataclass(frozen=True)
class MultiresContext:
    cfg: AlignConfig
    observer_fn: ObserverCallback | None
    resolved_schedule: ResolvedAlignmentSchedule
    active_mask_tuple: tuple[bool, ...]
    setup_alignment_state: object
    active_geometry_dofs: tuple[str, ...]
    factors_list: list[int]
    levels: list[MultiresLevel]


@dataclass(frozen=True)
class MultiresRunState:
    x_init: jnp.ndarray | None
    params5: jnp.ndarray | None
    prev_factor: int | None
    loss_hist: list[float]
    global_outer_stats: list[OuterStat]
    stopped_by_observer: bool
    final_observer_action: ObserverAction
    global_outer_idx: int
    global_elapsed_offset: float
    executed_outer_iters: int
    final_pose_model_variables: int | None
    final_per_view_variables: int | None
    final_pose_model_basis_shape: list[int] | None
    final_loss_kind: str | None
    final_gauge_fix: str
    final_gauge_fix_dofs: list[str]
    final_gauge_fix_stats: dict[str, object] | None
    last_level_index_processed: int | None
    setup_alignment_state: object


def _stage_index_or_none(stat: Mapping[str, object]) -> int | None:
    try:
        return int(stat["schedule_stage_index"])
    except Exception:
        return None


def _accumulate_stage_wall_time(
    level_wall_time: float,
    info: dict[str, object],
) -> float:
    raw_wall_time = info.get("wall_time_total")
    try:
        return level_wall_time + float(raw_wall_time or 0.0)
    except (TypeError, ValueError, OverflowError) as exc:
        _record_stat_conversion_error(
            info,  # type: ignore[arg-type]
            "wall_time_total",
            exc,
            errors_key="level_stats_errors",
        )
        return level_wall_time


def _pose_stage_optimizer_or_raise(stage: ResolvedAlignmentStage) -> str:
    optimizer = str(stage.optimizer_kind)
    if optimizer in _SUPPORTED_POSE_STAGE_OPTIMIZERS:
        return optimizer
    supported = ", ".join(f"{name!r}" for name in sorted(_SUPPORTED_POSE_STAGE_OPTIMIZERS))
    raise ValueError(
        f"Unsupported pose-stage optimizer {optimizer!r} for alignment stage {stage.name!r}; "
        f"pose stages support {supported}. Add a pose-stage implementation before selecting "
        "this optimizer."
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
        geometry_dofs=(),
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
    recon_iters = 0 if stage.objective_kind == "fixed_volume" else int(cfg.recon_iters)
    cfg_stage = replace(
        cfg,
        schedule=None,
        optimise_dofs=stage.active_pose_dofs,
        geometry_dofs=(),
        opt_method=str(pose_optimizer),
        outer_iters=int(stage.maxiter),
        recon_iters=recon_iters,
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


def _build_multires_checkpoint_state(
    *,
    x: jnp.ndarray,
    params5: jnp.ndarray,
    motion_coeffs: object,
    level_index: int,
    level_factor: int,
    completed_outer_iters_in_level: int,
    global_outer_iters_completed: int,
    prev_factor: int | None,
    loss: list[float],
    outer_stats: list[OuterStat],
    L: object,
    small_impr_streak: int,
    elapsed_offset: float,
    level_complete: bool,
    run_complete: bool,
    setup_alignment_state: object,
    active_geometry_dofs: tuple[str, ...],
    stage: ResolvedAlignmentStage,
    stage_completed: bool,
    completed_outer_iters_in_stage: int,
) -> AlignMultiresResumeState:
    return AlignMultiresResumeState(
        x=x,
        params5=params5,
        motion_coeffs=motion_coeffs,
        level_index=int(level_index),
        level_factor=int(level_factor),
        completed_outer_iters_in_level=int(completed_outer_iters_in_level),
        global_outer_iters_completed=int(global_outer_iters_completed),
        prev_factor=prev_factor,
        loss=list(loss),
        outer_stats=[dict(stat) for stat in outer_stats],
        L=L,
        small_impr_streak=int(small_impr_streak),
        elapsed_offset=float(elapsed_offset),
        level_complete=bool(level_complete),
        run_complete=bool(run_complete),
        geometry_calibration_state=_geometry_calibration_payload(
            setup_alignment_state,
            active_geometry_dofs,
        ),
        stage_index=int(stage.index),
        stage_name=stage.name,
        stage_completed=bool(stage_completed),
        completed_outer_iters_in_stage=int(completed_outer_iters_in_stage),
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
    return state


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
