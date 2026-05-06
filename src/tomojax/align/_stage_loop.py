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
    adapt_legacy_observer,
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
    from .model.schedules import ResolvedAlignmentStage

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
    level_wall_time = 0.0
    level_action: ObserverAction = "continue"
    stage_resume_consumed = False

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
                x_lvl=x_lvl,
                params5=params5,
            )
            setup_alignment_state = setup_alignment_state.replace(
                pose=PoseState(params5),
                volume=x_lvl,
            )
            proposal_wall_time = time.perf_counter() - proposal_start
            level_wall_time += proposal_wall_time
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
            enriched = stage_runtime.enrich_stats(
                [stat],
                stage=stage,
                global_start=stage_global_start,
            )
            level_stats.extend(enriched)
            info["wall_time_total"] = float(level_wall_time)
            info["completed_outer_iters"] = len(level_stats)
            info["proposal"] = dict(proposal_info)
            continue

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
            setup_result = _optimize_setup_geometry_bilevel_for_level(
                geometry=geometry,
                grid=grid,
                detector=detector,
                projections=projections,
                init_x=x_lvl,
                init_params5=params5,
                state=setup_alignment_state,
                active_geometry_dofs=stage.active_geometry_dofs,
                factor=int(level_factor),
                cfg=cfg_stage,
                loss_spec=active_loss_spec,
                loss_name=active_loss_name,
                schedule_name=resolved_schedule.name,
                stage=stage,
            )
            x_lvl = setup_result.x
            setup_alignment_state = setup_result.state
            level_wall_time += time.perf_counter() - geometry_start
            enriched = stage_runtime.enrich_stats(
                [dict(stat) for stat in setup_result.checkpoint_outer_stats],
                stage=stage,
                global_start=stage_global_start,
            )
            level_stats.extend(enriched)
            level_losses.extend(setup_result.losses)
            info["wall_time_total"] = float(level_wall_time)
            info["completed_outer_iters"] = len(level_stats)
            continue

        if not stage.active_pose_dofs:
            continue

        pose_optimizer = _pose_stage_optimizer_or_raise(stage)
        pose_gauge_fix = (
            "mean_translation" if stage.gauge_policy == "anchor_mean" else cfg.gauge_fix
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
                grid,
                detector,
                setup_alignment_state.setup,
            )
            align_kwargs["det_grid_override"] = apply_setup_to_detector_grid(
                detector,
                setup_alignment_state.setup,
                level_factor=int(level_factor),
            )
        else:
            geometry_for_align = geometry

        align_resume_state = None
        if (
            resuming_this_level
            and resume_state is not None
            and int(stage.index) == level_resume.resume_stage_index
            and not level_resume.resume_stage_completed
            and not stage_resume_consumed
        ):
            align_resume_state = AlignResumeState(
                x=resume_state.x,
                params5=resume_state.params5,
                motion_coeffs=resume_state.motion_coeffs,
                start_outer_iter=level_resume.resume_stage_iters,
                loss=list(level_resume.resume_stage_losses),
                outer_stats=[dict(stat) for stat in level_resume.resume_stage_stats],
                L=resume_state.L,
                small_impr_streak=int(resume_state.small_impr_streak),
                elapsed_offset=float(resume_state.elapsed_offset - global_elapsed_offset),
            )
            stage_resume_consumed = True

        x_lvl, params5, info = align(
            geometry_for_align,
            grid,
            detector,
            projections,
            cfg=cfg_stage,
            init_x=x_lvl,
            init_params5=params5,
            observer=stage_runtime.observer_for_stage(stage, stage_global_start),
            resume_state=align_resume_state,
            checkpoint_callback=stage_runtime.checkpoint_for_stage(
                stage=stage,
                global_start=stage_global_start,
                setup_alignment_state=setup_alignment_state,
                active_geometry_dofs=active_geometry_dofs,
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
        enriched = stage_runtime.enrich_stats(
            [dict(stat) for stat in info.get("outer_stats", [])],
            stage=stage,
            global_start=stage_global_start,
        )
        level_stats.extend(enriched)
        level_losses.extend(float(value) for value in info.get("loss", []))
        level_wall_time = _accumulate_stage_wall_time(level_wall_time, info)
        level_action = _normalize_observer_action(info.get("observer_action"))
        final_gauge_fix = str(info.get("gauge_fix", final_gauge_fix))
        final_gauge_fix_dofs = list(info.get("gauge_fix_dofs", final_gauge_fix_dofs))
        final_gauge_fix_stats = dict(info.get("gauge_fix_final", {}) or {})
        if level_action != "continue":
            break

    return StageRunResult(
        x_lvl=x_lvl,
        params5=params5,
        info=info,
        setup_alignment_state=setup_alignment_state,
        level_stats=level_stats,
        level_losses=level_losses,
        level_wall_time=level_wall_time,
        level_action=level_action,
        final_gauge_fix=final_gauge_fix,
        final_gauge_fix_dofs=final_gauge_fix_dofs,
        final_gauge_fix_stats=final_gauge_fix_stats,
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

            def project_seed_view(
                transform: jnp.ndarray,
                *,
                seed_grid: Grid = g,
                seed_detector: Detector = d,
                seed_volume: jnp.ndarray = x_seed,
            ) -> jnp.ndarray:
                return forward_project_view_T(
                    transform,
                    seed_grid,
                    seed_detector,
                    seed_volume,
                    use_checkpoint=cfg.checkpoint_projector,
                    gather_dtype=cfg.gather_dtype,
                )

            vm_pred = jax.vmap(
                project_seed_view,
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

        level_run = _prepare_multires_level_state(
            resume_state=resume_state,
            level_index=int(li),
            loss_hist=loss_hist,
            global_outer_stats=global_outer_stats,
            executed_outer_iters=executed_outer_iters,
        )
        stats_before_level = level_run.stats_before_level
        loss_before_level = level_run.loss_before_level
        global_before_level = level_run.global_before_level
        level_stats: list[OuterStat] = [dict(stat) for stat in level_run.preserved_level_stats]
        level_losses: list[float] = [float(value) for value in level_run.preserved_level_losses]
        x_lvl = x0 if x0 is not None else jnp.zeros((g.nx, g.ny, g.nz), dtype=jnp.float32)
        params5 = params0 if params0 is not None else jnp.zeros((y.shape[0], 5), dtype=jnp.float32)
        stage_runtime = StageRuntime(
            level_index=int(li),
            level_factor=int(lvl["factor"]),
            global_before_level=global_before_level,
            global_elapsed_offset=global_elapsed_offset,
            active_loss_name=active_loss_name,
            schedule_name=resolved_schedule.name,
            stats_before_level=stats_before_level,
            loss_before_level=loss_before_level,
            level_stats=level_stats,
            level_losses=level_losses,
            prev_factor=prev_factor,
            observer_fn=observer_fn,
            checkpoint_callback=checkpoint_callback,
        )

        stage_result = _run_multires_level_stages(
            geometry=geometry,
            grid=g,
            detector=d,
            projections=y,
            cfg=cfg,
            resolved_schedule=resolved_schedule,
            active_loss_spec=active_loss_spec,
            active_loss_name=active_loss_name,
            setup_alignment_state=setup_alignment_state,
            active_geometry_dofs=active_geometry_dofs,
            level_factor=int(lvl["factor"]),
            stage_runtime=stage_runtime,
            resume_state=resume_state,
            resuming_this_level=resuming_this_level,
            level_resume=level_run,
            global_elapsed_offset=global_elapsed_offset,
            x_lvl=x_lvl,
            params5=params5,
            level_stats=level_stats,
            level_losses=level_losses,
            final_gauge_fix=final_gauge_fix,
            final_gauge_fix_dofs=final_gauge_fix_dofs,
            final_gauge_fix_stats=final_gauge_fix_stats or {},
        )
        x_lvl = stage_result.x_lvl
        params5 = stage_result.params5
        info = stage_result.info
        setup_alignment_state = stage_result.setup_alignment_state
        level_stats = stage_result.level_stats
        level_losses = stage_result.level_losses
        level_wall_time = stage_result.level_wall_time
        level_action = stage_result.level_action
        final_gauge_fix = stage_result.final_gauge_fix
        final_gauge_fix_dofs = stage_result.final_gauge_fix_dofs
        final_gauge_fix_stats = stage_result.final_gauge_fix_stats

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
        _emit_level_completion_checkpoint(
            checkpoint_callback=checkpoint_callback,
            x_lvl=x_lvl,
            params5=params5,
            info=info,
            level_index=int(li),
            level_factor=int(lvl["factor"]),
            level_completed_after=level_completed_after,
            global_outer_idx=int(global_outer_idx),
            prev_factor=prev_factor,
            loss_hist=loss_hist,
            global_outer_stats=global_outer_stats,
            global_elapsed_offset=global_elapsed_offset,
            level_complete=level_complete,
            setup_alignment_state=setup_alignment_state,
            active_geometry_dofs=active_geometry_dofs,
            resolved_schedule=resolved_schedule,
            level_stats=level_stats,
        )
        final_observer_action = level_action
        if level_action == "stop_run":
            stopped_by_observer = True
            break
        if level_action == "advance_level":
            continue

    x_final = _final_multires_volume(x_init=x_init, prev_factor=prev_factor, grid=grid)
    run_complete = _multires_run_is_complete(
        params5=params5,
        stopped_by_observer=stopped_by_observer,
        resume_state=resume_state,
        last_level_index_processed=last_level_index_processed,
        level_count=len(levels),
    )
    _emit_run_completion_checkpoint(
        checkpoint_callback=checkpoint_callback,
        params5=params5,
        run_complete=run_complete,
        x_final=x_final,
        level_count=len(levels),
        executed_outer_iters=executed_outer_iters,
        loss_hist=loss_hist,
        global_outer_stats=global_outer_stats,
        global_elapsed_offset=global_elapsed_offset,
        setup_alignment_state=setup_alignment_state,
        active_geometry_dofs=active_geometry_dofs,
        resolved_schedule=resolved_schedule,
    )

    return (
        x_final,
        params5 if params5 is not None else jnp.zeros((projections.shape[0], 5), jnp.float32),
        _final_align_multires_info(
            loss_hist=loss_hist,
            factors_list=factors_list,
            final_loss_kind=final_loss_kind,
            cfg=cfg,
            stopped_by_observer=stopped_by_observer,
            final_observer_action=final_observer_action,
            executed_outer_iters=executed_outer_iters,
            global_elapsed_offset=global_elapsed_offset,
            global_outer_stats=global_outer_stats,
            resolved_schedule=resolved_schedule,
            geometry=geometry,
            final_pose_model_variables=final_pose_model_variables,
            final_per_view_variables=final_per_view_variables,
            final_pose_model_basis_shape=final_pose_model_basis_shape,
            active_geometry_dofs=active_geometry_dofs,
            final_gauge_fix=final_gauge_fix,
            final_gauge_fix_dofs=final_gauge_fix_dofs,
            final_gauge_fix_stats=final_gauge_fix_stats,
            setup_alignment_state=setup_alignment_state,
        ),
    )
