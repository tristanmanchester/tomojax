from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

import jax.numpy as jnp

from tomojax.core.geometry.base import Detector, Grid

from ._config import AlignConfig
from ._model.schedules import ResolvedAlignmentSchedule, ResolvedAlignmentStage
from ._observer import ObserverAction, ObserverCallback, OuterStat
from ._results import (
    AlignMultiresCheckpointCallback,
    AlignMultiresResumeState,
    AlignResumeState,
    _record_stat_conversion_error,
    enrich_multires_stage_stat as _enrich_multires_stage_stat,
)
from ._setup_stage import _geometry_calibration_payload

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
