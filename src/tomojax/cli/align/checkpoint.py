from __future__ import annotations

# ruff: noqa: D100,D103,TC001,TC002
from collections.abc import Mapping
from dataclasses import dataclass
import logging
from typing import cast

import jax.numpy as jnp

from tomojax._typed_arrays import jax_float32_array, object_list, object_mapping
from tomojax.align.api import (
    AlignConfig,
    AlignmentCheckpointGeometrySnapshot,
    AlignmentCheckpointMetadataInput,
    AlignmentCheckpointProgress,
    AlignmentProjectionIdentity,
    AlignMultiresResumeState,
    AlignResumeState,
    CheckpointError,
    CheckpointMetadata,
    ScheduleResumeState,
    build_alignment_checkpoint_metadata_from_input,
    load_alignment_checkpoint,
    normalize_schedule_resume_state,
    save_alignment_checkpoint,
    validate_alignment_checkpoint,
)
from tomojax.geometry import Detector, Grid
from tomojax.io import JsonValue, normalize_json

from .command import AlignCommand
from .types import AlignCliCheckpointCallbacks, AlignCliRunPlan


@dataclass(frozen=True, slots=True)
class AlignCliCheckpointMetadataContext:
    """Stable CLI-owned inputs shared by checkpoint metadata builders."""

    meta: object
    projections: jnp.ndarray
    cfg: AlignConfig
    command: AlignCommand
    recon_grid: Grid
    detector: Detector
    gather_dtype: str
    schedule_metadata: dict[str, object] | None = None


def metadata_int(value: object, default: int = 0) -> int:
    if isinstance(value, int | float | str):
        return int(value)
    return default


def metadata_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    return default


def metadata_list(value: object) -> list[object]:
    return object_list(value)


def metadata_json_list(value: object) -> list[JsonValue]:
    normalized = normalize_json(value)
    if isinstance(normalized, list):
        return normalized
    return []


def metadata_json_mapping(value: object) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if isinstance(normalized, dict):
        return normalized
    return {}


def _checkpoint_cli_options(command: AlignCommand, *, gather_dtype: str) -> dict[str, object]:
    return {
        "align_profile": command.align_profile,
        "mode": command.mode,
        "roi": command.roi,
        "grid": command.grid,
        "requested_gather_dtype": command.requested_gather_dtype,
        "gather_dtype": str(gather_dtype),
        "recon_algo": command.recon_algo,
        "views_per_batch": command.views_per_batch,
        "spdhg_seed": command.spdhg_seed,
        "recon_positivity": command.recon_positivity,
        "projector_unroll": command.projector_unroll,
        "projector_backend": command.projector_backend,
        "quality_tier": command.quality_tier,
        "fallback_policy": command.fallback_policy,
        "checkpoint_projector": command.checkpoint_projector,
        "mask_vol": command.mask_vol,
        "gauge_fix": command.gauge_fix,
        "gauge_policy": command.gauge_policy,
        "optimise_dofs": command.optimise_dofs,
        "freeze_dofs": command.freeze_dofs,
        "schedule": command.schedule,
    }


def checkpoint_metadata_context_from_plan(
    plan: AlignCliRunPlan,
) -> AlignCliCheckpointMetadataContext:
    return AlignCliCheckpointMetadataContext(
        meta=plan.meta,
        projections=plan.projections,
        cfg=plan.cfg,
        command=plan.command,
        recon_grid=plan.recon_grid,
        detector=plan.detector,
        gather_dtype=plan.gather_dtype,
        schedule_metadata=plan.schedule_metadata,
    )


def initial_checkpoint_metadata(
    *,
    context: AlignCliCheckpointMetadataContext,
    levels: list[int] | None,
) -> CheckpointMetadata:
    return checkpoint_metadata_from_context(
        context,
        AlignmentCheckpointProgress(
            levels=levels,
            level_index=0,
            level_factor=1,
            completed_outer_iters_in_level=0,
            global_outer_iters_completed=0,
            prev_factor=None,
            L_prev=None,
            small_impr_streak=0,
            elapsed_offset=0.0,
            level_complete=False,
            run_complete=False,
        ),
    )


def checkpoint_metadata_from_context(
    context: AlignCliCheckpointMetadataContext,
    progress: AlignmentCheckpointProgress,
    *,
    state_grid: Grid | None = None,
    state_detector: Detector | None = None,
    schedule_state: ScheduleResumeState | None = None,
    geometry_calibration_state: dict[str, object] | None = None,
) -> CheckpointMetadata:
    state_grid = context.recon_grid if state_grid is None else state_grid
    state_detector = context.detector if state_detector is None else state_detector
    geometry_meta = getattr(getattr(context.meta, "metadata", context.meta), "geometry_meta", None)
    geometry_type = getattr(context.meta, "geometry_type", "parallel")
    return build_alignment_checkpoint_metadata_from_input(
        AlignmentCheckpointMetadataInput(
            projection=AlignmentProjectionIdentity(
                shape=tuple(int(v) for v in context.projections.shape),
                dtype=str(context.projections.dtype),
            ),
            geometry=AlignmentCheckpointGeometrySnapshot(
                geometry_type=str(geometry_type),
                geometry_meta=geometry_meta,
                reconstruction_grid=context.recon_grid.to_dict(),
                detector=context.detector.to_dict(),
                state_grid=state_grid.to_dict(),
                state_detector=state_detector.to_dict(),
                geometry_calibration_state=geometry_calibration_state,
            ),
            progress=progress,
            config=context.cfg,
            cli_options=_checkpoint_cli_options(context.command, gather_dtype=context.gather_dtype),
            random_state={
                "alignment": None,
                "seed_translations": (
                    "deterministic_phase_correlation" if context.cfg.seed_translations else None
                ),
            },
            schedule_metadata=context.schedule_metadata,
            schedule_state=schedule_state,
        )
    )


def checkpoint_progress(
    *,
    levels: list[int] | None,
    level_index: int,
    level_factor: int,
    completed_outer_iters_in_level: int,
    global_outer_iters_completed: int,
    prev_factor: int | None,
    L_prev: float | None,
    small_impr_streak: int,
    elapsed_offset: float,
    level_complete: bool,
    run_complete: bool,
) -> AlignmentCheckpointProgress:
    return AlignmentCheckpointProgress(
        levels=levels,
        level_index=int(level_index),
        level_factor=int(level_factor),
        completed_outer_iters_in_level=int(completed_outer_iters_in_level),
        global_outer_iters_completed=int(global_outer_iters_completed),
        prev_factor=prev_factor,
        current_inner_iteration=0,
        L_prev=L_prev,
        small_impr_streak=small_impr_streak,
        elapsed_offset=elapsed_offset,
        level_complete=level_complete,
        run_complete=run_complete,
    )


def _default_schedule_resume_state() -> ScheduleResumeState:
    return {
        "stage_index": 0,
        "stage_name": None,
        "stage_completed": False,
        "completed_outer_iters_in_stage": 0,
    }


def _schedule_resume_state_from_checkpoint(metadata: CheckpointMetadata) -> ScheduleResumeState:
    raw_schedule_state = metadata.get("schedule_state")
    if raw_schedule_state is not None and not isinstance(raw_schedule_state, Mapping):
        raise CheckpointError("corrupt checkpoint: invalid schedule_state")
    try:
        return (
            normalize_schedule_resume_state(raw_schedule_state) or _default_schedule_resume_state()
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointError("corrupt checkpoint: invalid schedule_state") from exc


def resume_state_from_checkpoint(
    checkpoint_path: str,
    *,
    expected_metadata: CheckpointMetadata,
    used_multires: bool,
) -> AlignResumeState | AlignMultiresResumeState:
    checkpoint = load_alignment_checkpoint(checkpoint_path)
    validate_alignment_checkpoint(checkpoint, expected_metadata)
    metadata = checkpoint.metadata
    if used_multires:
        schedule_state = _schedule_resume_state_from_checkpoint(metadata)
        prev_factor_value = metadata.get("prev_factor")
        geometry_calibration_state = metadata.get("geometry_calibration_state")
        return AlignMultiresResumeState(
            x=jax_float32_array(checkpoint.x),
            params5=jax_float32_array(checkpoint.params5),
            motion_coeffs=(
                None
                if checkpoint.motion_coeffs is None
                else jax_float32_array(checkpoint.motion_coeffs)
            ),
            level_index=int(metadata.get("level_index", 0)),
            level_factor=int(metadata.get("level_factor", 1)),
            completed_outer_iters_in_level=int(metadata.get("completed_outer_iters_in_level", 0)),
            global_outer_iters_completed=int(metadata.get("global_outer_iters_completed", 0)),
            prev_factor=None if prev_factor_value is None else int(prev_factor_value),
            loss=list(checkpoint.loss_history),
            outer_stats=[dict(stat) for stat in checkpoint.outer_stats],
            L=metadata.get("L_prev"),
            small_impr_streak=int(metadata.get("small_impr_streak", 0)),
            elapsed_offset=float(metadata.get("elapsed_offset", 0.0)),
            level_complete=bool(metadata.get("level_complete", False)),
            run_complete=bool(metadata.get("run_complete", False)),
            geometry_calibration_state=(
                object_mapping(cast("object", geometry_calibration_state))
                if isinstance(geometry_calibration_state, dict)
                else None
            ),
            stage_index=schedule_state["stage_index"],
            stage_name=schedule_state["stage_name"],
            stage_completed=schedule_state["stage_completed"],
            completed_outer_iters_in_stage=schedule_state["completed_outer_iters_in_stage"],
        )
    return AlignResumeState(
        x=jax_float32_array(checkpoint.x),
        params5=jax_float32_array(checkpoint.params5),
        motion_coeffs=(
            None
            if checkpoint.motion_coeffs is None
            else jax_float32_array(checkpoint.motion_coeffs)
        ),
        start_outer_iter=int(metadata.get("completed_outer_iters_in_level", 0)),
        loss=list(checkpoint.loss_history),
        outer_stats=[dict(stat) for stat in checkpoint.outer_stats],
        L=metadata.get("L_prev"),
        small_impr_streak=int(metadata.get("small_impr_streak", 0)),
        elapsed_offset=float(metadata.get("elapsed_offset", 0.0)),
    )


def _state_grid_detector_for_checkpoint(
    plan: AlignCliRunPlan,
    level_factor: int,
    *,
    run_complete: bool,
) -> tuple[Grid, Detector]:
    if plan.run_levels is None or run_complete:
        return plan.recon_grid, plan.detector
    from tomojax.core.multires import scale_detector, scale_grid

    return scale_grid(plan.recon_grid, int(level_factor)), scale_detector(
        plan.detector,
        int(level_factor),
    )


def make_align_cli_checkpoint_callbacks(plan: AlignCliRunPlan) -> AlignCliCheckpointCallbacks:
    metadata_context = checkpoint_metadata_context_from_plan(plan)

    def write_single_checkpoint(
        state: AlignResumeState,
        *,
        run_complete: bool = False,
    ) -> None:
        if plan.checkpoint_path is None:
            return
        completed = int(state.start_outer_iter)
        every = int(plan.checkpoint_every or 1)
        if not run_complete and (completed <= 0 or completed % every != 0):
            return
        metadata = checkpoint_metadata_from_context(
            metadata_context,
            checkpoint_progress(
                levels=None,
                level_index=0,
                level_factor=1,
                completed_outer_iters_in_level=completed,
                global_outer_iters_completed=completed,
                prev_factor=None,
                L_prev=state.L,
                small_impr_streak=int(state.small_impr_streak),
                elapsed_offset=float(state.elapsed_offset),
                level_complete=run_complete or completed >= int(plan.cfg.outer_iters),
                run_complete=run_complete,
            ),
        )
        save_alignment_checkpoint(
            plan.checkpoint_path,
            x=state.x,
            params5=state.params5,
            motion_coeffs=state.motion_coeffs,
            loss_history=state.loss,
            outer_stats=state.outer_stats,
            metadata=metadata,
        )
        logging.info("Saved alignment checkpoint to %s", plan.checkpoint_path)

    def write_multires_checkpoint(state: AlignMultiresResumeState) -> None:
        if plan.checkpoint_path is None:
            return
        completed = int(state.global_outer_iters_completed)
        every = int(plan.checkpoint_every or 1)
        if (
            not state.run_complete
            and not state.level_complete
            and (completed <= 0 or completed % every != 0)
        ):
            return
        state_grid, state_detector = _state_grid_detector_for_checkpoint(
            plan,
            int(state.level_factor),
            run_complete=bool(state.run_complete),
        )
        metadata = checkpoint_metadata_from_context(
            metadata_context,
            checkpoint_progress(
                levels=plan.run_levels,
                level_index=int(state.level_index),
                level_factor=int(state.level_factor),
                completed_outer_iters_in_level=int(state.completed_outer_iters_in_level),
                global_outer_iters_completed=completed,
                prev_factor=state.prev_factor,
                L_prev=state.L,
                small_impr_streak=int(state.small_impr_streak),
                elapsed_offset=float(state.elapsed_offset),
                level_complete=bool(state.level_complete),
                run_complete=bool(state.run_complete),
            ),
            state_grid=state_grid,
            state_detector=state_detector,
            schedule_state={
                "stage_index": int(state.stage_index),
                "stage_name": state.stage_name,
                "stage_completed": bool(state.stage_completed),
                "completed_outer_iters_in_stage": int(state.completed_outer_iters_in_stage),
            },
            geometry_calibration_state=state.geometry_calibration_state,
        )
        save_alignment_checkpoint(
            plan.checkpoint_path,
            x=state.x,
            params5=state.params5,
            motion_coeffs=state.motion_coeffs,
            loss_history=state.loss,
            outer_stats=state.outer_stats,
            metadata=metadata,
        )
        logging.info("Saved alignment checkpoint to %s", plan.checkpoint_path)

    return AlignCliCheckpointCallbacks(
        single=write_single_checkpoint, multires=write_multires_checkpoint
    )
