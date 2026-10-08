"""``tomojax align``'s checkpoints: the run a plan is, and when to write one.

The record and the resume state are :mod:`tomojax.alignment`'s
(:class:`~tomojax.alignment.api.AlignmentRun`), shared with ``tomojax.align``.
"""

from __future__ import annotations

# ruff: noqa: D103,TC001,TC002
from typing import cast

import jax.numpy as jnp

from tomojax._typed_arrays import object_list
from tomojax.alignment.api import (
    AlignConfig,
    AlignmentRun,
    AlignMultiresResumeState,
    AlignResumeState,
    write_alignment_checkpoint,
)
from tomojax.geometry import Geometry, ScanGeometry
from tomojax.io.api import JsonValue, normalize_json

from .command import AlignCommand, public_mode
from .types import AlignCliCheckpointCallbacks, AlignCliRunPlan


def metadata_int(value: object, default: int = 0) -> int:
    if isinstance(value, int | float | str):
        return int(value)
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


def checkpoint_run(
    *,
    projections: jnp.ndarray,
    geometry: Geometry,
    command: AlignCommand,
    cfg: AlignConfig,
    levels: list[int] | None,
    schedule_metadata: dict[str, object] | None,
) -> AlignmentRun:
    """The run a checkpoint of this command must match to resume it."""
    return AlignmentRun(
        projections=projections,
        geometry=cast("ScanGeometry", geometry),  # every built-in geometry is one
        config=cfg,
        mode=public_mode(command.mode),
        levels=levels,
        schedule_metadata=schedule_metadata,
    )


def make_align_cli_checkpoint_callbacks(plan: AlignCliRunPlan) -> AlignCliCheckpointCallbacks:
    """Callbacks writing ``plan``'s checkpoint every ``--checkpoint-every`` outer iterations."""
    path, run = plan.checkpoint_path, plan.checkpoint_run
    every = int(plan.checkpoint_every or 1)

    def due(completed: int) -> bool:
        return completed > 0 and completed % every == 0

    def write_single_checkpoint(state: AlignResumeState, *, run_complete: bool = False) -> None:
        if path is not None and (run_complete or due(int(state.start_outer_iter))):
            write_alignment_checkpoint(path, run, state, run_complete=run_complete)

    def write_multires_checkpoint(state: AlignMultiresResumeState) -> None:
        finished = state.run_complete or state.level_complete
        if path is not None and (finished or due(int(state.global_outer_iterations_completed))):
            write_alignment_checkpoint(path, run, state)

    return AlignCliCheckpointCallbacks(
        single=write_single_checkpoint, multires=write_multires_checkpoint
    )
