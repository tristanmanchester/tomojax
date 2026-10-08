"""Resuming an alignment from its checkpoint.

An :class:`AlignmentRun` is what a checkpoint must match to resume a run: the
projections and geometry (by a fingerprint, and the grid and detector by
value), the mode, the levels and the configuration.
:func:`write_alignment_checkpoint` saves a solver state with that record, and
:func:`resume_state_from_checkpoint` reads one back, refusing a checkpoint of
another run. :func:`alignment_checkpointing` joins the two for one path, as
``tomojax.align(scan, checkpoint=...)`` uses them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from functools import cached_property
import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from tomojax._typed_arrays import jax_float32_array, object_mapping
from tomojax._version import __version__
from tomojax.alignment._geometry.parametrizations import pad_pose_params
from tomojax.alignment._results import AlignMultiresResumeState, AlignResumeState
from tomojax.alignment.io.checkpoint import (
    CHECKPOINT_KIND,
    MULTIRES_GEOMETRY_VERSION,
    SCHEMA_VERSION,
    CheckpointError,
    CheckpointMetadata,
    ScheduleResumeState,
    load_alignment_checkpoint,
    normalize_json,
    normalize_schedule_resume_state,
    save_alignment_checkpoint,
    validate_alignment_checkpoint,
)

if TYPE_CHECKING:
    import os

    from tomojax.alignment._config import AlignConfig
    from tomojax.core.geometry.base import Detector, Grid, ScanGeometry

LOG = logging.getLogger(__name__)
# Views hashed into a run's fingerprint: enough to tell two scans apart, few
# enough to read quickly from a large stack.
_FINGERPRINT_VIEWS = 16

type ResumeState = AlignResumeState | AlignMultiresResumeState
type CheckpointCallback = Callable[[AlignMultiresResumeState], None]


@dataclass(frozen=True, kw_only=True)
class AlignmentRun:
    """What a checkpoint must match to resume an alignment.

    ``geometry`` is the one the run starts from, before any calibration, on
    the grid it reconstructs. ``levels`` are a multiresolution run's
    coarse-to-fine factors, or None for a single-resolution run.
    ``schedule_metadata``, when given, must match as well.
    """

    projections: Any
    geometry: ScanGeometry
    config: AlignConfig
    mode: str
    levels: Sequence[int] | None
    schedule_metadata: Mapping[str, Any] | None = None

    @property
    def grid(self) -> Grid:
        """The reconstruction grid, the geometry's."""
        return self.geometry.grid

    @property
    def detector(self) -> Detector:
        """The detector, the geometry's."""
        return self.geometry.detector

    @cached_property
    def fingerprint(self) -> str:
        """A hash of sixteen evenly spaced views and of every setting of the geometry.

        The geometry's settings include its view angles and any per-view pose
        corrections.
        """
        shape = tuple(int(n) for n in self.projections.shape)
        views = np.unique(np.linspace(0, shape[0] - 1, min(shape[0], _FINGERPRINT_VIEWS)).round())
        digest = hashlib.blake2b(repr(shape).encode(), digest_size=16)
        sample = np.asarray(self.projections[views.astype(np.int64)], np.float32)
        digest.update(np.ascontiguousarray(sample).tobytes())
        _hash_geometry(self.geometry, digest)
        return digest.hexdigest()


def _hash_geometry(value: object, digest: hashlib.blake2b) -> None:
    """Add ``value``, a geometry or one of its settings, to ``digest``.

    Geometries, beams, grids and detectors are dataclasses: each field that
    takes part in equality is added in turn, numbers as float64 arrays.
    """
    if is_dataclass(value) and not isinstance(value, type):
        digest.update(type(value).__name__.encode())
        for item in fields(value):
            if item.compare:
                _hash_geometry(getattr(value, item.name), digest)
    elif isinstance(value, tuple | list) and any(is_dataclass(v) for v in value):
        for item in cast("Sequence[object]", value):
            _hash_geometry(item, digest)
    elif value is None or isinstance(value, str):
        digest.update(repr(value).encode())
    else:
        array = np.asarray(value, np.float64)
        digest.update(repr(array.shape).encode() + np.ascontiguousarray(array).tobytes())


def alignment_checkpoint_metadata(
    run: AlignmentRun,
    state: ResumeState | None = None,
    *,
    run_complete: bool = False,
) -> CheckpointMetadata:
    """The checkpoint metadata of ``run`` at ``state`` (at the start for None).

    ``run_complete`` marks a single-resolution state as the finished run; a
    multiresolution state records that itself.
    """
    metadata: dict[str, Any] = {
        "checkpoint_kind": CHECKPOINT_KIND,
        "schema_version": SCHEMA_VERSION,
        "tomojax_version": __version__,
        "projection_shape": [int(v) for v in run.projections.shape],
        "projection_dtype": str(run.projections.dtype),
        "fingerprint": run.fingerprint,
        "reconstruction_grid": normalize_json(run.grid.to_dict()),
        "detector": normalize_json(run.detector.to_dict()),
        "mode": str(run.mode),
        "levels": None if run.levels is None else [int(v) for v in run.levels],
        "config": normalize_json(run.config),
        "schedule_metadata": normalize_json(run.schedule_metadata),
        **_progress(run, state, run_complete=run_complete),
    }
    if run.levels is not None and any(int(f) > 1 for f in run.levels):
        # Version 1 padded odd detector edges and changed coarse volume bounds.
        # Reusing those states under the corrected pyramid would move the data.
        metadata["multires_geometry_version"] = MULTIRES_GEOMETRY_VERSION
    # Keep persisted metadata strict JSON.
    _ = json.dumps(metadata, allow_nan=False, sort_keys=True)
    return cast("CheckpointMetadata", metadata)


def _progress(
    run: AlignmentRun, state: ResumeState | None, *, run_complete: bool
) -> dict[str, Any]:
    """The progress fields of ``state``'s checkpoint metadata."""
    if isinstance(state, AlignMultiresResumeState):
        return _multires_progress(run, state)
    completed = 0 if state is None else int(state.start_outer_iter)
    finished = state is not None and completed >= int(run.config.outer_iterations)
    return {
        "state_grid": normalize_json(run.grid.to_dict()),
        "state_detector": normalize_json(run.detector.to_dict()),
        "level_index": 0,
        "level_factor": 1,
        "completed_outer_iterations_in_level": completed,
        "global_outer_iterations_completed": completed,
        "current_inner_iteration": 0,
        "prev_factor": None,
        "lipschitz": None if state is None or state.lipschitz is None else float(state.lipschitz),
        "small_impr_streak": 0 if state is None else int(state.small_impr_streak),
        "elapsed_offset": 0.0 if state is None else float(state.elapsed_offset),
        "schedule_state": None,
        "geometry_calibration_state": None,
        "level_complete": bool(run_complete or finished),
        "run_complete": bool(run_complete),
    }


def _multires_progress(run: AlignmentRun, state: AlignMultiresResumeState) -> dict[str, Any]:
    """:func:`_progress` of a multiresolution state, on its level's grid until the run ends."""
    from tomojax.core.multires import scale_detector, scale_grid

    factor = int(state.level_factor)
    grid, detector = run.grid, run.detector
    if run.levels is not None and not state.run_complete:
        grid, detector = scale_grid(grid, factor), scale_detector(detector, factor)
    schedule_state: ScheduleResumeState = {
        "stage_index": int(state.stage_index),
        "stage_name": state.stage_name,
        "stage_completed": bool(state.stage_completed),
        "completed_outer_iterations_in_stage": int(state.completed_outer_iterations_in_stage),
    }
    return {
        "state_grid": normalize_json(grid.to_dict()),
        "state_detector": normalize_json(detector.to_dict()),
        "level_index": int(state.level_index),
        "level_factor": factor,
        "completed_outer_iterations_in_level": int(state.completed_outer_iterations_in_level),
        "global_outer_iterations_completed": int(state.global_outer_iterations_completed),
        "current_inner_iteration": 0,
        "prev_factor": None if state.prev_factor is None else int(state.prev_factor),
        "lipschitz": None if state.lipschitz is None else float(state.lipschitz),
        "small_impr_streak": int(state.small_impr_streak),
        "elapsed_offset": float(state.elapsed_offset),
        "schedule_state": schedule_state,
        "geometry_calibration_state": normalize_json(state.geometry_calibration_state),
        "level_complete": bool(state.level_complete),
        "run_complete": bool(state.run_complete),
    }


def write_alignment_checkpoint(
    path: str | os.PathLike[str],
    run: AlignmentRun,
    state: ResumeState,
    *,
    run_complete: bool = False,
) -> None:
    """Save ``state`` of ``run`` to ``path``, replacing the file there atomically."""
    save_alignment_checkpoint(
        path,
        x=state.x,
        pose_params=state.pose_params,
        motion_coeffs=state.motion_coeffs,
        loss_history=state.loss,
        outer_stats=state.outer_stats,
        metadata=alignment_checkpoint_metadata(run, state, run_complete=run_complete),
    )
    LOG.info("Saved alignment checkpoint to %s", path)


def resume_state_from_checkpoint(path: str | os.PathLike[str], run: AlignmentRun) -> ResumeState:
    """The state ``run`` resumes from, read from the checkpoint at ``path``.

    A multiresolution state for a run with levels, a single-resolution one
    otherwise. Raises :class:`CheckpointError` when the file is no checkpoint
    of ``run``.
    """
    checkpoint = load_alignment_checkpoint(path)
    validate_alignment_checkpoint(checkpoint, alignment_checkpoint_metadata(run))
    metadata = checkpoint.metadata
    saved_config = cast("object", metadata.get("config", {}))
    if not isinstance(saved_config, Mapping):
        raise CheckpointError("corrupt checkpoint: config must be a mapping")
    config = cast("Mapping[str, object]", saved_config)
    shared: dict[str, Any] = {
        "pose_translation_frame": str(config.get("pose_translation_frame", "object")),
        "ray_integrator": str(config.get("ray_integrator", "sampled")),
        "x": jax_float32_array(checkpoint.x),
        "pose_params": jax_float32_array(pad_pose_params(checkpoint.pose_params)),
        "motion_coeffs": (
            None
            if checkpoint.motion_coeffs is None
            else jax_float32_array(checkpoint.motion_coeffs)
        ),
        "loss": list(checkpoint.loss_history),
        "outer_stats": [dict(stat) for stat in checkpoint.outer_stats],
        "lipschitz": metadata.get("lipschitz"),
        "small_impr_streak": int(metadata.get("small_impr_streak", 0)),
        "elapsed_offset": float(metadata.get("elapsed_offset", 0.0)),
    }
    if run.levels is None:
        start = int(metadata.get("completed_outer_iterations_in_level", 0))
        return AlignResumeState(start_outer_iter=start, **shared)
    schedule = _schedule_resume_state_from_checkpoint(metadata)
    prev_factor = metadata.get("prev_factor")
    calibration = metadata.get("geometry_calibration_state")
    return AlignMultiresResumeState(
        **shared,
        level_index=int(metadata.get("level_index", 0)),
        level_factor=int(metadata.get("level_factor", 1)),
        completed_outer_iterations_in_level=int(
            metadata.get("completed_outer_iterations_in_level", 0)
        ),
        global_outer_iterations_completed=int(metadata.get("global_outer_iterations_completed", 0)),
        prev_factor=None if prev_factor is None else int(prev_factor),
        level_complete=bool(metadata.get("level_complete", False)),
        run_complete=bool(metadata.get("run_complete", False)),
        geometry_calibration_state=(
            object_mapping(cast("object", calibration)) if isinstance(calibration, dict) else None
        ),
        stage_index=schedule["stage_index"],
        stage_name=schedule["stage_name"],
        stage_completed=schedule["stage_completed"],
        completed_outer_iterations_in_stage=schedule["completed_outer_iterations_in_stage"],
    )


def _schedule_resume_state_from_checkpoint(metadata: CheckpointMetadata) -> ScheduleResumeState:
    raw = metadata.get("schedule_state")
    if raw is not None and not isinstance(raw, Mapping):
        raise CheckpointError("corrupt checkpoint: invalid schedule_state")
    try:
        state = normalize_schedule_resume_state(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointError("corrupt checkpoint: invalid schedule_state") from exc
    return state or {
        "stage_index": 0,
        "stage_name": None,
        "stage_completed": False,
        "completed_outer_iterations_in_stage": 0,
    }


def alignment_checkpointing(
    path: str | os.PathLike[str] | None,
    run: AlignmentRun,
) -> tuple[AlignMultiresResumeState | None, CheckpointCallback | None]:
    """Resume a multiresolution ``run`` from ``path``, and checkpoint it there.

    Returns the state to resume from, None when there is no file at ``path``,
    and the ``align_multires`` callback writing a checkpoint there after each
    outer iteration; both are None without a ``path``. A file there that is
    no checkpoint of ``run`` raises :class:`CheckpointError` saying how it
    differs, and is left as it is.
    """
    if path is None:
        return None, None
    if run.levels is None:
        raise ValueError("alignment_checkpointing takes a multiresolution run (with levels)")
    file = Path(path)
    resume = None
    if file.exists():
        try:
            resume = cast("AlignMultiresResumeState", resume_state_from_checkpoint(file, run))
        except CheckpointError as error:
            raise CheckpointError(
                f"{file} cannot resume this alignment ({error}); remove it or choose "
                "another checkpoint path"
            ) from error
        LOG.info("Resuming alignment from checkpoint %s", file)

    def write(state: AlignMultiresResumeState) -> None:
        write_alignment_checkpoint(file, run, state)

    return resume, write


__all__ = [
    "AlignmentRun",
    "alignment_checkpoint_metadata",
    "alignment_checkpointing",
    "resume_state_from_checkpoint",
    "write_alignment_checkpoint",
]
