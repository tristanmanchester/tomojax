"""Alignment checkpoint files: their metadata, persistence and validation.

What a checkpoint must match to resume a run, and the state it resumes, are
in :mod:`tomojax.alignment.io.resume`.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from functools import cached_property
import json
import os
from pathlib import Path
from threading import RLock
from typing import Any, Required, TypedDict, cast
from uuid import uuid4

import numpy as np

from tomojax.alignment._geometry.parametrizations import pad_pose_params
from tomojax.alignment._model.dofs import POSE_WIDTH
from tomojax.io.api import normalize_json as _normalize_json

CHECKPOINT_KIND = "tomojax.alignment.checkpoint"
# 2: pose tables are stored as "pose_params".
# 3: settings and progress use the solver vocabulary (iterations, tv_weight,
#    lipschitz, outer_iterations, ...).
# 4: a run is identified by its mode, configuration, levels, grid and a
#    fingerprint of its projections and geometry; the command line's
#    "cli_options" and the geometry type and metadata are gone.
SCHEMA_VERSION = 4
MULTIRES_GEOMETRY_VERSION = 2
# Metadata a checkpoint must share with the run it resumes.
# The fingerprint covers the grid and detector too, so it comes last: a
# difference in them is named as such.
IDENTITY_KEYS = (
    "tomojax_version",
    "projection_shape",
    "projection_dtype",
    "reconstruction_grid",
    "detector",
    "mode",
    "levels",
    "multires_geometry_version",
    "config",
    "fingerprint",
)

_IDENTITY_NAMES = {"fingerprint": "fingerprint of the projections and geometry"}


class CheckpointError(ValueError):
    """Raised when an alignment checkpoint cannot be loaded or resumed."""


class CheckpointMetadata(TypedDict, total=False):
    """Persisted JSON metadata stored inside an alignment checkpoint.

    Attributes
    ----------
    checkpoint_kind : Required[str]
        File-kind identifier for an alignment checkpoint.
    schema_version : Required[int]
        Checkpoint layout schema version.
    tomojax_version : str | None
        TomoJAX version that wrote the checkpoint.
    projection_shape : Required[list[int]]
        Input shape (views, nv, nu).
    projection_dtype : Required[str]
        Input projection dtype name.
    fingerprint : Required[str]
        Run fingerprint of sampled projections and geometry settings.
    reconstruction_grid : Required[dict[str, Any]]
        Full-resolution reconstruction grid metadata.
    detector : Required[dict[str, Any]]
        Full-resolution detector metadata.
    state_grid : Required[dict[str, Any]]
        Grid on which the saved volume currently lives.
    state_detector : Required[dict[str, Any]]
        Detector at the current resolution level.
    mode : Required[str]
        Requested alignment mode.
    levels : list[int] | None
        Requested coarse-to-fine factors, or None for a single-resolution run.
    multires_geometry_version : int
        Version of the coarsened-geometry convention.
    level_index : Required[int]
        Zero-based current resolution-level index.
    level_factor : Required[int]
        Current resolution coarsening factor.
    completed_outer_iterations_in_level : Required[int]
        Outer iterations completed in the current level.
    global_outer_iterations_completed : Required[int]
        Outer iterations completed across all levels.
    current_inner_iteration : int
        Current reconstruction-iteration index.
    prev_factor : int | None
        Previously completed resolution factor, or None.
    lipschitz : float | None
        Most recent data-gradient Lipschitz estimate, or None.
    small_impr_streak : int
        Successive outer iterations with small improvement.
    elapsed_offset : float
        Elapsed seconds preceding the saved state.
    config : Required[Any]
        Serialized solver configuration.
    schedule_metadata : Any
        Schedule identity metadata.
    schedule_state : ScheduleResumeState | None
        Current stage and completed work for resuming the schedule.
    geometry_calibration_state : Any
        Final persisted setup-geometry state, when present.
    level_complete : bool
        Whether the current resolution level has finished.
    run_complete : bool
        Whether the entire run has finished.
    factors_skipped_reason : str | None
        Reason finer levels were skipped, or None.
    has_motion_coeffs : bool
        Whether the archive contains motion-model coefficients.
    """

    checkpoint_kind: Required[str]
    schema_version: Required[int]
    tomojax_version: str | None
    projection_shape: Required[list[int]]
    projection_dtype: Required[str]
    fingerprint: Required[str]
    reconstruction_grid: Required[dict[str, Any]]
    detector: Required[dict[str, Any]]
    state_grid: Required[dict[str, Any]]
    state_detector: Required[dict[str, Any]]
    mode: Required[str]
    levels: list[int] | None
    multires_geometry_version: int
    level_index: Required[int]
    level_factor: Required[int]
    completed_outer_iterations_in_level: Required[int]
    global_outer_iterations_completed: Required[int]
    current_inner_iteration: int
    prev_factor: int | None
    lipschitz: float | None
    small_impr_streak: int
    elapsed_offset: float
    config: Required[Any]
    schedule_metadata: Any
    schedule_state: ScheduleResumeState | None
    geometry_calibration_state: Any
    level_complete: bool
    run_complete: bool
    factors_skipped_reason: str | None
    has_motion_coeffs: bool


class ScheduleResumeState(TypedDict):
    """Stable multires schedule fields persisted for checkpoint resume."""

    stage_index: int
    stage_name: str | None
    stage_completed: bool
    completed_outer_iterations_in_stage: int


class AlignmentCheckpoint:
    """Alignment checkpoint arrays and metadata.

    A loaded checkpoint reads arrays on first access and keeps them in memory.
    It owns the open file until its arrays are read or the checkpoint is released.
    Replacing that file's path does not change this checkpoint.

    Parameters
    ----------
    x : numpy.ndarray
        Current reconstruction in (x, y, z) order on the current level grid.
    pose_params : numpy.ndarray
        Per-view corrections, columns (alpha, beta, phi, dx, dz, dy); angles are radians
        and translations physical lengths.
    motion_coeffs : numpy.ndarray | None
        Coefficients of a polynomial or spline motion model; None for independent per-view
         parameters.
    loss_history : list[float]
        Recorded objective values in iteration order.
    outer_stats : list[dict[str, Any]]
        Per-outer-iteration diagnostics in execution order.
    metadata : CheckpointMetadata
        Checkpoint identity, schema and progress metadata.
    """

    def __init__(
        self,
        x: np.ndarray,
        pose_params: np.ndarray,
        motion_coeffs: np.ndarray | None,
        loss_history: list[float],
        outer_stats: list[dict[str, Any]],
        metadata: CheckpointMetadata,
    ) -> None:
        self.__dict__.update(x=x, pose_params=pose_params, motion_coeffs=motion_coeffs)
        self.loss_history, self.outer_stats, self.metadata = loss_history, outer_stats, metadata
        self._archive: np.lib.npyio.NpzFile | None = None
        self._lock = RLock()

    def _read_array(self, name: str) -> np.ndarray | None:
        # cached_property alone permits duplicate concurrent reads on Python 3.12+.
        with self._lock:
            if name in self.__dict__:
                return self.__dict__[name]
            try:
                if name == "motion_coeffs" and not self.metadata.get("has_motion_coeffs", False):
                    value = None
                else:
                    assert self._archive is not None
                    raw = self._archive[name]
                    value = (
                        pad_pose_params(raw)
                        if name == "pose_params"
                        else np.asarray(raw, dtype=np.float32)
                    )
                self.__dict__[name] = value
                required = {"x", "pose_params"}
                if self.metadata.get("has_motion_coeffs", False):
                    required.add("motion_coeffs")
                if required <= self.__dict__.keys() and self._archive is not None:
                    self._archive.close()
                    self._archive = None
                return value
            except Exception as exc:
                raise CheckpointError(f"could not read checkpoint array {name}: {exc}") from exc

    @cached_property
    def x(self) -> np.ndarray:
        """The FP32 reconstruction on the checkpoint's grid.

        Returns
        -------
        numpy.ndarray
            Volume in (x, y, z) order.
        """
        return cast("np.ndarray", self._read_array("x"))

    @cached_property
    def pose_params(self) -> np.ndarray:
        """The FP32 six-column pose correction table.

        Returns
        -------
        numpy.ndarray
            Corrections in radians and physical length units.
        """
        return cast("np.ndarray", self._read_array("pose_params"))

    @cached_property
    def motion_coeffs(self) -> np.ndarray | None:
        """The motion-model coefficients.

        Returns
        -------
        numpy.ndarray or None
            Coefficients, or None for independent view poses.
        """
        return self._read_array("motion_coeffs")


class _LoadedCheckpoint(AlignmentCheckpoint):
    def __init__(
        self,
        archive: np.lib.npyio.NpzFile,
        metadata: CheckpointMetadata,
        loss_history: list[float],
        outer_stats: list[dict[str, Any]],
    ) -> None:
        self._archive, self._lock = archive, RLock()
        self.loss_history, self.outer_stats, self.metadata = loss_history, outer_stats, metadata


def normalize_json(value: Any) -> Any:
    """Convert common runtime objects into deterministic JSON-compatible values."""
    return _normalize_json(value, sort_mapping_keys=True, catch_to_dict_errors=True)


def _normalize_json_object(value: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = normalize_json(value or {})
    if not isinstance(normalized, Mapping):
        raise TypeError("checkpoint metadata object field normalized to a non-object value")
    return dict(normalized)


def normalize_schedule_resume_state(
    schedule_state: Mapping[str, Any] | ScheduleResumeState | None,
) -> ScheduleResumeState | None:
    """Normalize checkpoint schedule progress into the persisted resume schema."""
    if schedule_state is None:
        return None
    normalized = _normalize_json_object(schedule_state)
    stage_name = normalized.get("stage_name")
    if stage_name is not None and not isinstance(stage_name, str):
        raise TypeError("checkpoint schedule_state.stage_name must be a string or null")
    return {
        "stage_index": int(normalized["stage_index"]),
        "stage_name": stage_name,
        "stage_completed": bool(normalized["stage_completed"]),
        "completed_outer_iterations_in_stage": int(
            normalized["completed_outer_iterations_in_stage"]
        ),
    }


def save_alignment_checkpoint(
    path: str | os.PathLike[str],
    *,
    x: Any,
    pose_params: Any,
    motion_coeffs: Any | None = None,
    loss_history: list[float] | tuple[float, ...] = (),
    outer_stats: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    metadata: Mapping[str, Any],
) -> None:
    """Atomically write an alignment checkpoint as `.npz` plus JSON metadata.

    Parameters
    ----------
    path : str or path-like
        Checkpoint or sidecar file path.
    x : array-like
        Current reconstruction of shape (nx, ny, nz).
    pose_params : array-like
        Per-view corrections with columns (alpha, beta, phi, dx, dz[, dy]); angles are
        radians and translations physical lengths.
    motion_coeffs : array-like or None, optional
        Polynomial or spline motion coefficients; None for per-view motion.
    loss_history : sequence of float, optional
        Recorded losses in iteration order.
    outer_stats : sequence of dict, optional
        Per-iteration diagnostics in execution order.
    metadata : mapping
        Checkpoint identity, geometry and progress fields.

    Raises
    ------
    OSError
        If the destination cannot be created or replaced.
    """
    out_path = Path(path)
    if out_path.parent != Path():
        out_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_metadata = cast("CheckpointMetadata", normalize_json(dict(metadata)))
    normalized_metadata["checkpoint_kind"] = CHECKPOINT_KIND
    normalized_metadata["schema_version"] = SCHEMA_VERSION
    normalized_metadata["has_motion_coeffs"] = motion_coeffs is not None
    metadata_json = json.dumps(normalized_metadata, allow_nan=False, sort_keys=True)
    outer_stats_json = json.dumps(normalize_json(list(outer_stats)), allow_nan=False)

    tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with tmp_path.open("wb") as fh:
            arrays: dict[str, Any] = {
                "x": np.asarray(x, dtype=np.float32),
                "pose_params": np.asarray(pose_params, dtype=np.float32),
                "loss_history": np.asarray(loss_history, dtype=np.float64),
                "metadata_json": np.asarray(metadata_json),
                "outer_stats_json": np.asarray(outer_stats_json),
            }
            if motion_coeffs is not None:
                arrays["motion_coeffs"] = np.asarray(motion_coeffs, dtype=np.float32)
            else:
                arrays["motion_coeffs"] = np.zeros((0,), dtype=np.float32)
            np.savez_compressed(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        tmp_path.replace(out_path)
        _fsync_parent(out_path)
    except Exception:
        with suppress(OSError):
            tmp_path.unlink()
        raise


def _fsync_parent(path: Path) -> None:
    parent = path.parent if path.parent != Path() else Path()
    try:
        fd = os.open(parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def load_alignment_checkpoint(path: str | os.PathLike[str]) -> AlignmentCheckpoint:
    """Load an alignment checkpoint and convert malformed files to CheckpointError.

    Parameters
    ----------
    path : str or path-like
        Checkpoint or sidecar file path.

    Returns
    -------
    AlignmentCheckpoint
        Array fields loaded on first access, with loss history and metadata.

    Raises
    ------
    CheckpointError
        If the header cannot be read or is malformed. Malformed arrays raise
        CheckpointError when accessed.
    """
    z = None
    try:
        z = np.load(path, allow_pickle=False)
        files = set(z.files)
        required = {
            "x",
            "pose_params",
            "loss_history",
            "metadata_json",
            "outer_stats_json",
        }
        missing = sorted(required - files)
        if missing:
            raise CheckpointError(
                f"corrupt checkpoint: missing required field(s): {', '.join(missing)}"
            )

        metadata = _load_json_scalar(z["metadata_json"], field="metadata_json")
        outer_stats = _load_json_scalar(z["outer_stats_json"], field="outer_stats_json")
        if not isinstance(metadata, dict):
            raise CheckpointError("corrupt checkpoint: metadata_json must contain an object")
        if not isinstance(outer_stats, list):
            raise CheckpointError("corrupt checkpoint: outer_stats_json must contain a list")

        if metadata.get("has_motion_coeffs", False) and "motion_coeffs" not in files:
            raise CheckpointError(
                "corrupt checkpoint: metadata declares motion_coeffs but the array is missing"
            )
        return _LoadedCheckpoint(
            z,
            metadata=cast("CheckpointMetadata", dict(metadata)),
            loss_history=[float(v) for v in np.asarray(z["loss_history"]).reshape(-1)],
            outer_stats=[dict(item) for item in outer_stats],
        )
    except Exception as exc:
        close = getattr(z, "close", None)
        if close is not None:
            close()
        if isinstance(exc, CheckpointError):
            raise
        raise CheckpointError(f"could not read checkpoint {path}: {exc}") from exc


def _load_json_scalar(value: np.ndarray, *, field: str) -> Any:
    try:
        raw = value.item()
    except Exception as exc:
        raise CheckpointError(f"corrupt checkpoint: {field} must be a scalar JSON string") from exc
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raw = str(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"corrupt checkpoint: invalid {field}: {exc}") from exc


def _validate_kind_and_schema(metadata: Mapping[str, Any]) -> None:
    """Reject checkpoints of another kind, or from another schema version."""
    if metadata.get("checkpoint_kind") != CHECKPOINT_KIND:
        raise CheckpointError(
            f"corrupt checkpoint: unsupported checkpoint kind {metadata.get('checkpoint_kind')!r}"
        )
    version = metadata.get("schema_version")
    if isinstance(version, int) and 0 < version < SCHEMA_VERSION:
        raise CheckpointError(
            f"incompatible checkpoint: schema version {version} predates this version of "
            f"TomoJAX (schema {SCHEMA_VERSION}), which records alignments differently; "
            "restart the alignment"
        )
    if version != SCHEMA_VERSION:
        raise CheckpointError(f"corrupt checkpoint: unsupported schema version {version!r}")


def _check_identity(metadata: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Reject a checkpoint of another run: the first identity key that differs, and how."""
    for key in IDENTITY_KEYS:
        if key not in expected:
            continue
        actual_value = normalize_json(metadata.get(key))
        expected_value = normalize_json(expected.get(key))
        if actual_value != expected_value:
            name = _IDENTITY_NAMES.get(key, key.replace("_", " "))
            raise CheckpointError(
                f"incompatible checkpoint: {name} {_difference(actual_value, expected_value)}"
            )


def _difference(saved: Any, current: Any) -> str:
    """How ``saved`` differs from ``current``: by entry, for two mappings."""
    if not (isinstance(saved, Mapping) and isinstance(current, Mapping)):
        return f"{saved!r} does not match current {current!r}"
    saved_map = cast("Mapping[str, Any]", saved)
    current_map = cast("Mapping[str, Any]", current)
    keys = sorted(
        k for k in {*saved_map, *current_map} if saved_map.get(k, ...) != current_map.get(k, ...)
    )
    return "differs in " + ", ".join(
        f"{k} (checkpoint {saved_map.get(k)!r}, current {current_map.get(k)!r})" for k in keys
    )


def validate_alignment_checkpoint(
    checkpoint: AlignmentCheckpoint,
    expected_metadata: Mapping[str, Any] | CheckpointMetadata,
) -> None:
    """Validate that a checkpoint can resume the current alignment request.

    Parameters
    ----------
    checkpoint : AlignmentCheckpoint
        Loaded checkpoint to compare with the requested run.
    expected_metadata : mapping
        Required identity and layout fields of the current alignment request.

    Raises
    ------
    CheckpointError
        If identity or layout fields do not match the current request.
    """
    metadata = checkpoint.metadata
    _validate_kind_and_schema(metadata)

    if any(f > 1 for f in metadata.get("levels") or ()) and (
        metadata.get("multires_geometry_version") != MULTIRES_GEOMETRY_VERSION
    ):
        raise CheckpointError(
            "incompatible checkpoint: multires geometry version differs from the current "
            "sampling convention; restart this multiresolution run"
        )

    expected = normalize_json(dict(expected_metadata))
    _check_identity(metadata, expected)

    expected_schedule = expected.get("schedule_metadata")
    actual_schedule = metadata.get("schedule_metadata")
    if (
        actual_schedule is not None
        and expected_schedule is not None
        and normalize_json(actual_schedule) != normalize_json(expected_schedule)
    ):
        raise CheckpointError(
            "incompatible checkpoint: schedule metadata does not match current request"
        )

    state_grid = metadata.get("state_grid")
    if not isinstance(state_grid, Mapping):
        raise CheckpointError("corrupt checkpoint: metadata is missing state_grid")
    expected_x_shape = (
        int(state_grid["nx"]),
        int(state_grid["ny"]),
        int(state_grid["nz"]),
    )
    if tuple(checkpoint.x.shape) != expected_x_shape:
        raise CheckpointError(
            "corrupt checkpoint: x shape "
            f"{list(checkpoint.x.shape)} does not match state grid {list(expected_x_shape)}"
        )

    projection_shape = metadata.get("projection_shape")
    if not isinstance(projection_shape, list) or len(projection_shape) != 3:
        raise CheckpointError(
            "corrupt checkpoint: metadata projection_shape must be a length-3 list"
        )
    expected_params_shape = (int(projection_shape[0]), POSE_WIDTH)
    # Checkpoints written before dy existed hold five columns; loaders pad them.
    legacy_shape = (int(projection_shape[0]), 5)
    if tuple(checkpoint.pose_params.shape) not in {expected_params_shape, legacy_shape}:
        actual_shape = list(checkpoint.pose_params.shape)
        expected_shape = list(expected_params_shape)
        raise CheckpointError(
            "corrupt checkpoint: pose_params shape "
            f"{actual_shape} does not match expected {expected_shape}"
        )

    if checkpoint.motion_coeffs is not None and checkpoint.motion_coeffs.ndim != 2:
        raise CheckpointError("corrupt checkpoint: motion_coeffs must be a 2-D array")
