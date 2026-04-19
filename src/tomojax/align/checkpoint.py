from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np


CHECKPOINT_KIND = "tomojax.align.checkpoint"
SCHEMA_VERSION = 1
_ALIGN_CONFIG_COMPAT_DEFAULTS = {
    "recon_algo": "fista",
    "recon_positivity": True,
    "spdhg_seed": 0,
}
_ALIGN_CLI_COMPAT_DEFAULTS = {
    "recon_algo": "fista",
    "spdhg_seed": 0,
    "recon_positivity": True,
}


class CheckpointError(RuntimeError):
    """Raised when an alignment checkpoint cannot be loaded or resumed."""


@dataclass(slots=True)
class AlignmentCheckpoint:
    x: np.ndarray
    params5: np.ndarray
    motion_coeffs: np.ndarray | None
    loss_history: list[float]
    outer_stats: list[dict[str, Any]]
    metadata: dict[str, Any]


def _tomojax_version() -> str | None:
    try:
        from importlib import metadata

        return metadata.version("tomojax")
    except Exception:
        try:
            import tomojax

            return getattr(tomojax, "__version__", None)
        except Exception:
            return None


def normalize_json(value: Any) -> Any:
    """Convert common runtime objects into deterministic JSON-compatible values."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return normalize_json(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): normalize_json(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, tuple | list | set | frozenset):
        return [normalize_json(v) for v in value]
    if isinstance(value, np.generic):
        return normalize_json(value.item())
    if isinstance(value, np.ndarray):
        return normalize_json(value.tolist())

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return normalize_json(to_dict())
        except Exception:
            pass
    return str(value)


def _normalize_checkpoint_compare_value(key: str, value: Any) -> Any:
    normalized = normalize_json(value)
    if key == "config" and isinstance(normalized, Mapping):
        with_defaults = dict(normalized)
        for default_key, default_value in _ALIGN_CONFIG_COMPAT_DEFAULTS.items():
            with_defaults.setdefault(default_key, default_value)
        return with_defaults
    if key == "cli_options" and isinstance(normalized, Mapping):
        with_defaults = dict(normalized)
        for default_key, default_value in _ALIGN_CLI_COMPAT_DEFAULTS.items():
            with_defaults.setdefault(default_key, default_value)
        return with_defaults
    return normalized


def build_alignment_checkpoint_metadata(
    *,
    projections_shape: tuple[int, ...] | list[int],
    projections_dtype: str,
    geometry_type: str,
    geometry_meta: Mapping[str, Any] | None,
    reconstruction_grid: Mapping[str, Any],
    detector: Mapping[str, Any],
    state_grid: Mapping[str, Any],
    state_detector: Mapping[str, Any],
    levels: list[int] | None,
    level_index: int,
    level_factor: int,
    completed_outer_iters_in_level: int,
    global_outer_iters_completed: int,
    config: Any,
    cli_options: Mapping[str, Any] | None = None,
    prev_factor: int | None = None,
    current_inner_iteration: int = 0,
    L_prev: float | None = None,
    small_impr_streak: int = 0,
    elapsed_offset: float = 0.0,
    random_state: Mapping[str, Any] | None = None,
    level_complete: bool = False,
    run_complete: bool = False,
) -> dict[str, Any]:
    """Build normalized checkpoint metadata shared by CLI and tests."""
    metadata = {
        "checkpoint_kind": CHECKPOINT_KIND,
        "schema_version": SCHEMA_VERSION,
        "tomojax_version": _tomojax_version(),
        "projection_shape": [int(v) for v in projections_shape],
        "projection_dtype": str(projections_dtype),
        "geometry_type": str(geometry_type),
        "geometry_meta": normalize_json(geometry_meta or {}),
        "reconstruction_grid": normalize_json(reconstruction_grid),
        "detector": normalize_json(detector),
        "state_grid": normalize_json(state_grid),
        "state_detector": normalize_json(state_detector),
        "levels": None if levels is None else [int(v) for v in levels],
        "level_index": int(level_index),
        "level_factor": int(level_factor),
        "completed_outer_iters_in_level": int(completed_outer_iters_in_level),
        "global_outer_iters_completed": int(global_outer_iters_completed),
        "current_inner_iteration": int(current_inner_iteration),
        "prev_factor": None if prev_factor is None else int(prev_factor),
        "L_prev": None if L_prev is None else float(L_prev),
        "small_impr_streak": int(small_impr_streak),
        "elapsed_offset": float(elapsed_offset),
        "config": normalize_json(config),
        "cli_options": normalize_json(cli_options or {}),
        "random_state": normalize_json(random_state or {"alignment": None}),
        "level_complete": bool(level_complete),
        "run_complete": bool(run_complete),
    }
    # Keep persisted metadata strict JSON.
    json.dumps(metadata, allow_nan=False, sort_keys=True)
    return metadata


def save_alignment_checkpoint(
    path: str | os.PathLike[str],
    *,
    x: Any,
    params5: Any,
    motion_coeffs: Any | None = None,
    loss_history: list[float] | tuple[float, ...] = (),
    outer_stats: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    metadata: Mapping[str, Any],
) -> None:
    """Atomically write an alignment checkpoint as `.npz` plus JSON metadata."""
    out_path = Path(path)
    if out_path.parent != Path("."):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_metadata = normalize_json(dict(metadata))
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
                "params5": np.asarray(params5, dtype=np.float32),
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
        os.replace(tmp_path, out_path)
        _fsync_parent(out_path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def _fsync_parent(path: Path) -> None:
    parent = path.parent if path.parent != Path("") else Path(".")
    try:
        fd = os.open(parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def load_alignment_checkpoint(path: str | os.PathLike[str]) -> AlignmentCheckpoint:
    """Load an alignment checkpoint and convert malformed files to CheckpointError."""
    try:
        with np.load(path, allow_pickle=False) as z:
            files = set(z.files)
            required = {
                "x",
                "params5",
                "motion_coeffs",
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

            has_motion_coeffs = bool(metadata.get("has_motion_coeffs", False))
            motion_coeffs = np.asarray(z["motion_coeffs"], dtype=np.float32)
            return AlignmentCheckpoint(
                x=np.asarray(z["x"], dtype=np.float32),
                params5=np.asarray(z["params5"], dtype=np.float32),
                motion_coeffs=motion_coeffs if has_motion_coeffs else None,
                loss_history=[float(v) for v in np.asarray(z["loss_history"]).reshape(-1)],
                outer_stats=[dict(item) for item in outer_stats],
                metadata=dict(metadata),
            )
    except CheckpointError:
        raise
    except Exception as exc:
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


def validate_alignment_checkpoint(
    checkpoint: AlignmentCheckpoint,
    expected_metadata: Mapping[str, Any],
) -> None:
    """Validate that a checkpoint can resume the current alignment request."""
    metadata = checkpoint.metadata
    if metadata.get("checkpoint_kind") != CHECKPOINT_KIND:
        raise CheckpointError(
            "corrupt checkpoint: unsupported checkpoint kind "
            f"{metadata.get('checkpoint_kind')!r}"
        )
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise CheckpointError(
            "corrupt checkpoint: unsupported schema version "
            f"{metadata.get('schema_version')!r}"
        )

    expected = normalize_json(dict(expected_metadata))
    for key in (
        "tomojax_version",
        "projection_shape",
        "projection_dtype",
        "geometry_type",
        "geometry_meta",
        "reconstruction_grid",
        "detector",
        "levels",
        "config",
        "cli_options",
    ):
        if key not in expected:
            continue
        actual_value = _normalize_checkpoint_compare_value(key, metadata.get(key))
        expected_value = _normalize_checkpoint_compare_value(key, expected.get(key))
        if actual_value != expected_value:
            raise CheckpointError(
                f"incompatible checkpoint: {key.replace('_', ' ')} "
                f"{actual_value!r} does not match current {expected_value!r}"
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
        raise CheckpointError("corrupt checkpoint: metadata projection_shape must be a length-3 list")
    expected_params_shape = (int(projection_shape[0]), 5)
    if tuple(checkpoint.params5.shape) != expected_params_shape:
        raise CheckpointError(
            "corrupt checkpoint: params5 shape "
            f"{list(checkpoint.params5.shape)} does not match expected {list(expected_params_shape)}"
        )

    if checkpoint.motion_coeffs is not None and checkpoint.motion_coeffs.ndim != 2:
        raise CheckpointError("corrupt checkpoint: motion_coeffs must be a 2-D array")
