# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import numpy.typing as npt

from tomojax._data import save_nxtomo
from tomojax.io._json import JsonValue, normalize_json
from tomojax.io._preprocess_impl.config import (
    _ANGLE_PATH,
    _PROJECTION_PATHS,
    LOG,
    PREPROCESS_SCHEMA_VERSION,
    PreprocessConfig,
    PreprocessResult,
    _validate_config,
)
from tomojax.io._preprocess_impl.correction import (
    FloatArray,
    IntArray,
    _auto_reject_views,
    _coverage_changed,
    _coverage_stats,
    correct_nxtomo_frames,
    repair_nonfinite_preprocess_output,
)
from tomojax.io._preprocess_impl.selection import _parse_crop_spec, _resolve_sample_view_indices
from tomojax.io._preprocess_impl.source import (
    _metadata_from_raw,
    _optional_sample_metadata,
    _resolve_dataset,
    _resolve_image_key_dataset,
)
from tomojax.io._preprocess_impl.writer import _write_preprocess_provenance


def _metadata_int(metadata: Mapping[str, object], key: str) -> int:
    value = metadata[key]
    if not isinstance(value, int):
        raise TypeError(f"preprocess metadata field {key!r} must be an int")
    return value


def _metadata_str(metadata: Mapping[str, object], key: str) -> str:
    value = metadata[key]
    if not isinstance(value, str):
        raise TypeError(f"preprocess metadata field {key!r} must be a string")
    return value


def _array_dim(values: npt.NDArray[np.generic], axis: int) -> int:
    shape = cast("tuple[int, ...]", values.shape)
    return int(shape[axis])


def _array_int_at(values: IntArray, index: int) -> int:
    value = cast("object", values[index])
    if isinstance(value, np.integer):
        return int(cast("int", value))
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise TypeError(f"expected integer array scalar, got {type(value).__name__}")
    return int(value)


def _json_int_list(values: IntArray) -> list[JsonValue]:
    return [cast("JsonValue", _array_int_at(values, i)) for i in range(_array_dim(values, 0))]


def _entry_group(file: h5py.File) -> h5py.Group:
    entry = file["/entry"]
    if not isinstance(entry, h5py.Group):
        raise TypeError("NXtomo /entry must be an HDF5 group")
    return entry


@dataclass(frozen=True, slots=True)
class _RawPreprocessInput:
    frame_path: str
    key_path: str
    angle_path: str
    frames: FloatArray
    angles_all: npt.NDArray[np.float32]
    sample_raw_indices: IntArray
    flat_raw_indices: IntArray
    dark_raw_indices: IntArray
    raw_frame_counts: dict[str, int]
    crop_bounds: tuple[int, int, int, int] | None
    crop_slices: tuple[int, int, int, int]
    original_shape: tuple[int, int, int]
    cropped_shape: tuple[int, int, int]


def preprocess_nxtomo(
    input_path: str | Path,
    output_path: str | Path,
    config: PreprocessConfig | None = None,
) -> PreprocessResult:
    """Preprocess raw NXtomo sample/flat/dark frames into corrected projections."""
    cfg = config or PreprocessConfig()
    output_dtype, output_domain = _validate_config(cfg)

    input_path = Path(input_path)
    output_path = Path(output_path)

    with h5py.File(input_path, "r") as file:
        raw = _load_raw_preprocess_input(file, cfg)

        candidate_sample_indices, view_selection_meta = _resolve_sample_view_indices(
            n_sample_views=int(raw.sample_raw_indices.size),
            config=cfg,
        )
        candidate_sample_raw_indices = raw.sample_raw_indices[candidate_sample_indices]
        reduced_raw_indices = np.concatenate(
            [candidate_sample_raw_indices, raw.flat_raw_indices, raw.dark_raw_indices]
        )
        reduced_image_key = np.concatenate(
            [
                np.zeros((int(candidate_sample_indices.size),), dtype=np.int32),
                np.ones((int(raw.flat_raw_indices.size),), dtype=np.int32),
                np.full((int(raw.dark_raw_indices.size),), 2, dtype=np.int32),
            ]
        )
        reduced_frames = raw.frames[reduced_raw_indices]
        candidate_angles = raw.angles_all[candidate_sample_raw_indices]

        corrected, flat_mean, dark_mean, warning_counts, correction_meta = correct_nxtomo_frames(
            reduced_frames,
            reduced_image_key,
            config=cfg,
            output_domain=output_domain,
        )

        auto_keep, auto_reject_meta = _auto_reject_views(
            corrected,
            candidate_sample_indices,
            mode=cfg.auto_reject,
            threshold=float(cfg.outlier_z_threshold),
        )
        output_unrepaired = corrected[auto_keep]
        final_sample_indices = candidate_sample_indices[auto_keep]
        final_raw_sample_indices = candidate_sample_raw_indices[auto_keep]
        output_angles = candidate_angles[auto_keep]
        output_repaired = repair_nonfinite_preprocess_output(output_unrepaired, warning_counts)
        output = np.asarray(output_repaired, dtype=output_dtype)
        output_image_key = np.zeros((_array_dim(output, 0),), dtype=np.int32)

        coverage_before = _coverage_stats(raw.angles_all[raw.sample_raw_indices])
        coverage_after = _coverage_stats(output_angles)
        if _coverage_changed(coverage_before, coverage_after):
            LOG.warning(
                "View rejection changed angular coverage: before=%s after=%s",
                coverage_before,
                coverage_after,
            )

        entry = _entry_group(file)
        align_params, angle_offset_deg, optional_meta_found = _optional_sample_metadata(
            entry,
            n_sample_views=int(raw.sample_raw_indices.size),
            selected_sample_view_indices=final_sample_indices,
        )
        metadata = _metadata_from_raw(
            entry,
            thetas_deg=output_angles,
            output_image_key=output_image_key,
            nv=raw.cropped_shape[1],
            nu=raw.cropped_shape[2],
            crop_bounds=raw.crop_bounds,
            original_nv=raw.original_shape[1],
            original_nu=raw.original_shape[2],
            align_params=align_params,
            angle_offset_deg=angle_offset_deg,
        )

    save_nxtomo(str(output_path), output, metadata=metadata)

    provenance = _build_preprocess_provenance(
        input_path=input_path,
        cfg=cfg,
        raw=raw,
        output=output,
        output_dtype=output_dtype,
        output_domain=output_domain,
        correction_meta=correction_meta,
        warning_counts=warning_counts,
        view_selection_meta=view_selection_meta,
        final_sample_indices=final_sample_indices,
        final_raw_sample_indices=final_raw_sample_indices,
        auto_reject_meta=auto_reject_meta,
        coverage_before=coverage_before,
        coverage_after=coverage_after,
        optional_meta_found=optional_meta_found,
    )
    _write_preprocess_provenance(
        output_path,
        provenance=provenance,
        flat_mean=flat_mean.astype(output_dtype, copy=False),
        dark_mean=dark_mean.astype(output_dtype, copy=False),
    )

    return PreprocessResult(
        sample_count=_array_dim(output, 0),
        flat_count=_metadata_int(correction_meta, "flat_count"),
        dark_count=_metadata_int(correction_meta, "dark_count"),
        output_shape=(_array_dim(output, 0), _array_dim(output, 1), _array_dim(output, 2)),
        output_domain=_metadata_str(correction_meta, "output_domain"),
        warning_counts=dict(warning_counts),
        provenance=provenance,
    )


def _load_raw_preprocess_input(
    file: h5py.File,
    cfg: PreprocessConfig,
) -> _RawPreprocessInput:
    frame_path, frame_dataset = _resolve_dataset(
        file,
        explicit_path=cfg.data_path,
        default_paths=_PROJECTION_PATHS,
        label="raw frame",
    )
    if frame_dataset.ndim != 3:
        raise ValueError(
            f"raw frame dataset must be 3D (n_frames, nv, nu), got {frame_dataset.shape}"
        )
    n_frames, raw_nv, raw_nu = (int(v) for v in frame_dataset.shape)
    crop_bounds = _parse_crop_spec(cfg.crop, nv=raw_nv, nu=raw_nu)
    if crop_bounds is None:
        y0, y1, x0, x1 = 0, raw_nv, 0, raw_nu
    else:
        y0, y1, x0, x1 = crop_bounds
    frames: FloatArray = np.asarray(frame_dataset[:, y0:y1, x0:x1], dtype=np.float64)
    cropped_nv, cropped_nu = int(y1 - y0), int(x1 - x0)

    key_path, key_dataset = _resolve_image_key_dataset(file, cfg.image_key_path)
    image_key: npt.NDArray[np.int32] = np.asarray(key_dataset[...], dtype=np.int32).reshape(-1)
    if image_key.shape != (n_frames,):
        raise ValueError(
            f"image_key length must match raw frame count ({n_frames}), got {image_key.shape[0]}"
        )

    angle_path, angle_dataset = _resolve_dataset(
        file,
        explicit_path=cfg.angles_path,
        default_paths=(_ANGLE_PATH,),
        label="rotation angle",
    )
    angles_all = np.asarray(angle_dataset[...], dtype=np.float32).reshape(-1)
    if angles_all.shape[0] != n_frames:
        raise ValueError(
            f"rotation_angle length must match raw frame count ({n_frames}), "
            f"got {angles_all.shape[0]}"
        )

    sample_mask: npt.NDArray[np.bool_] = np.asarray(image_key == 0, dtype=np.bool_)
    flat_mask: npt.NDArray[np.bool_] = np.asarray(image_key == 1, dtype=np.bool_)
    dark_mask: npt.NDArray[np.bool_] = np.asarray(image_key == 2, dtype=np.bool_)
    sample_raw_indices: IntArray = np.asarray(np.flatnonzero(sample_mask), dtype=np.int64)
    flat_raw_indices: IntArray = np.asarray(np.flatnonzero(flat_mask), dtype=np.int64)
    dark_raw_indices: IntArray = np.asarray(np.flatnonzero(dark_mask), dtype=np.int64)
    if sample_raw_indices.size == 0:
        raise ValueError("No sample frames found (image_key==0)")
    return _RawPreprocessInput(
        frame_path=frame_path,
        key_path=key_path,
        angle_path=angle_path,
        frames=frames,
        angles_all=angles_all,
        sample_raw_indices=sample_raw_indices,
        flat_raw_indices=flat_raw_indices,
        dark_raw_indices=dark_raw_indices,
        raw_frame_counts={
            "sample": int(sample_raw_indices.size),
            "flat": int(flat_raw_indices.size),
            "dark": int(dark_raw_indices.size),
        },
        crop_bounds=crop_bounds,
        crop_slices=(y0, y1, x0, x1),
        original_shape=(n_frames, raw_nv, raw_nu),
        cropped_shape=(n_frames, cropped_nv, cropped_nu),
    )


def _build_preprocess_provenance(
    *,
    input_path: Path,
    cfg: PreprocessConfig,
    raw: _RawPreprocessInput,
    output: np.ndarray,
    output_dtype: np.dtype,
    output_domain: str,
    correction_meta: Mapping[str, object],
    warning_counts: dict[str, int],
    view_selection_meta: dict[str, JsonValue],
    final_sample_indices: IntArray,
    final_raw_sample_indices: IntArray,
    auto_reject_meta: dict[str, JsonValue],
    coverage_before: dict[str, float | int | None],
    coverage_after: dict[str, float | int | None],
    optional_meta_found: dict[str, bool],
) -> dict[str, JsonValue]:
    y0, y1, x0, x1 = raw.crop_slices
    payload: dict[str, object] = {
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "input_path": str(input_path),
        "data_path": raw.frame_path,
        "angles_path": raw.angle_path,
        "image_key_path": raw.key_path,
        "frame_counts": raw.raw_frame_counts,
        "processing_frame_counts": {
            "candidate_sample": _metadata_int(correction_meta, "sample_count"),
            "final_sample": _array_dim(output, 0),
            "flat": _metadata_int(correction_meta, "flat_count"),
            "dark": _metadata_int(correction_meta, "dark_count"),
        },
        "output_domain": _metadata_str(correction_meta, "output_domain"),
        "epsilon": float(cfg.epsilon),
        "clip_min": None if cfg.clip_min is None else float(cfg.clip_min),
        "output_dtype": str(output_dtype),
        "correction_formula": (
            "transmission=(sample-mean(dark))/max(mean(flat)-mean(dark),epsilon); "
            "absorption=-log(max(transmission,log_min)) when output_domain=absorption"
        ),
        "log": bool(output_domain == "absorption"),
        "output_domain_policy": "absorption is the default reconstruction-ready domain",
        "assume_dark_field": cfg.assume_dark_field,
        "assume_flat_field": cfg.assume_flat_field,
        "dark_override_used": bool(correction_meta["dark_override_used"]),
        "flat_override_used": bool(correction_meta["flat_override_used"]),
        "warning_counts": dict(warning_counts),
        "select_views": cfg.select_views,
        "reject_views": cfg.reject_views,
        "select_views_file": None if cfg.select_views_file is None else str(cfg.select_views_file),
        "reject_views_file": None if cfg.reject_views_file is None else str(cfg.reject_views_file),
        "view_selection": view_selection_meta,
        "final_sample_view_indices": _json_int_list(final_sample_indices),
        "final_raw_frame_indices": _json_int_list(final_raw_sample_indices),
        "auto_reject": auto_reject_meta,
        "crop": cfg.crop,
        "crop_bounds": None
        if raw.crop_bounds is None
        else {
            "y0": int(y0),
            "y1": int(y1),
            "x0": int(x0),
            "x1": int(x1),
        },
        "original_projection_shape": [int(v) for v in raw.original_shape],
        "cropped_projection_shape": [int(v) for v in raw.cropped_shape],
        "final_projection_shape": [
            _array_dim(output, 0),
            _array_dim(output, 1),
            _array_dim(output, 2),
        ],
        "angular_coverage_before": coverage_before,
        "angular_coverage_after": coverage_after,
        "optional_sample_metadata_filtered": optional_meta_found,
    }
    normalized = normalize_json(payload)
    if not isinstance(normalized, dict):
        raise TypeError("preprocess provenance must normalize to a JSON object")
    return cast("dict[str, JsonValue]", normalized)
