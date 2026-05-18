from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ._preprocess_config import (
    _ANGLE_PATH,
    _PROJECTION_PATHS,
    LOG,
    PREPROCESS_SCHEMA_VERSION,
    PreprocessConfig,
    PreprocessResult,
    _validate_config,
)
from ._preprocess_correction import (
    _auto_reject_views,
    _correct_frames,
    _coverage_changed,
    _coverage_stats,
    _repair_nonfinite_output,
)
from ._preprocess_selection import _parse_crop_spec, _resolve_sample_view_indices
from ._preprocess_source import (
    _metadata_from_raw,
    _optional_sample_metadata,
    _resolve_dataset,
    _resolve_image_key_dataset,
)
from ._preprocess_writer import _write_preprocess_provenance
from .io_hdf5 import save_nxtomo


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
        frames = np.asarray(frame_dataset[:, y0:y1, x0:x1])
        cropped_nv, cropped_nu = int(y1 - y0), int(x1 - x0)

        key_path, key_dataset = _resolve_image_key_dataset(file, cfg.image_key_path)
        image_key = np.asarray(key_dataset[...], dtype=np.int32).reshape(-1)
        if image_key.shape != (n_frames,):
            raise ValueError(
                f"image_key length must match raw frame count ({n_frames}), "
                f"got {image_key.shape[0]}"
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

        sample_mask = image_key == 0
        flat_mask = image_key == 1
        dark_mask = image_key == 2
        sample_raw_indices = np.flatnonzero(sample_mask)
        flat_raw_indices = np.flatnonzero(flat_mask)
        dark_raw_indices = np.flatnonzero(dark_mask)
        raw_frame_counts = {
            "sample": int(sample_raw_indices.size),
            "flat": int(flat_raw_indices.size),
            "dark": int(dark_raw_indices.size),
        }
        if sample_raw_indices.size == 0:
            raise ValueError("No sample frames found (image_key==0)")

        candidate_sample_indices, view_selection_meta = _resolve_sample_view_indices(
            n_sample_views=int(sample_raw_indices.size),
            config=cfg,
        )
        candidate_sample_raw_indices = sample_raw_indices[candidate_sample_indices]
        reduced_raw_indices = np.concatenate(
            [candidate_sample_raw_indices, flat_raw_indices, dark_raw_indices]
        )
        reduced_image_key = np.concatenate(
            [
                np.zeros((int(candidate_sample_indices.size),), dtype=np.int32),
                np.ones((int(flat_raw_indices.size),), dtype=np.int32),
                np.full((int(dark_raw_indices.size),), 2, dtype=np.int32),
            ]
        )
        reduced_frames = frames[reduced_raw_indices]
        candidate_angles = angles_all[candidate_sample_raw_indices]

        corrected, flat_mean, dark_mean, warning_counts, correction_meta = _correct_frames(
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
        output_repaired = _repair_nonfinite_output(output_unrepaired, warning_counts)
        output = np.asarray(output_repaired, dtype=output_dtype)
        output_image_key = np.zeros((int(output.shape[0]),), dtype=np.int32)

        coverage_before = _coverage_stats(angles_all[sample_raw_indices])
        coverage_after = _coverage_stats(output_angles)
        if _coverage_changed(coverage_before, coverage_after):
            LOG.warning(
                "View rejection changed angular coverage: before=%s after=%s",
                coverage_before,
                coverage_after,
            )

        align_params, angle_offset_deg, optional_meta_found = _optional_sample_metadata(
            file["/entry"],
            n_sample_views=int(sample_raw_indices.size),
            selected_sample_view_indices=final_sample_indices,
        )
        metadata = _metadata_from_raw(
            file["/entry"],
            thetas_deg=output_angles,
            output_image_key=output_image_key,
            nv=cropped_nv,
            nu=cropped_nu,
            crop_bounds=crop_bounds,
            original_nv=raw_nv,
            original_nu=raw_nu,
            align_params=align_params,
            angle_offset_deg=angle_offset_deg,
        )

    save_nxtomo(str(output_path), output, metadata=metadata)

    provenance: dict[str, Any] = {
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "input_path": str(input_path),
        "data_path": frame_path,
        "angles_path": angle_path,
        "image_key_path": key_path,
        "frame_counts": raw_frame_counts,
        "processing_frame_counts": {
            "candidate_sample": int(correction_meta["sample_count"]),
            "final_sample": int(output.shape[0]),
            "flat": int(correction_meta["flat_count"]),
            "dark": int(correction_meta["dark_count"]),
        },
        "output_domain": correction_meta["output_domain"],
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
        "final_sample_view_indices": final_sample_indices.tolist(),
        "final_raw_frame_indices": final_raw_sample_indices.tolist(),
        "auto_reject": auto_reject_meta,
        "crop": cfg.crop,
        "crop_bounds": None
        if crop_bounds is None
        else {
            "y0": int(y0),
            "y1": int(y1),
            "x0": int(x0),
            "x1": int(x1),
        },
        "original_projection_shape": [int(n_frames), int(raw_nv), int(raw_nu)],
        "cropped_projection_shape": [int(n_frames), int(cropped_nv), int(cropped_nu)],
        "final_projection_shape": [int(v) for v in output.shape],
        "angular_coverage_before": coverage_before,
        "angular_coverage_after": coverage_after,
        "optional_sample_metadata_filtered": optional_meta_found,
    }
    _write_preprocess_provenance(
        output_path,
        provenance=provenance,
        flat_mean=flat_mean.astype(output_dtype, copy=False),
        dark_mean=dark_mean.astype(output_dtype, copy=False),
    )

    return PreprocessResult(
        sample_count=int(output.shape[0]),
        flat_count=int(correction_meta["flat_count"]),
        dark_count=int(correction_meta["dark_count"]),
        output_shape=(int(output.shape[0]), int(output.shape[1]), int(output.shape[2])),
        output_domain=str(correction_meta["output_domain"]),
        warning_counts=dict(warning_counts),
        provenance=provenance,
    )
