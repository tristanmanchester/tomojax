"""Public raw-dataset preprocessing facade."""
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportAny=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnusedCallResult=false

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imageio.v3 as iio
import numpy as np

from tomojax._data import (
    NXTomoMetadata,
    save_nxtomo,
)
from tomojax.io._contrast import flat_dark_to_absorption, flat_dark_to_transmission
from tomojax.io._preprocess_impl import (
    PreprocessConfig,
    PreprocessResult,
    preprocess_nxtomo,
    validate_preprocess_numeric_config,
    write_preprocess_provenance,
)
from tomojax.io._tiff import TIFF_SUFFIXES, tiff_files

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomojax.core.geometry import Detector, Grid
    from tomojax.core.geometry.base import DetectorDict, GridDict

type PathLike = str | Path

_TIFF_SUFFIXES = TIFF_SUFFIXES


@dataclass(frozen=True)
class _TiffPreprocessInput:
    projections: np.ndarray
    flats: np.ndarray
    darks: np.ndarray
    angles: np.ndarray
    projection_files: list[Path]
    flat_files: list[Path]
    dark_files: list[Path]
    raw_projection_shape: list[int]
    crop_bounds: tuple[int, int, int, int] | None


@dataclass(frozen=True)
class _CorrectedTiffFrames:
    output: np.ndarray
    flat_mean64: np.ndarray
    dark_mean64: np.ndarray
    warning_counts: dict[str, int]
    output_dtype: np.dtype
    output_domain: str


def preprocess_tiff_stack(
    projections_path: PathLike,
    *,
    flats_path: PathLike,
    darks_path: PathLike,
    angles_path: PathLike,
    output_path: PathLike,
    config: PreprocessConfig | None = None,
    detector: Detector | DetectorDict | None = None,
    grid: Grid | GridDict | None = None,
    geometry_type: str = "parallel",
    geometry_metadata: Mapping[str, Any] | None = None,
    sample_name: str = "sample",
) -> PreprocessResult:
    """Preprocess TIFF projections/flats/darks plus an angle sidecar into NXtomo.

    This path is intentionally explicit rather than a generic loader dispatch:
    TIFF users provide separate stacks for samples, flats, darks, and angles.
    """
    cfg = config or PreprocessConfig()
    output_dtype, output_domain = _validate_public_preprocess_config(cfg)
    inputs = _load_tiff_preprocess_input(
        projections_path=projections_path,
        flats_path=flats_path,
        darks_path=darks_path,
        angles_path=angles_path,
        config=cfg,
    )
    corrected = _correct_tiff_frames(
        inputs,
        config=cfg,
        output_dtype=output_dtype,
        output_domain=output_domain,
    )
    metadata = _metadata_from_tiff_inputs(
        inputs,
        corrected,
        detector=detector,
        grid=grid,
        geometry_type=geometry_type,
        geometry_metadata=geometry_metadata,
        sample_name=sample_name,
    )
    output_path = Path(output_path)
    save_nxtomo(str(output_path), corrected.output, metadata=metadata)

    provenance = _build_tiff_preprocess_provenance(
        inputs,
        corrected,
        projections_path=projections_path,
        flats_path=flats_path,
        darks_path=darks_path,
        angles_path=angles_path,
        config=cfg,
    )
    write_preprocess_provenance(
        output_path,
        provenance=provenance,
        flat_mean=corrected.flat_mean64.astype(output_dtype, copy=False),
        dark_mean=corrected.dark_mean64.astype(output_dtype, copy=False),
    )
    return _result_from_tiff_output(inputs, corrected, provenance=provenance)


def _result_from_tiff_output(
    inputs: _TiffPreprocessInput,
    corrected: _CorrectedTiffFrames,
    *,
    provenance: dict[str, Any],
) -> PreprocessResult:
    return PreprocessResult(
        sample_count=int(corrected.output.shape[0]),
        flat_count=int(inputs.flats.shape[0]),
        dark_count=int(inputs.darks.shape[0]),
        output_shape=(
            int(corrected.output.shape[0]),
            int(corrected.output.shape[1]),
            int(corrected.output.shape[2]),
        ),
        output_domain=corrected.output_domain,
        warning_counts=corrected.warning_counts,
        provenance=provenance,
    )


def _load_tiff_preprocess_input(
    *,
    projections_path: PathLike,
    flats_path: PathLike,
    darks_path: PathLike,
    angles_path: PathLike,
    config: PreprocessConfig,
) -> _TiffPreprocessInput:
    projections, projection_files = _load_tiff_stack(projections_path)
    flats, flat_files = _load_tiff_stack(flats_path)
    darks, dark_files = _load_tiff_stack(darks_path)
    if projections.ndim != 3 or flats.ndim != 3 or darks.ndim != 3:
        raise ValueError("TIFF projections, flats, and darks must be 3D stacks")
    if projections.shape[1:] != flats.shape[1:] or projections.shape[1:] != darks.shape[1:]:
        raise ValueError(
            "TIFF projections, flats, and darks must have matching detector shapes; "
            f"got projections={projections.shape[1:]}, flats={flats.shape[1:]}, "
            f"darks={darks.shape[1:]}"
        )

    raw_shape = [int(v) for v in projections.shape]
    crop_bounds = _parse_crop_spec(
        config.crop,
        nv=int(projections.shape[1]),
        nu=int(projections.shape[2]),
    )
    if crop_bounds is not None:
        y0, y1, x0, x1 = crop_bounds
        projections = projections[:, y0:y1, x0:x1]
        flats = flats[:, y0:y1, x0:x1]
        darks = darks[:, y0:y1, x0:x1]

    angles = _load_angles(angles_path)
    if angles.shape[0] != projections.shape[0]:
        raise ValueError(
            f"angles length {angles.shape[0]} does not match projection count "
            f"{projections.shape[0]}"
        )
    return _TiffPreprocessInput(
        projections=projections,
        flats=flats,
        darks=darks,
        angles=angles,
        projection_files=projection_files,
        flat_files=flat_files,
        dark_files=dark_files,
        raw_projection_shape=raw_shape,
        crop_bounds=crop_bounds,
    )


def _correct_tiff_frames(
    inputs: _TiffPreprocessInput,
    *,
    config: PreprocessConfig,
    output_dtype: np.dtype,
    output_domain: str,
) -> _CorrectedTiffFrames:
    flat_mean64 = inputs.flats.mean(axis=0, dtype=np.float64)
    dark_mean64 = inputs.darks.mean(axis=0, dtype=np.float64)
    denominator_raw = flat_mean64 - dark_mean64
    transmission_raw = (
        inputs.projections.astype(np.float64, copy=False) - dark_mean64
    ) / np.maximum(denominator_raw, float(config.epsilon))
    warning_counts = {
        "nonpositive_flat_denominator": int(np.count_nonzero(denominator_raw <= 0.0)),
        "nonpositive_transmission": int(np.count_nonzero(transmission_raw <= 0.0)),
    }
    if output_domain == "absorption":
        corrected = flat_dark_to_absorption(
            inputs.projections,
            inputs.flats,
            inputs.darks,
            min_intensity=float(config.epsilon),
            transmission_min=max(float(config.epsilon), float(config.clip_min or config.epsilon)),
        )
    else:
        corrected = flat_dark_to_transmission(
            inputs.projections,
            inputs.flats,
            inputs.darks,
            denominator_min=float(config.epsilon),
            clip_min=None if config.clip_min is None else float(config.clip_min),
        )
    nonfinite = int(np.count_nonzero(~np.isfinite(corrected)))
    warning_counts["nonfinite_output"] = nonfinite
    output = np.asarray(
        np.nan_to_num(np.asarray(corrected), nan=0.0, posinf=0.0, neginf=0.0),
        dtype=output_dtype,
    )
    return _CorrectedTiffFrames(
        output=output,
        flat_mean64=flat_mean64,
        dark_mean64=dark_mean64,
        warning_counts=warning_counts,
        output_dtype=output_dtype,
        output_domain=output_domain,
    )


def _metadata_from_tiff_inputs(
    inputs: _TiffPreprocessInput,
    corrected: _CorrectedTiffFrames,
    *,
    detector: Detector | DetectorDict | None,
    grid: Grid | GridDict | None,
    geometry_type: str,
    geometry_metadata: Mapping[str, Any] | None,
    sample_name: str,
) -> NXTomoMetadata:
    nv, nu = (int(v) for v in corrected.output.shape[1:])
    return NXTomoMetadata(
        thetas_deg=np.asarray(inputs.angles, dtype=np.float32),
        image_key=np.zeros((int(corrected.output.shape[0]),), dtype=np.int32),
        detector=_detector_metadata(detector, nv=nv, nu=nu),
        grid=grid,
        geometry_type=str(geometry_type),
        geometry_meta=dict(geometry_metadata or {}),
        sample_name=str(sample_name),
        source_name="TomoJAX TIFF preprocess",
        source_type="tiff_stack",
        source_probe="x-ray",
    )


def _build_tiff_preprocess_provenance(
    inputs: _TiffPreprocessInput,
    corrected: _CorrectedTiffFrames,
    *,
    projections_path: PathLike,
    flats_path: PathLike,
    darks_path: PathLike,
    angles_path: PathLike,
    config: PreprocessConfig,
) -> dict[str, Any]:
    output = corrected.output
    return {
        "schema_version": 1,
        "input_format": "tiff_stack",
        "projection_path": str(projections_path),
        "flat_path": str(flats_path),
        "dark_path": str(darks_path),
        "angles_path": str(angles_path),
        "projection_file_count": len(inputs.projection_files),
        "flat_file_count": len(inputs.flat_files),
        "dark_file_count": len(inputs.dark_files),
        "frame_counts": {
            "sample": int(output.shape[0]),
            "flat": int(inputs.flats.shape[0]),
            "dark": int(inputs.darks.shape[0]),
        },
        "processing_frame_counts": {
            "candidate_sample": int(output.shape[0]),
            "final_sample": int(output.shape[0]),
            "flat": int(inputs.flats.shape[0]),
            "dark": int(inputs.darks.shape[0]),
        },
        "output_domain": corrected.output_domain,
        "output_domain_policy": "absorption is the default reconstruction-ready domain",
        "epsilon": float(config.epsilon),
        "clip_min": None if config.clip_min is None else float(config.clip_min),
        "output_dtype": str(corrected.output_dtype),
        "correction_formula": (
            "transmission=(sample-mean(dark))/max(mean(flat)-mean(dark),epsilon); "
            "absorption=-log(max(transmission,log_min)) when output_domain=absorption"
        ),
        "assume_dark_field": None,
        "assume_flat_field": None,
        "dark_override_used": False,
        "flat_override_used": False,
        "selected_views": list(range(int(output.shape[0]))),
        "rejected_views": [],
        "crop": config.crop,
        "crop_bounds": None
        if inputs.crop_bounds is None
        else {
            "y0": int(inputs.crop_bounds[0]),
            "y1": int(inputs.crop_bounds[1]),
            "x0": int(inputs.crop_bounds[2]),
            "x1": int(inputs.crop_bounds[3]),
        },
        "original_projection_shape": inputs.raw_projection_shape,
        "final_projection_shape": [int(v) for v in output.shape],
        "warning_counts": corrected.warning_counts,
    }


def _validate_public_preprocess_config(config: PreprocessConfig) -> tuple[np.dtype, str]:
    output_dtype, output_domain = validate_preprocess_numeric_config(config)
    if (
        any(
            value is not None
            for value in (
                config.select_views,
                config.reject_views,
                config.select_views_file,
                config.reject_views_file,
            )
        )
        or str(config.auto_reject).strip().lower() != "off"
    ):
        raise ValueError(
            "TIFF stack preprocessing does not support NXtomo view-selection options; "
            "curate the TIFF/angle sidecars before preprocessing"
        )
    return output_dtype, output_domain


def _load_tiff_stack(path: PathLike) -> tuple[np.ndarray, list[Path]]:
    files = tiff_files(Path(path))
    if not files:
        raise ValueError(f"no TIFF files found under {path}")
    stack = np.stack(
        [np.asarray(iio.imread(file), dtype=np.float32) for file in files],
        axis=0,
    )
    return stack, files


def _load_angles(path: PathLike) -> np.ndarray:
    sidecar = Path(path)
    if sidecar.suffix.lower() == ".npy":
        values = np.asarray(np.load(sidecar), dtype=np.float32)
    else:
        rows: list[float] = []
        for line in sidecar.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            token = text.split(",", 1)[0].strip()
            try:
                rows.append(float(token))
            except ValueError:
                if not rows:
                    continue
                raise
        values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("angle sidecar must be one-dimensional")
    return values


def _parse_crop_spec(crop: str | None, *, nv: int, nu: int) -> tuple[int, int, int, int] | None:
    if crop is None or not str(crop).strip():
        return None
    parts = str(crop).strip().split(",")
    if len(parts) != 2:
        raise ValueError("--crop must be formatted as y0:y1,x0:x1")
    y0, y1 = _parse_crop_axis(parts[0], limit=nv, axis_name="y")
    x0, x1 = _parse_crop_axis(parts[1], limit=nu, axis_name="x")
    return y0, y1, x0, x1


def _parse_crop_axis(axis_text: str, *, limit: int, axis_name: str) -> tuple[int, int]:
    parts = axis_text.strip().split(":")
    if len(parts) != 2 or parts[0] == "" or parts[1] == "":
        raise ValueError(f"--crop {axis_name} range must be formatted as start:stop")
    start = int(parts[0])
    stop = int(parts[1])
    if start < 0:
        raise ValueError(f"--crop {axis_name} range must be non-negative")
    if stop <= start:
        raise ValueError(f"--crop {axis_name} range must be non-empty")
    if stop > limit:
        raise ValueError(
            f"--crop {axis_name} range {axis_text!r} is out of bounds for size {limit}"
        )
    return start, stop


def _detector_metadata(
    detector: Detector | DetectorDict | None,
    *,
    nv: int,
    nu: int,
) -> Detector | DetectorDict:
    if detector is not None:
        return detector
    return {"nu": int(nu), "nv": int(nv), "du": 1.0, "dv": 1.0, "det_center": [0.0, 0.0]}


__all__ = [
    "PreprocessConfig",
    "PreprocessResult",
    "preprocess_nxtomo",
    "preprocess_tiff_stack",
]
