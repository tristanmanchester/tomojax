"""Public raw-dataset preprocessing facade."""
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportAny=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnusedCallResult=false

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import imageio.v3 as iio
import numpy as np

from tomojax.data.io_hdf5 import NXTomoMetadata, save_nxtomo
from tomojax.data.preprocess import (
    PreprocessConfig,
    PreprocessResult,
    preprocess_nxtomo,
)
from tomojax.io._contrast import flat_dark_to_absorption, flat_dark_to_transmission

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomojax.core.geometry import Detector, Grid
    from tomojax.core.geometry.base import DetectorDict, GridDict

type PathLike = str | Path

_TIFF_SUFFIXES = {".tif", ".tiff"}
_OUTPUT_DTYPES = {"float32": np.float32, "float64": np.float64}


def preprocess_tiff_stack(
    projections_path: PathLike,
    flats_path: PathLike,
    darks_path: PathLike,
    angles_path: PathLike,
    output_path: PathLike,
    config: PreprocessConfig | None = None,
    *,
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
        cfg.crop,
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

    flat_mean64 = flats.mean(axis=0, dtype=np.float64)
    dark_mean64 = darks.mean(axis=0, dtype=np.float64)
    denominator_raw = flat_mean64 - dark_mean64
    transmission_raw = (projections.astype(np.float64, copy=False) - dark_mean64) / np.maximum(
        denominator_raw, float(cfg.epsilon)
    )
    warning_counts = {
        "nonpositive_flat_denominator": int(np.count_nonzero(denominator_raw <= 0.0)),
        "nonpositive_transmission": int(np.count_nonzero(transmission_raw <= 0.0)),
    }
    if output_domain == "absorption":
        corrected = flat_dark_to_absorption(
            projections,
            flats,
            darks,
            min_intensity=float(cfg.epsilon),
            transmission_min=max(float(cfg.epsilon), float(cfg.clip_min or cfg.epsilon)),
        )
    else:
        corrected = flat_dark_to_transmission(
            projections,
            flats,
            darks,
            denominator_min=float(cfg.epsilon),
            clip_min=None if cfg.clip_min is None else float(cfg.clip_min),
        )
    nonfinite = int(np.count_nonzero(~np.isfinite(corrected)))
    warning_counts["nonfinite_output"] = nonfinite
    output = np.asarray(
        np.nan_to_num(np.asarray(corrected), nan=0.0, posinf=0.0, neginf=0.0),
        dtype=output_dtype,
    )

    nv, nu = (int(v) for v in output.shape[1:])
    metadata = NXTomoMetadata(
        thetas_deg=np.asarray(angles, dtype=np.float32),
        image_key=np.zeros((int(output.shape[0]),), dtype=np.int32),
        detector=_detector_metadata(detector, nv=nv, nu=nu),
        grid=grid,
        geometry_type=str(geometry_type),
        geometry_meta=dict(geometry_metadata or {}),
        sample_name=str(sample_name),
        source_name="TomoJAX TIFF preprocess",
        source_type="tiff_stack",
        source_probe="x-ray",
    )
    output_path = Path(output_path)
    save_nxtomo(str(output_path), output, metadata=metadata)

    provenance: dict[str, Any] = {
        "schema_version": 1,
        "input_format": "tiff_stack",
        "projection_path": str(projections_path),
        "flat_path": str(flats_path),
        "dark_path": str(darks_path),
        "angles_path": str(angles_path),
        "projection_file_count": len(projection_files),
        "flat_file_count": len(flat_files),
        "dark_file_count": len(dark_files),
        "frame_counts": {
            "sample": int(output.shape[0]),
            "flat": int(flats.shape[0]),
            "dark": int(darks.shape[0]),
        },
        "processing_frame_counts": {
            "candidate_sample": int(output.shape[0]),
            "final_sample": int(output.shape[0]),
            "flat": int(flats.shape[0]),
            "dark": int(darks.shape[0]),
        },
        "output_domain": output_domain,
        "output_domain_policy": "absorption is the default reconstruction-ready domain",
        "epsilon": float(cfg.epsilon),
        "clip_min": None if cfg.clip_min is None else float(cfg.clip_min),
        "output_dtype": str(output_dtype),
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
        "crop": cfg.crop,
        "crop_bounds": None
        if crop_bounds is None
        else {
            "y0": int(crop_bounds[0]),
            "y1": int(crop_bounds[1]),
            "x0": int(crop_bounds[2]),
            "x1": int(crop_bounds[3]),
        },
        "original_projection_shape": raw_shape,
        "final_projection_shape": [int(v) for v in output.shape],
        "warning_counts": warning_counts,
    }
    _write_preprocess_provenance(
        output_path,
        provenance=provenance,
        flat_mean=flat_mean64.astype(output_dtype, copy=False),
        dark_mean=dark_mean64.astype(output_dtype, copy=False),
    )
    return PreprocessResult(
        sample_count=int(output.shape[0]),
        flat_count=int(flats.shape[0]),
        dark_count=int(darks.shape[0]),
        output_shape=(int(output.shape[0]), int(output.shape[1]), int(output.shape[2])),
        output_domain=output_domain,
        warning_counts=warning_counts,
        provenance=provenance,
    )


def _validate_public_preprocess_config(config: PreprocessConfig) -> tuple[np.dtype, str]:
    if config.log is not None:
        output_domain = "absorption" if config.log else "transmission"
    else:
        output_domain = str(config.output_domain).strip().lower()
    if output_domain not in {"absorption", "transmission"}:
        raise ValueError("output_domain must be one of: absorption, transmission")
    if not np.isfinite(config.epsilon) or config.epsilon <= 0:
        raise ValueError("epsilon must be a positive finite value")
    if config.clip_min is not None and (not np.isfinite(config.clip_min) or config.clip_min <= 0):
        raise ValueError("clip_min must be a positive finite value when provided")
    if config.output_dtype not in _OUTPUT_DTYPES:
        allowed = ", ".join(sorted(_OUTPUT_DTYPES))
        raise ValueError(f"output_dtype must be one of: {allowed}")
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
    return np.dtype(_OUTPUT_DTYPES[config.output_dtype]), output_domain


def _tiff_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in _TIFF_SUFFIXES:
            raise ValueError(f"{path} is not a TIFF file")
        return [path]
    if path.is_dir():
        return sorted(
            file
            for file in path.iterdir()
            if file.is_file() and file.suffix.lower() in _TIFF_SUFFIXES
        )
    raise FileNotFoundError(path)


def _load_tiff_stack(path: PathLike) -> tuple[np.ndarray, list[Path]]:
    files = _tiff_files(Path(path))
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(val) for val in value]
    return value


def _write_preprocess_provenance(
    output_path: PathLike,
    *,
    provenance: dict[str, Any],
    flat_mean: np.ndarray,
    dark_mean: np.ndarray,
) -> None:
    with h5py.File(output_path, "r+") as file:
        entry = file.require_group("entry")
        processing = entry.require_group("processing")
        processing.attrs["NX_class"] = "NXprocess"
        tomojax = processing.require_group("tomojax")
        tomojax.attrs["NX_class"] = "NXcollection"
        if "preprocess" in tomojax:
            del tomojax["preprocess"]
        group = tomojax.create_group("preprocess")
        group.attrs["NX_class"] = "NXcollection"
        for key, value in provenance.items():
            if isinstance(value, dict | list | tuple):
                group.attrs[key] = json.dumps(_json_safe(value), sort_keys=True)
            elif value is None:
                group.attrs[key] = "null"
            else:
                group.attrs[key] = value
        group.create_dataset(
            "flat_mean",
            data=np.asarray(flat_mean),
            chunks=True,
            compression="lzf",
        )
        group.create_dataset(
            "dark_mean",
            data=np.asarray(dark_mean),
            chunks=True,
            compression="lzf",
        )


__all__ = [
    "PreprocessConfig",
    "PreprocessResult",
    "preprocess_nxtomo",
    "preprocess_tiff_stack",
]
