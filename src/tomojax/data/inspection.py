"""Lightweight HDF5/NXtomo inspection helpers."""

from __future__ import annotations

from pathlib import Path
import json
import math
from typing import Any, TypeAlias

import h5py
import imageio.v3 as iio
import numpy as np

from ..recon.quicklook import scale_to_uint8

PathLike: TypeAlias = str | Path

ProjectionReport: TypeAlias = dict[str, Any]
InspectionReport: TypeAlias = dict[str, Any]

_PROJECTION_PATHS = (
    "/entry/instrument/detector/data",
    "/entry/data/projections",
    "/entry/projections",
)
_ANGLE_PATH = "/entry/sample/transformations/rotation_angle"
_IMAGE_KEY_PATH = "/entry/instrument/detector/image_key"
_ALIGN_PATH = "/entry/processing/tomojax/align"
_DEFAULT_MAX_EXACT_STATS_ELEMENTS = 5_000_000
_MAX_PERCENTILE_SAMPLE_ELEMENTS = 1_000_000


def _attr_to_str(value: object, default: str | None = None) -> str | None:
    if value is None:
        return default
    try:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return _attr_to_str(value.item(), default)
            if value.size >= 1:
                return _attr_to_str(value.flat[0], default)
        return str(value)
    except Exception:
        return default


def _json_attr_to_mapping(value: object) -> dict[str, Any] | None:
    text = _attr_to_str(value)
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _json_safe(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    return value


def _dataset_at(file: h5py.File, path: str) -> h5py.Dataset | None:
    try:
        obj = file.get(path)
    except Exception:
        return None
    return obj if isinstance(obj, h5py.Dataset) else None


def _find_projection_dataset(file: h5py.File) -> tuple[str | None, h5py.Dataset | None]:
    for path in _PROJECTION_PATHS:
        dataset = _dataset_at(file, path)
        if dataset is not None:
            return path, dataset
    return None, None


def _iter_view_blocks(dataset: h5py.Dataset) -> object:
    n_views = int(dataset.shape[0])
    block = 1
    if dataset.ndim == 3:
        plane_elements = max(1, int(dataset.shape[1]) * int(dataset.shape[2]))
        block = max(1, min(n_views, _DEFAULT_MAX_EXACT_STATS_ELEMENTS // plane_elements))
    for start in range(0, n_views, block):
        stop = min(n_views, start + block)
        yield np.asarray(dataset[start:stop])


def _sample_projection_values(dataset: h5py.Dataset) -> np.ndarray:
    total = int(dataset.size)
    if total <= _MAX_PERCENTILE_SAMPLE_ELEMENTS:
        return np.asarray(dataset[...]).ravel()

    n_views = int(dataset.shape[0])
    plane_elements = max(1, total // max(1, n_views))
    n_sample_views = max(1, min(n_views, _MAX_PERCENTILE_SAMPLE_ELEMENTS // plane_elements))
    view_indices = np.linspace(0, n_views - 1, num=n_sample_views, dtype=np.int64)
    samples = [np.asarray(dataset[int(i)]).ravel() for i in view_indices]
    if not samples:
        return np.asarray([], dtype=dataset.dtype)
    return np.concatenate(samples)


def _projection_stats(dataset: h5py.Dataset) -> tuple[dict[str, float | None], dict[str, int]]:
    finite_count = 0
    finite_sum = 0.0
    finite_min = math.inf
    finite_max = -math.inf
    nan_count = 0
    posinf_count = 0
    neginf_count = 0

    for block in _iter_view_blocks(dataset):
        arr = np.asarray(block)
        nan_count += int(np.isnan(arr).sum())
        posinf_count += int(np.isposinf(arr).sum())
        neginf_count += int(np.isneginf(arr).sum())
        finite = np.isfinite(arr)
        if np.any(finite):
            values = arr[finite].astype(np.float64, copy=False)
            finite_count += int(values.size)
            finite_sum += float(values.sum(dtype=np.float64))
            finite_min = min(finite_min, float(values.min()))
            finite_max = max(finite_max, float(values.max()))

    sample = _sample_projection_values(dataset)
    sample_finite = sample[np.isfinite(sample)].astype(np.float64, copy=False)
    if finite_count == 0 or sample_finite.size == 0:
        stats = {
            "min": None,
            "p01": None,
            "mean": None,
            "p50": None,
            "p99": None,
            "max": None,
        }
    else:
        p01, p50, p99 = np.percentile(sample_finite, [1.0, 50.0, 99.0])
        stats = {
            "min": float(finite_min),
            "p01": float(p01),
            "mean": float(finite_sum / finite_count),
            "p50": float(p50),
            "p99": float(p99),
            "max": float(finite_max),
        }

    nonfinite = {
        "nan_count": int(nan_count),
        "posinf_count": int(posinf_count),
        "neginf_count": int(neginf_count),
        "inf_count": int(posinf_count + neginf_count),
    }
    return stats, nonfinite


def _projection_report(path: str | None, dataset: h5py.Dataset | None) -> ProjectionReport:
    if dataset is None:
        return {
            "found": False,
            "path": None,
            "shape": None,
            "dtype": None,
            "n_views": None,
            "detector_shape": None,
            "storage_bytes": None,
            "stats": {
                "min": None,
                "p01": None,
                "mean": None,
                "p50": None,
                "p99": None,
                "max": None,
            },
            "nonfinite": {
                "nan_count": None,
                "posinf_count": None,
                "neginf_count": None,
                "inf_count": None,
            },
        }
    if dataset.ndim != 3:
        raise ValueError(f"projection dataset must be 3D (n_views, nv, nu), got {dataset.shape}")

    stats, nonfinite = _projection_stats(dataset)
    n_views, nv, nu = (int(v) for v in dataset.shape)
    return {
        "found": True,
        "path": path,
        "shape": [n_views, nv, nu],
        "dtype": str(dataset.dtype),
        "n_views": n_views,
        "detector_shape": {"nv": nv, "nu": nu},
        "storage_bytes": int(dataset.size * dataset.dtype.itemsize),
        "stats": stats,
        "nonfinite": nonfinite,
    }


def _angles_report(file: h5py.File) -> dict[str, Any]:
    dataset = _dataset_at(file, _ANGLE_PATH)
    if dataset is None:
        return {
            "found": False,
            "path": None,
            "count": None,
            "units": None,
            "min_deg": None,
            "max_deg": None,
            "coverage_deg": None,
        }

    values = np.asarray(dataset[...], dtype=np.float64).ravel()
    finite = values[np.isfinite(values)]
    units = _attr_to_str(dataset.attrs.get("units"))
    if finite.size == 0:
        min_deg = max_deg = coverage_deg = None
    else:
        min_deg = float(finite.min())
        max_deg = float(finite.max())
        coverage_deg = float(max_deg - min_deg)
    return {
        "found": True,
        "path": _ANGLE_PATH,
        "count": int(values.size),
        "units": units,
        "min_deg": min_deg,
        "max_deg": max_deg,
        "coverage_deg": coverage_deg,
    }


def _geometry_report(file: h5py.File) -> dict[str, Any]:
    geom = file.get("/entry/geometry")
    found = isinstance(geom, h5py.Group) and "type" in geom.attrs
    meta = None
    if isinstance(geom, h5py.Group):
        meta = _json_attr_to_mapping(geom.attrs.get("geometry_meta_json"))
    return {
        "found": bool(found),
        "type": _attr_to_str(geom.attrs.get("type")) if found and isinstance(geom, h5py.Group) else None,
        "meta_found": meta is not None,
        "meta_keys": sorted(str(k) for k in meta.keys()) if meta is not None else [],
    }


def _detector_metadata_report(file: h5py.File) -> dict[str, Any]:
    detector = file.get("/entry/instrument/detector")
    meta = (
        _json_attr_to_mapping(detector.attrs.get("detector_meta_json"))
        if isinstance(detector, h5py.Group)
        else None
    )
    if meta is None:
        return {
            "found": False,
            "nu": None,
            "nv": None,
            "du": None,
            "dv": None,
            "det_center": None,
        }
    return {
        "found": True,
        "nu": _json_safe(meta.get("nu")),
        "nv": _json_safe(meta.get("nv")),
        "du": _json_safe(meta.get("du")),
        "dv": _json_safe(meta.get("dv")),
        "det_center": _json_safe(meta.get("det_center")),
    }


def _flats_darks_report(file: h5py.File) -> dict[str, Any]:
    dataset = _dataset_at(file, _IMAGE_KEY_PATH)
    if dataset is None:
        return {
            "image_key_found": False,
            "image_key_counts": {},
            "flats_present": False,
            "darks_present": False,
            "flat_count": 0,
            "dark_count": 0,
        }
    keys = np.asarray(dataset[...], dtype=np.int64).ravel()
    unique, counts = np.unique(keys, return_counts=True)
    count_map = {str(int(k)): int(v) for k, v in zip(unique, counts, strict=True)}
    flat_count = int(count_map.get("1", 0))
    dark_count = int(count_map.get("2", 0))
    return {
        "image_key_found": True,
        "image_key_counts": count_map,
        "flats_present": flat_count > 0,
        "darks_present": dark_count > 0,
        "flat_count": flat_count,
        "dark_count": dark_count,
    }


def _alignment_report(file: h5py.File) -> dict[str, Any]:
    align = file.get(_ALIGN_PATH)
    if not isinstance(align, h5py.Group):
        return {
            "found": False,
            "params_found": False,
            "params_shape": None,
            "angle_offset_found": False,
            "angle_offset_shape": None,
            "misalign_spec_found": False,
            "gauge_fix_found": False,
        }
    params = align.get("thetas")
    angle_offset = align.get("angle_offset_deg")
    misalign_spec = _json_attr_to_mapping(align.attrs.get("misalign_spec_json"))
    gauge_fix = _json_attr_to_mapping(align.attrs.get("gauge_fix_json"))
    params_found = isinstance(params, h5py.Dataset)
    angle_offset_found = isinstance(angle_offset, h5py.Dataset)
    return {
        "found": bool(
            params_found
            or angle_offset_found
            or misalign_spec is not None
            or gauge_fix is not None
        ),
        "params_found": bool(params_found),
        "params_shape": [int(v) for v in params.shape] if params_found else None,
        "angle_offset_found": bool(angle_offset_found),
        "angle_offset_shape": [int(v) for v in angle_offset.shape] if angle_offset_found else None,
        "misalign_spec_found": misalign_spec is not None,
        "gauge_fix_found": gauge_fix is not None,
    }


def _grid_shape_from_metadata(file: h5py.File, projection: ProjectionReport) -> list[int] | None:
    entry = file.get("/entry")
    grid = (
        _json_attr_to_mapping(entry.attrs.get("grid_meta_json"))
        if isinstance(entry, h5py.Group)
        else None
    )
    if grid is not None:
        try:
            return [int(grid["nx"]), int(grid["ny"]), int(grid["nz"])]
        except (KeyError, TypeError, ValueError):
            pass

    det_shape = projection.get("detector_shape")
    if isinstance(det_shape, dict):
        nu = det_shape.get("nu")
        nv = det_shape.get("nv")
        if nu is not None and nv is not None:
            return [int(nu), int(nu), int(nv)]
    return None


def _memory_estimates(file: h5py.File, projection: ProjectionReport) -> dict[str, Any]:
    if not projection.get("found"):
        return {
            "feasible": False,
            "reconstruction_grid_shape": None,
            "input_projection_bytes": None,
            "modes": {},
            "notes": "Projection data not found; memory estimates are unavailable.",
        }

    grid_shape = _grid_shape_from_metadata(file, projection)
    projection_bytes = int(projection["storage_bytes"])
    if grid_shape is None:
        return {
            "feasible": False,
            "reconstruction_grid_shape": None,
            "input_projection_bytes": projection_bytes,
            "modes": {},
            "notes": "Could not infer reconstruction grid shape.",
        }

    voxels = int(grid_shape[0]) * int(grid_shape[1]) * int(grid_shape[2])
    volume_bytes = voxels * np.dtype(np.float32).itemsize
    modes = {
        "fbp_fp32": {
            "estimated_working_set_bytes": int(projection_bytes + 2 * volume_bytes),
        },
        "fista_tv_fp32": {
            "estimated_working_set_bytes": int(projection_bytes + 5 * volume_bytes),
        },
        "spdhg_tv_fp32": {
            "estimated_working_set_bytes": int(projection_bytes + 5 * volume_bytes),
        },
    }
    return {
        "feasible": True,
        "reconstruction_grid_shape": grid_shape,
        "input_projection_bytes": projection_bytes,
        "modes": modes,
        "notes": "Heuristic fp32 working-set estimates; actual JAX/XLA peak memory may differ.",
    }


def inspect_nxtomo(path: PathLike) -> InspectionReport:
    """Inspect an HDF5/NXtomo file without materializing TomoJAX/JAX objects."""

    input_path = Path(path)
    with h5py.File(input_path, "r") as file:
        projection_path, projection_dataset = _find_projection_dataset(file)
        if projection_dataset is None:
            raise KeyError("Could not find projections dataset under /entry")
        projection = _projection_report(projection_path, projection_dataset)
        report = {
            "schema_version": 1,
            "input_path": str(input_path),
            "projection": projection,
            "angles": _angles_report(file),
            "geometry": _geometry_report(file),
            "detector_metadata": _detector_metadata_report(file),
            "flats_darks": _flats_darks_report(file),
            "alignment": _alignment_report(file),
            "memory_estimates": _memory_estimates(file, projection),
        }
    return report


def _fmt_value(value: object, *, precision: int = 6) -> str:
    if value is None:
        return "not found"
    if isinstance(value, float):
        return f"{value:.{precision}g}"
    return str(value)


def _fmt_bool_presence(found: bool) -> str:
    return "present" if found else "not found"


def format_inspection_report(report: InspectionReport) -> str:
    """Format an inspection report for terminal output."""

    projection = report["projection"]
    angles = report["angles"]
    geometry = report["geometry"]
    detector_metadata = report["detector_metadata"]
    flats_darks = report["flats_darks"]
    alignment = report["alignment"]
    memory = report["memory_estimates"]

    lines = [f"TomoJAX inspection: {report['input_path']}"]
    if projection["found"]:
        stats = projection["stats"]
        nonfinite = projection["nonfinite"]
        lines.extend(
            [
                f"Projection shape: {projection['shape']}",
                f"Dtype: {projection['dtype']}",
                f"Views: {projection['n_views']}",
                f"Detector shape: {projection['detector_shape']}",
                (
                    "Stats: "
                    f"min={_fmt_value(stats['min'])}, "
                    f"p01={_fmt_value(stats['p01'])}, "
                    f"mean={_fmt_value(stats['mean'])}, "
                    f"p50={_fmt_value(stats['p50'])}, "
                    f"p99={_fmt_value(stats['p99'])}, "
                    f"max={_fmt_value(stats['max'])}"
                ),
                (
                    "NaN/Inf counts: "
                    f"nan={nonfinite['nan_count']}, "
                    f"+inf={nonfinite['posinf_count']}, "
                    f"-inf={nonfinite['neginf_count']}, "
                    f"inf_total={nonfinite['inf_count']}"
                ),
            ]
        )
    else:
        lines.append("Projection shape: not found")

    if angles["found"]:
        lines.append(
            "Angle coverage: "
            f"{_fmt_value(angles['coverage_deg'])} deg "
            f"(min={_fmt_value(angles['min_deg'])}, "
            f"max={_fmt_value(angles['max_deg'])}, "
            f"count={angles['count']}, "
            f"units={_fmt_value(angles['units'])})"
        )
    else:
        lines.append("Angle coverage: not found")

    lines.append(f"Geometry type: {_fmt_value(geometry['type'])}")
    lines.append(f"Geometry metadata: {_fmt_bool_presence(bool(geometry['meta_found']))}")
    if detector_metadata["found"]:
        lines.append(
            "Detector metadata: "
            f"nu={detector_metadata['nu']}, nv={detector_metadata['nv']}, "
            f"du={detector_metadata['du']}, dv={detector_metadata['dv']}, "
            f"det_center={detector_metadata['det_center']}"
        )
    else:
        lines.append("Detector metadata: not found")

    if flats_darks["flats_present"] or flats_darks["darks_present"]:
        lines.append(
            "Flats/darks: "
            f"flats={flats_darks['flat_count']}, darks={flats_darks['dark_count']}"
        )
    elif flats_darks["image_key_found"]:
        lines.append("Flats/darks: not found (image_key present; no flat/dark frames)")
    else:
        lines.append("Flats/darks: not found")

    if alignment["found"]:
        parts = []
        if alignment["params_found"]:
            parts.append(f"params shape={alignment['params_shape']}")
        if alignment["angle_offset_found"]:
            parts.append(f"angle_offset shape={alignment['angle_offset_shape']}")
        if alignment["misalign_spec_found"]:
            parts.append("misalign_spec present")
        if alignment["gauge_fix_found"]:
            parts.append("gauge_fix present")
        lines.append(f"Alignment parameters: {', '.join(parts)}")
    else:
        lines.append("Alignment parameters: not found")

    if memory["feasible"]:
        lines.append(
            "Memory estimates: "
            f"grid={memory['reconstruction_grid_shape']}, "
            f"fbp_fp32={memory['modes']['fbp_fp32']['estimated_working_set_bytes']} bytes, "
            f"fista_tv_fp32={memory['modes']['fista_tv_fp32']['estimated_working_set_bytes']} bytes, "
            f"spdhg_tv_fp32={memory['modes']['spdhg_tv_fp32']['estimated_working_set_bytes']} bytes"
        )
    else:
        lines.append(f"Memory estimates: not found ({memory['notes']})")

    return "\n".join(lines)


def save_projection_quicklook(input_path: PathLike, output_path: PathLike) -> Path:
    """Save a percentile-scaled central projection PNG."""

    in_path = Path(input_path)
    out_path = Path(output_path)
    with h5py.File(in_path, "r") as file:
        _, dataset = _find_projection_dataset(file)
        if dataset is None:
            raise KeyError("Could not find projections dataset under /entry")
        if dataset.ndim != 3:
            raise ValueError(f"projection dataset must be 3D (n_views, nv, nu), got {dataset.shape}")
        central = np.asarray(dataset[int(dataset.shape[0]) // 2])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, scale_to_uint8(central))
    return out_path
