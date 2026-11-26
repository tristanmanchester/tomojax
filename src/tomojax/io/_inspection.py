"""Bounded-memory dataset and acquisition inspection helpers owned by the IO boundary."""

from __future__ import annotations

from collections.abc import Iterator
import json
import math
from pathlib import Path
from typing import Any, Protocol, cast

import h5py
import numpy as np
from numpy.typing import NDArray

from tomojax._data import (
    ROTATION_ANGLE_PATHS,
    LocatedFrames,
    locate_frames,
    saved_without_frames,
)
from tomojax._typed_arrays import (
    any_true,
    finite_values_float64,
    linspace_indices,
    max_float,
    min_float,
    numpy_array,
    numpy_float64_array,
    object_list,
    object_mapping,
    sum_float64,
    true_count,
    unique_int_count_map,
)
from tomojax.geometry import Grid
from tomojax.io._acquisition import read_acquisition
from tomojax.io._datasets import ProjectionDataset
from tomojax.io._inspection_types import (
    AlignmentReport,
    AnglesReport,
    DetectorMetadataReport,
    FlatsDarksReport,
    GeometryReport,
    InspectionReport,
    MemoryEstimatesReport,
    NonfiniteReport,
    ProjectionReport,
    ProjectionStatsReport,
    VolumeReport,
    WorkingSetEstimate,
)
from tomojax.io.projection_layout import saved_npz_layout, saved_projection_layout

type PathLike = str | Path


_ALIGN_PATH = "/entry/processing/tomojax/align"  # where the NXtomo writer puts poses
_VOLUME_PATH = "/entry/processing/tomojax/volume"
_DEFAULT_MAX_EXACT_STATS_ELEMENTS = 5_000_000
_MAX_PERCENTILE_SAMPLE_ELEMENTS = 1_000_000


class _ProjectionStack(Protocol):
    """The bounded indexing contract shared by HDF5 and vendor frame stacks."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...

    @property
    def dtype(self) -> np.dtype[Any]: ...

    def __getitem__(self, key: int | slice, /) -> object: ...


def _attr_to_str(value: object, default: str | None = None) -> str | None:
    if value is None:
        return default
    try:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        if isinstance(value, np.ndarray):
            array = numpy_array(cast("object", value))
            if array.shape == ():
                return _attr_to_str(cast("object", array.item()), default)
            if array.size >= 1:
                return _attr_to_str(cast("object", array.flat[0]), default)
        return str(cast("object", value))
    except (TypeError, UnicodeDecodeError, ValueError):
        return default


def _json_attr_to_mapping(value: object) -> dict[str, Any] | None:
    text = _attr_to_str(value)
    if not text:
        return None
    try:
        parsed = cast("object", json.loads(text))
    except (json.JSONDecodeError, TypeError):
        return None
    return object_mapping(cast("object", parsed)) if isinstance(parsed, dict) else None


def _json_safe(value: object) -> object:
    if isinstance(value, np.generic):
        return cast("object", value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in object_list(cast("object", value.tolist()))]
    if isinstance(value, dict):
        return {
            key: _json_safe(item) for key, item in object_mapping(cast("object", value)).items()
        }
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in object_list(cast("object", value))]
    return value


def _dataset_at(file: h5py.File, path: str) -> h5py.Dataset | None:
    obj = file.get(path)
    return obj if isinstance(obj, h5py.Dataset) else None


def _iter_view_blocks(dataset: _ProjectionStack) -> Iterator[NDArray[np.float64]]:
    n_views = int(dataset.shape[0])
    block = 1
    if dataset.ndim == 3:
        plane_elements = max(1, int(dataset.shape[1]) * int(dataset.shape[2]))
        block = max(1, min(n_views, _DEFAULT_MAX_EXACT_STATS_ELEMENTS // plane_elements))
    for start in range(0, n_views, block):
        stop = min(n_views, start + block)
        yield numpy_float64_array(dataset[start:stop])


def _sample_projection_values(dataset: _ProjectionStack) -> NDArray[np.float64]:
    total = math.prod(dataset.shape)
    if total <= _MAX_PERCENTILE_SAMPLE_ELEMENTS:
        return numpy_float64_array(dataset[:]).ravel()

    n_views = int(dataset.shape[0])
    plane_elements = max(1, total // max(1, n_views))
    n_sample_views = max(1, min(n_views, _MAX_PERCENTILE_SAMPLE_ELEMENTS // plane_elements))
    view_indices = linspace_indices(0, n_views - 1, num=n_sample_views)
    samples: list[NDArray[np.float64]] = [
        numpy_float64_array(dataset[int(index)]).ravel() for index in view_indices
    ]
    if not samples:
        return np.asarray([], dtype=np.float64)
    return cast("NDArray[np.float64]", np.concatenate(samples))


def projection_stats(dataset: _ProjectionStack) -> tuple[ProjectionStatsReport, NonfiniteReport]:
    """Return finite-value projection statistics and nonfinite counts for a frame stack.

    Parameters
    ----------
    dataset : array-like
        HDF5 dataset, NumPy array or lazily indexed projection stack, summarized
        in bounded view blocks.

    Returns
    -------
    tuple of dict
        Projection statistics and nonfinite-value counts.
    """
    finite_count = 0
    finite_sum = 0.0
    finite_min = math.inf
    finite_max = -math.inf
    nan_count = 0
    posinf_count = 0
    neginf_count = 0

    for block in _iter_view_blocks(dataset):
        arr = numpy_float64_array(block)
        nan_count += true_count(np.isnan(arr))
        posinf_count += true_count(np.isposinf(arr))
        neginf_count += true_count(np.isneginf(arr))
        finite = np.isfinite(arr)
        if any_true(finite):
            values = finite_values_float64(arr, finite)
            finite_count += int(values.size)
            finite_sum += sum_float64(values)
            finite_min = min(finite_min, min_float(values))
            finite_max = max(finite_max, max_float(values))

    sample = _sample_projection_values(dataset)
    sample_finite = finite_values_float64(sample, np.isfinite(sample))
    if finite_count == 0:
        stats: ProjectionStatsReport = {
            "min": None,
            "p01": None,
            "mean": None,
            "p50": None,
            "p99": None,
            "max": None,
        }
    else:
        if sample_finite.size == 0:
            p01 = None
            p50 = None
            p99 = None
        else:
            percentiles = np.percentile(sample_finite, [1.0, 50.0, 99.0]).tolist()
            p01, p50, p99 = (float(value) for value in percentiles)
        stats = {
            "min": float(finite_min),
            "p01": p01,
            "mean": float(finite_sum / finite_count),
            "p50": p50,
            "p99": p99,
            "max": float(finite_max),
        }

    nonfinite: NonfiniteReport = {
        "nan_count": int(nan_count),
        "posinf_count": int(posinf_count),
        "neginf_count": int(neginf_count),
        "inf_count": int(posinf_count + neginf_count),
    }
    return stats, nonfinite


def _projection_report(
    path: str | None, dataset: _ProjectionStack | None, *, statistics: bool = True
) -> ProjectionReport:
    report: ProjectionReport = {
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
    if dataset is None:
        return report
    if dataset.ndim not in (2, 3):
        raise ValueError(
            f"projection dataset must be (view, u) or (view, v, u), got {dataset.shape}"
        )

    if statistics:
        report["stats"], report["nonfinite"] = projection_stats(dataset)
    n_views, nu = int(dataset.shape[0]), int(dataset.shape[-1])
    nv = 1 if dataset.ndim == 2 else int(dataset.shape[1])
    return {
        "found": True,
        "path": path,
        "shape": [n_views, nv, nu],
        "dtype": str(dataset.dtype),
        "n_views": n_views,
        "detector_shape": {"nv": nv, "nu": nu},
        "storage_bytes": math.prod(dataset.shape) * dataset.dtype.itemsize,
        "stats": report["stats"],
        "nonfinite": report["nonfinite"],
    }


def _angles_report(file: h5py.File, located: LocatedFrames | None) -> AnglesReport:
    if located is None:  # a reconstruction without its frames: the geometry's angles
        path = next((p for p in ROTATION_ANGLE_PATHS if p in file), None)
        dataset = None if path is None else _dataset_at(file, path)
        values = None if dataset is None else np.asarray(dataset[...], np.float64)
        keys = None
    else:
        path, values, keys = located.angles_path, located.angles, located.image_key
        dataset = None if path is None else _dataset_at(file, path)
    units = None if dataset is None else _attr_to_str(dataset.attrs.get("units"))
    return _angle_summary(values, keys, path, units)


def _angle_summary(
    values: np.ndarray | None, keys: np.ndarray | None, path: str | None, units: str | None
) -> AnglesReport:
    """Summarize sample angles, recorded either for every frame or for sample views only."""
    samples = np.asarray([]) if values is None else values
    if keys is not None and samples.size == keys.size:
        samples = samples[keys == 0]
    finite = samples[np.isfinite(samples)]
    low = float(finite.min()) if finite.size else None
    high = float(finite.max()) if finite.size else None
    return {
        "found": bool(values is not None and values.size),
        "path": path,
        "count": None if values is None else int(values.size),
        "units": units,
        "min": low,
        "max": high,
        "coverage": None if low is None or high is None else high - low,
    }


def _geometry_report(file: h5py.File) -> GeometryReport:
    geom = file.get("/entry/geometry")
    found = isinstance(geom, h5py.Group) and "type" in geom.attrs
    meta = None
    if isinstance(geom, h5py.Group):
        meta = _json_attr_to_mapping(geom.attrs.get("geometry_meta_json"))
    return {
        "found": bool(found),
        "type": _attr_to_str(geom.attrs.get("type"))
        if found and isinstance(geom, h5py.Group)
        else None,
        "meta_found": meta is not None,
        "meta_keys": sorted(str(k) for k in meta) if meta is not None else [],
    }


def _volume_report(file: h5py.File | None) -> VolumeReport:
    volume = None if file is None else file.get(_VOLUME_PATH)
    if file is None or not isinstance(volume, h5py.Dataset):
        return {"found": False, "shape": None, "axes": None, "method": None}
    group = file["/entry/processing/tomojax"]
    geom = file.get("/entry/geometry")
    meta = (
        _json_attr_to_mapping(geom.attrs.get("geometry_meta_json"))
        if isinstance(geom, h5py.Group)
        else None
    )
    method = cast("object", meta.get("reconstruction_method")) if meta is not None else None
    return {
        "found": True,
        "shape": [int(v) for v in volume.shape],
        "axes": _attr_to_str(group.attrs.get("volume_axes_order")) or "zyx",
        "method": None if method is None else str(method),
    }


def _detector_metadata_report(file: h5py.File) -> DetectorMetadataReport:
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
            "center": None,
        }
    return {
        "found": True,
        "nu": _json_safe(meta.get("nu")),
        "nv": _json_safe(meta.get("nv")),
        "du": _json_safe(meta.get("du")),
        "dv": _json_safe(meta.get("dv")),
        "center": _json_safe(meta.get("det_center")),  # the file's key for the center
    }


def _flats_darks_report(located: LocatedFrames | None) -> FlatsDarksReport:
    if located is None:
        return {
            "image_key_found": False,
            "image_key_path": None,
            "image_key_counts": {},
            "sample_count": 0,
            "flats_present": False,
            "darks_present": False,
            "flat_count": 0,
            "dark_count": 0,
        }
    keys = located.image_key
    count_map = {} if keys is None else unique_int_count_map(keys)
    sample_count = located.stack.shape[0] if keys is None else int(count_map.get("0", 0))
    flat_count = int(count_map.get("1", 0)) + (0 if located.flats is None else len(located.flats))
    dark_count = int(count_map.get("2", 0)) + (0 if located.darks is None else len(located.darks))
    return {
        "image_key_found": keys is not None,
        "image_key_path": located.image_key_path,
        "image_key_counts": count_map,
        "sample_count": sample_count,
        "flats_present": flat_count > 0,
        "darks_present": dark_count > 0,
        "flat_count": flat_count,
        "dark_count": dark_count,
    }


def _corrections_report(file: h5py.File) -> list[str]:
    """The corrections that made the projections, as recorded with them."""
    geom = file.get("/entry/geometry")
    meta = (
        _json_attr_to_mapping(geom.attrs.get("geometry_meta_json"))
        if isinstance(geom, h5py.Group)
        else None
    )
    return recorded_corrections(None if meta is None else meta.get("corrections"))


def recorded_corrections(saved: object) -> list[str]:
    if not isinstance(saved, list):
        return []
    from tomojax.corrections import Correction

    entries = cast("list[object]", saved)
    return [
        str(Correction.from_dict(cast("dict[str, Any]", c))) for c in entries if isinstance(c, dict)
    ]


def _alignment_report(file: h5py.File | None) -> AlignmentReport:
    align = None if file is None else file.get(_ALIGN_PATH)
    if not isinstance(align, h5py.Group):
        return alignment_fields(None, None, None, None)
    return alignment_fields(
        align.get("thetas"),
        align.get("angle_offset"),
        _json_attr_to_mapping(align.attrs.get("misalign_spec_json")),
        _json_attr_to_mapping(align.attrs.get("gauge_fix_json")),
    )


def alignment_fields(
    params: h5py.Dataset | h5py.Group | np.ndarray | None,
    angle_offset: h5py.Dataset | h5py.Group | np.ndarray | None,
    misalign_spec: object,
    gauge_fix: object,
) -> AlignmentReport:
    params_found = isinstance(params, h5py.Dataset | np.ndarray)
    angle_offset_found = isinstance(angle_offset, h5py.Dataset | np.ndarray)
    return {
        "found": bool(
            params_found or angle_offset_found or misalign_spec is not None or gauge_fix is not None
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
        try:  # the grid that loading this file would build
            saved = Grid.from_dict(grid)
        except (KeyError, TypeError, ValueError):
            pass
        else:
            return [saved.nx, saved.ny, saved.nz]

    det_shape = projection.get("detector_shape")
    if isinstance(det_shape, dict):
        nu = det_shape.get("nu")
        nv = det_shape.get("nv")
        if nu is not None and nv is not None:
            return [int(nu), int(nu), int(nv)]
    return None


def _memory_estimates(
    grid_shape: list[int] | None, projection: ProjectionReport
) -> MemoryEstimatesReport:
    if not projection.get("found"):
        return {
            "feasible": False,
            "reconstruction_grid_shape": None,
            "input_projection_bytes": None,
            "modes": {},
            "notes": "Projection data not found; memory estimates are unavailable.",
        }

    projection_bytes = int(projection["storage_bytes"] or 0)
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
    modes: dict[str, WorkingSetEstimate] = {
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
    """Inspect an HDF5/NXtomo file without materializing TomoJAX/JAX objects.

    A reconstruction saved without its scan holds no frames: its report has no
    projections, flats or darks, and its angles are the geometry's.
    """
    input_path = Path(path)
    located = None if saved_without_frames(input_path) else locate_frames(str(input_path))
    with h5py.File(input_path, "r") as file:
        projection_path = None if located is None else located.stack.dataset
        dataset = None if projection_path is None else _dataset_at(file, projection_path)
        projection = _projection_report(projection_path, dataset)
        report: InspectionReport = {
            "schema_version": 1,
            "input_path": str(input_path),
            "projection": projection,
            "angles": _angles_report(file, located),
            "geometry": _geometry_report(file),
            "detector_metadata": _detector_metadata_report(file),
            "flats_darks": _flats_darks_report(located),
            "corrections": _corrections_report(file),
            "alignment": _alignment_report(file),
            "memory_estimates": _memory_estimates(
                _grid_shape_from_metadata(file, projection), projection
            ),
            "volume": _volume_report(file),
        }
    return report


def _record_report(
    path: PathLike,
    record: ProjectionDataset,
    counts: _ProjectionStack | None,
    *,
    flat_count: int = 0,
    dark_count: int = 0,
    statistics: bool = True,
) -> InspectionReport:
    """A report on a record read whole: a vendor acquisition's frames, or line integrals."""
    projection = _projection_report(
        getattr(counts, "member", getattr(counts, "dataset", str(path))),
        counts,
        statistics=statistics,
    )
    n, nv, nu = (0, 0, 0) if counts is None else counts.shape
    angles = record.angles
    detector = record.detector
    acquisition = record.geometry_metadata.get("acquisition", {})
    key_path = acquisition.get("image_key_path")
    key_counts = acquisition.get("image_key_counts", {})
    if record.image_key is not None:
        key_counts = unique_int_count_map(record.image_key)
        key_path = key_path or "image_key.npy"
    pitches = acquisition.get("detector_pixel_size", [None, None])
    grid_shape = (
        [nu, nu, nv] if record.grid is None else [record.grid.nx, record.grid.ny, record.grid.nz]
    )
    return {
        "schema_version": 1,
        "input_path": str(path),
        "projection": projection,
        "angles": _angle_summary(angles, record.image_key, acquisition.get("angles_path"), "deg"),
        "geometry": {
            "found": True,
            "type": record.geometry_type,
            "meta_found": bool(record.geometry_metadata),
            "meta_keys": sorted(record.geometry_metadata),
        },
        "detector_metadata": {
            "found": detector is not None,
            "nu": nu if detector is None else detector.nu,
            "nv": nv if detector is None else detector.nv,
            "du": pitches[0] if detector is None else detector.du,
            "dv": pitches[1] if detector is None else detector.dv,
            "center": None if detector is None else list(detector.center),
        },
        "flats_darks": {
            "image_key_found": bool(key_path),
            "image_key_path": key_path,
            "image_key_counts": key_counts,
            "sample_count": n
            if record.image_key is None
            else int(np.count_nonzero(record.image_key == 0)),
            "flats_present": flat_count > 0,
            "darks_present": dark_count > 0,
            "flat_count": flat_count,
            "dark_count": dark_count,
        },
        "corrections": recorded_corrections(record.geometry_metadata.get("corrections")),
        "alignment": alignment_fields(
            record.align_params, record.angle_offset, record.misalign_spec, record.align_gauge
        ),
        "volume": _volume_report(None)
        if record.volume is None
        else {
            "found": True,
            "shape": list(record.volume.shape),
            "axes": "zyx",
            "method": record.geometry_metadata.get("reconstruction_method"),
        },
        "memory_estimates": _memory_estimates(grid_shape, projection),
    }


def inspect_dataset(path: PathLike) -> InspectionReport:
    """Inspect a measured projection dataset before reconstruction.

    Parameters
    ----------
    path : str or pathlib.Path
        HDF5/NXtomo dataset or recognized vendor acquisition file or folder.

    Returns
    -------
    InspectionReport
        Projection, acquisition, geometry and estimated memory report.

    Raises
    ------
    ValueError
        If projections or acquisition metadata have invalid dimensions.
    KeyError
        If projection data cannot be found.
    OSError
        If the input cannot be opened.
    """
    if Path(path).suffix.lower() == ".npz":
        record, arrays = saved_npz_layout(Path(path))
        key = record.image_key
        return _record_report(
            path,
            record,
            arrays.get("projections"),
            statistics=False,
            flat_count=int(np.count_nonzero(key == 1)),
            dark_count=int(np.count_nonzero(key == 2)),
        )
    if Path(path).suffix.lower() == ".mat":  # PSI cSAXS phase line integrals
        record, samples = saved_projection_layout(Path(path))
        return _record_report(path, record, samples)
    raw = read_acquisition(path)
    if raw is None:
        return inspect_nxtomo(Path(path))
    flats = len(raw.flats) if raw.flats is not None else int(raw.white_level is not None)
    darks = 0 if raw.darks is None else len(raw.darks)
    counts = cast("_ProjectionStack", raw.counts)
    return _record_report(path, raw.record, counts, flat_count=flats, dark_count=darks)


__all__ = [
    "AlignmentReport",
    "AnglesReport",
    "DetectorMetadataReport",
    "FlatsDarksReport",
    "GeometryReport",
    "InspectionReport",
    "MemoryEstimatesReport",
    "NonfiniteReport",
    "ProjectionReport",
    "ProjectionStatsReport",
    "WorkingSetEstimate",
    "inspect_dataset",
    "inspect_nxtomo",
    "projection_stats",
]
