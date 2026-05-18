from __future__ import annotations

import json
import logging
import os

import h5py
import numpy as np

from tomojax.core.geometry.base import DetectorDict
from tomojax.geometry import VOLUME_AXES_ATTR

from ._io_types import JsonObject, LoadedDataset, SourceInfo

LOG = logging.getLogger(__name__)


def _axes_warnings_silenced() -> bool:
    return os.environ.get("TOMOJAX_AXES_SILENCE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _axes_log_warning(message: str, *args: object) -> None:
    if not _axes_warnings_silenced():
        LOG.warning(message, *args)


def _attr_to_str(v: object, default: str | None = None) -> str | None:
    """Robustly convert an HDF5 attribute to a Python string.

    Handles h5py special string dtypes, numpy scalars/arrays, bytes, and plain str.
    """
    if v is None:
        return default
    try:
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="ignore")
        if isinstance(v, np.ndarray):
            if v.shape == ():
                v = v.item()
                return _attr_to_str(v, default)
            # 1-D array of length 1 or more — take first element
            if v.size >= 1:
                return _attr_to_str(v.flat[0], default)
        return str(v)
    except (TypeError, UnicodeDecodeError, ValueError):
        return default


def _ensure_group(root: h5py.Group, name: str, nx_class: str | None = None) -> h5py.Group:
    g = root.require_group(name)
    if nx_class:
        g.attrs["NX_class"] = nx_class
    return g


def _write_string_attr(obj: h5py.Group | h5py.Dataset, key: str, value: str) -> None:
    obj.attrs[key] = np.array(value, dtype=h5py.string_dtype(encoding="utf-8"))


def _normalize_geometry_type(geometry_type: str | None) -> str:
    gtype = "parallel" if geometry_type is None else str(geometry_type).strip().lower()
    if gtype == "parallel":
        return gtype
    if gtype in {"lamino", "laminography"}:
        return "lamino"
    raise ValueError(
        f"Unsupported geometry_type {geometry_type!r}; expected 'parallel' or 'lamino'"
    )


def _load_json_mapping_attr(
    raw_attr: object,
    *,
    path: str,
    context: str,
) -> JsonObject | None:
    if raw_attr is None:
        return None
    payload = _attr_to_str(raw_attr)
    if not payload:
        return None
    try:
        loaded = json.loads(payload)
    except json.JSONDecodeError as exc:
        _axes_log_warning(
            "load_nxtomo: ignoring malformed %s JSON for %s: %s",
            context,
            path,
            exc,
        )
        return None
    if not isinstance(loaded, dict):
        _axes_log_warning(
            "load_nxtomo: ignoring non-object %s JSON for %s",
            context,
            path,
        )
        return None
    return loaded


def _default_detector_meta(projections: np.ndarray) -> DetectorDict:
    nv, nu = int(projections.shape[1]), int(projections.shape[2])
    return {
        "nu": nu,
        "nv": nv,
        "du": 1.0,
        "dv": 1.0,
        "det_center": [0.0, 0.0],
    }


def _detector_group(entry: h5py.Group) -> h5py.Group | None:
    inst_grp = entry.get("instrument")
    if inst_grp is None:
        return None
    return inst_grp.get("detector")


def _load_image_key(entry: h5py.Group, *, n_views: int, path: str) -> np.ndarray:
    det_grp = _detector_group(entry)
    if det_grp is None or "image_key" not in det_grp:
        _axes_log_warning(
            "load_nxtomo: missing image_key for %s; defaulting to zeros",
            path,
        )
        return np.zeros((n_views,), dtype=np.int32)
    return np.asarray(det_grp["image_key"][...], dtype=np.int32)


def _load_rotation_angles(entry: h5py.Group, *, n_views: int, path: str) -> np.ndarray:
    sample_grp = entry.get("sample")
    trans_grp = None if sample_grp is None else sample_grp.get("transformations")
    if trans_grp is None or "rotation_angle" not in trans_grp:
        _axes_log_warning(
            "load_nxtomo: missing rotation_angle for %s; defaulting to zeros",
            path,
        )
        return np.zeros((n_views,), dtype=np.float32)
    return np.asarray(trans_grp["rotation_angle"][...], dtype=np.float32)


def _load_geometry_metadata(
    out: LoadedDataset,
    entry: h5py.Group,
    *,
    path: str,
) -> None:
    geom = entry.get("geometry")
    geom_type = "parallel"
    if geom is not None and "type" in geom.attrs:
        raw_type = _attr_to_str(geom.attrs.get("type"), default="parallel")
        geom_type = _normalize_geometry_type(raw_type)
    out["geometry_type"] = geom_type
    if geom is None:
        return
    meta_dict = _load_json_mapping_attr(
        geom.attrs.get("geometry_meta_json"),
        path=path,
        context="geometry metadata",
    )
    if meta_dict is None:
        return
    out["geometry_meta"] = meta_dict
    for key, val in meta_dict.items():
        out.setdefault(key, val)


def _load_grid_metadata(out: LoadedDataset, entry: h5py.Group, *, path: str) -> None:
    grid_dict = _load_json_mapping_attr(
        entry.attrs.get("grid_meta_json"),
        path=path,
        context="grid metadata",
    )
    if grid_dict is not None:
        out["grid"] = grid_dict


def _load_detector_metadata(
    out: LoadedDataset,
    entry: h5py.Group,
    projections: np.ndarray,
    *,
    path: str,
) -> None:
    det_grp = _detector_group(entry)
    detector_dict = None
    if det_grp is not None:
        detector_dict = _load_json_mapping_attr(
            det_grp.attrs.get("detector_meta_json"),
            path=path,
            context="detector metadata",
        )
    if detector_dict is None:
        _axes_log_warning(
            "load_nxtomo: missing detector metadata for %s; "
            "synthesizing unit detector from projection shape",
            path,
        )
        detector_dict = _default_detector_meta(projections)
    out["detector"] = detector_dict


def _load_source_metadata(out: LoadedDataset, entry: h5py.Group) -> None:
    inst_grp = entry.get("instrument")
    if inst_grp is None:
        return
    source_grp = inst_grp.get("SOURCE") or inst_grp.get("source")
    if source_grp is None:
        return
    source_info: SourceInfo = {}
    for key in ("name", "type", "probe"):
        if key in source_grp:
            source_info[key] = _attr_to_str(source_grp[key][()])
    if not source_info:
        return
    out["source"] = source_info
    if "name" in source_info:
        out["source_name"] = source_info["name"]
    if "type" in source_info:
        out["source_type"] = source_info["type"]
    if "probe" in source_info:
        out["source_probe"] = source_info["probe"]


def _load_processing_metadata(
    out: LoadedDataset,
    entry: h5py.Group,
    *,
    path: str,
) -> tuple[np.ndarray | None, str | None]:
    processing_grp = entry.get("processing")
    tomojax_grp = None if processing_grp is None else processing_grp.get("tomojax")
    if tomojax_grp is None:
        return None, None

    volume_raw = tomojax_grp["volume"][...] if "volume" in tomojax_grp else None
    volume_axes_attr = _attr_to_str(tomojax_grp.attrs.get(VOLUME_AXES_ATTR))

    frame_attr = tomojax_grp.attrs.get("frame")
    if frame_attr is not None:
        frame = _attr_to_str(frame_attr)
        if frame:
            out["frame"] = frame

    align_grp = tomojax_grp.get("align")
    if align_grp is not None:
        if "thetas" in align_grp:
            out["align_params"] = align_grp["thetas"][...]
        align_gauge = _load_json_mapping_attr(
            align_grp.attrs.get("gauge_fix_json"),
            path=path,
            context="alignment gauge fix",
        )
        if align_gauge is not None:
            out["align_gauge"] = align_gauge
        if "angle_offset_deg" in align_grp:
            out["angle_offset_deg"] = align_grp["angle_offset_deg"][...]
        misalign_spec = _load_json_mapping_attr(
            align_grp.attrs.get("misalign_spec_json"),
            path=path,
            context="misalignment spec",
        )
        if misalign_spec is not None:
            out["misalign_spec"] = misalign_spec

    calibration_grp = tomojax_grp.get("calibration")
    if calibration_grp is not None:
        geometry_calibration = _load_json_mapping_attr(
            calibration_grp.attrs.get("geometry_calibration_json"),
            path=path,
            context="geometry calibration",
        )
        if geometry_calibration is not None:
            out["geometry_calibration"] = geometry_calibration

    simulation_grp = tomojax_grp.get("simulation")
    if simulation_grp is not None:
        simulation_artefacts = _load_json_mapping_attr(
            simulation_grp.attrs.get("artefacts_json"),
            path=path,
            context="simulation artefacts",
        )
        if simulation_artefacts is not None:
            out["simulation_artefacts"] = simulation_artefacts

    return volume_raw, volume_axes_attr


def _load_sample_metadata(out: LoadedDataset, entry: h5py.Group) -> None:
    sample_grp = entry.get("sample")
    if sample_grp is not None and "name" in sample_grp:
        out["sample_name"] = _attr_to_str(sample_grp["name"][()], default=None)
