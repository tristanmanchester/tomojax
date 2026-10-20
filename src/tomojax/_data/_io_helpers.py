from __future__ import annotations

import json
import logging
import os
from typing import cast

import h5py
import numpy as np

from tomojax.geometry.api import VOLUME_AXES_ATTR

from ._io_acquisition import ROTATION_ANGLE_PATHS, angles_in_degrees, checked_image_key
from ._io_types import JsonObject, LoadedDataset, SourceInfo
from .geometry_meta import normalize_geometry_type

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
    return checked_image_key(np.asarray(det_grp["image_key"][...]), frames=n_views, path=path)


def _load_rotation_angles(
    entry: h5py.Group, *, n_views: int, path: str, image_key: np.ndarray
) -> np.ndarray | None:
    angles = next((entry.file[p] for p in ROTATION_ANGLE_PATHS if p in entry.file), None)
    if angles is None:
        return None
    if not isinstance(angles, h5py.Dataset):
        raise ValueError(f"{path}: rotation angles must be a dataset")
    units = _attr_to_str(angles.attrs.get("units"), default="") or ""
    return angles_in_degrees(
        np.asarray(angles[...]), units, frames=n_views, path=path, image_key=image_key
    ).astype(np.float32)


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
        geom_type = normalize_geometry_type(raw_type)
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
    # A default numerical zero is not evidence of a recorded/calibrated centre.
    # Keep saved provenance, including an earlier inferred or still absent centre.
    geometry_meta = dict(cast("JsonObject", out.get("geometry_meta") or {}))
    geometry_meta.setdefault(
        "detector_center_status",
        "recorded"
        if detector_dict is not None and "det_center" in detector_dict
        else "not_recorded",
    )
    out["geometry_meta"] = geometry_meta
    if detector_dict is None or not {"du", "dv", "det_center"} <= detector_dict.keys():
        # A partial calibration cannot construct a Detector. Retain recorded
        # pixel pitches for inspection, without fabricating the absent centre.
        sizes = [_pixel_size(det_grp, name) for name in ("x_pixel_size", "y_pixel_size")]
        geometry_meta["detector_pixel_size"] = list(sizes)
        detector_dict = None
    out["detector"] = detector_dict


def _pixel_size(group: h5py.Group | None, name: str) -> float | None:
    """A positive scalar dataset ``name`` of ``group``, if there is one."""
    found = None if group is None else group.get(name)
    if not isinstance(found, h5py.Dataset) or found.size != 1:
        return None
    value = float(np.asarray(found[()]).reshape(-1)[0])
    return value if np.isfinite(value) and value > 0 else None


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


_PROCESSING_JSON_SECTIONS = (
    ("align", "gauge_fix_json", "align_gauge", "alignment gauge fix"),
    ("align", "misalign_spec_json", "misalign_spec", "misalignment spec"),
    ("calibration", "geometry_calibration_json", "geometry_calibration", "geometry calibration"),
    ("simulation", "artefacts_json", "simulation_artefacts", "simulation artefacts"),
)


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
        if "angle_offset_deg" in align_grp:
            out["angle_offset_deg"] = align_grp["angle_offset_deg"][...]
    for group, attr, key, context in _PROCESSING_JSON_SECTIONS:
        section = tomojax_grp.get(group)
        if section is None:
            continue
        mapping = _load_json_mapping_attr(section.attrs.get(attr), path=path, context=context)
        if mapping is not None:
            out[key] = mapping

    return volume_raw, volume_axes_attr


def _load_sample_metadata(out: LoadedDataset, entry: h5py.Group) -> None:
    sample_grp = entry.get("sample")
    if sample_grp is not None and "name" in sample_grp:
        out["sample_name"] = _attr_to_str(sample_grp["name"][()], default=None)
