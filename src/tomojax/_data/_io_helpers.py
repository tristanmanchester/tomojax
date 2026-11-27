from __future__ import annotations

import json
import logging
import math
import os
from typing import TYPE_CHECKING, cast

import h5py
import numpy as np

from ._io_types import JsonObject, LoadedDataset, SourceInfo
from .geometry_meta import normalize_geometry_type

LOG = logging.getLogger(__name__)


if TYPE_CHECKING:
    import jax


def sizes_of(projections: np.ndarray | None, volume: np.ndarray | jax.Array | None) -> str:
    """What a dataset write holds, by part, for its error messages."""
    return ", ".join(
        f"{name} {math.prod(data.shape) * np.dtype(data.dtype).itemsize / 2**30:.1f} GiB"
        for name, data in (("volume", volume), ("projections", projections))
        if data is not None
    )


def _axes_warnings_silenced() -> bool:
    return os.environ.get("TOMOJAX_AXES_SILENCE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def axes_log_warning(message: str, *args: object) -> None:
    if not _axes_warnings_silenced():
        LOG.warning(message, *args)


def attr_to_str(v: object, default: str | None = None) -> str | None:
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
                return attr_to_str(v, default)
            # 1-D array of length 1 or more — take first element
            if v.size >= 1:
                return attr_to_str(v.flat[0], default)
        return str(v)
    except (TypeError, UnicodeDecodeError, ValueError):
        return default


def child_group(parent: h5py.Group, name: str) -> h5py.Group | None:
    """The group at ``name`` under ``parent``, or None when nothing is there."""
    if name not in parent:
        return None
    child = parent[name]  # unlike get(), indexing a broken link raises
    if isinstance(child, h5py.Group):
        return child
    raise ValueError(f"{parent.name}/{name} is a dataset where NXtomo expects a group")


def child_dataset(parent: h5py.Group, name: str) -> h5py.Dataset | None:
    """The dataset at ``name`` under ``parent``, or None when nothing is there."""
    if name not in parent:
        return None
    child = parent[name]  # unlike get(), indexing a broken link raises
    if isinstance(child, h5py.Dataset):
        return child
    raise ValueError(f"{parent.name}/{name} is a group where NXtomo expects a dataset")


def ensure_group(root: h5py.Group, name: str, nx_class: str | None = None) -> h5py.Group:
    g = root.require_group(name)
    if nx_class:
        g.attrs["NX_class"] = nx_class
    return g


def write_string_attr(obj: h5py.Group | h5py.Dataset, key: str, value: str) -> None:
    obj.attrs[key] = np.array(value, dtype=h5py.string_dtype(encoding="utf-8"))


def _load_json_mapping_attr(
    raw_attr: object,
    *,
    path: str,
    context: str,
) -> JsonObject | None:
    if raw_attr is None:
        return None
    payload = attr_to_str(raw_attr)
    if not payload:
        return None
    try:
        loaded = json.loads(payload)
    except json.JSONDecodeError as exc:
        axes_log_warning(
            "load_nxtomo: ignoring malformed %s JSON for %s: %s",
            context,
            path,
            exc,
        )
        return None
    if not isinstance(loaded, dict):
        axes_log_warning(
            "load_nxtomo: ignoring non-object %s JSON for %s",
            context,
            path,
        )
        return None
    return loaded


def detector_group(entry: h5py.Group) -> h5py.Group | None:
    inst_grp = child_group(entry, "instrument")
    return None if inst_grp is None else child_group(inst_grp, "detector")


def load_geometry_metadata(
    out: LoadedDataset,
    entry: h5py.Group,
    *,
    path: str,
) -> None:
    geom = child_group(entry, "geometry")
    geom_type = "parallel"
    if geom is not None and "type" in geom.attrs:
        raw_type = attr_to_str(geom.attrs.get("type"), default="parallel")
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
    if (poses := child_dataset(geom, "simulation_true_poses")) is not None:
        info = meta_dict.get("simulation_info")
        simulation = info.get("simulation") if isinstance(info, dict) else None
        if not isinstance(simulation, dict):
            raise ValueError(f"{path}: simulation poses require simulation metadata")
        simulation["true_poses"] = np.asarray(poses).tolist()
    out["geometry_meta"] = meta_dict
    for key, val in meta_dict.items():
        out.setdefault(key, val)


def load_grid_metadata(out: LoadedDataset, entry: h5py.Group, *, path: str) -> None:
    grid_dict = _load_json_mapping_attr(
        entry.attrs.get("grid_meta_json"),
        path=path,
        context="grid metadata",
    )
    if grid_dict is not None:
        out["grid"] = grid_dict


def load_detector_metadata(
    out: LoadedDataset,
    entry: h5py.Group,
    *,
    path: str,
) -> None:
    det_grp = detector_group(entry)
    detector_dict = None
    if det_grp is not None:
        detector_dict = _load_json_mapping_attr(
            det_grp.attrs.get("detector_meta_json"),
            path=path,
            context="detector metadata",
        )
    # A default numerical zero is not evidence of a recorded/calibrated center.
    # Keep saved provenance, including an earlier inferred or still absent center.
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
        # pixel pitches for inspection, without fabricating the absent center.
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


def load_source_metadata(out: LoadedDataset, entry: h5py.Group) -> None:
    inst_grp = child_group(entry, "instrument")
    if inst_grp is None:
        return
    source_grp = child_group(inst_grp, "SOURCE") or child_group(inst_grp, "source")
    if source_grp is None:
        return
    source_info: SourceInfo = {}
    for key in ("name", "type", "probe"):
        if (value := child_dataset(source_grp, key)) is not None:
            source_info[key] = attr_to_str(value[()])
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
    ("simulation", "artifacts_json", "simulation_artifacts", "simulation artifacts"),
)


def load_processing_metadata(
    out: LoadedDataset,
    entry: h5py.Group,
    *,
    path: str,
) -> tuple[h5py.Dataset | None, str | None]:
    tomojax_grp = child_group(entry, "processing/tomojax")
    if tomojax_grp is None:
        return None, None
    volume = child_dataset(tomojax_grp, "volume")
    volume_axes_attr = attr_to_str(tomojax_grp.attrs.get("volume_axes_order"))

    frame_attr = tomojax_grp.attrs.get("frame")
    if frame_attr is not None:
        frame = attr_to_str(frame_attr)
        if frame:
            out["frame"] = frame

    align_grp = child_group(tomojax_grp, "align")
    if align_grp is not None:
        if (thetas := child_dataset(align_grp, "thetas")) is not None:
            out["align_params"] = np.asarray(thetas[...])
        if (offset := child_dataset(align_grp, "angle_offset")) is not None:
            out["angle_offset"] = np.asarray(offset[...])
    for group, attr, key, context in _PROCESSING_JSON_SECTIONS:
        section = child_group(tomojax_grp, group)
        if section is None:
            continue
        mapping = _load_json_mapping_attr(section.attrs.get(attr), path=path, context=context)
        if mapping is not None:
            out[key] = mapping

    return volume, volume_axes_attr


def load_sample_metadata(out: LoadedDataset, entry: h5py.Group) -> None:
    sample_grp = child_group(entry, "sample")
    name = None if sample_grp is None else child_dataset(sample_grp, "name")
    if name is not None:
        out["sample_name"] = attr_to_str(name[()], default=None)
