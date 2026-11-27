from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING

import h5py
import numpy as np

from ._io_types import LoadedDataset, SourceInfo

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


def native_file(file: h5py.File) -> bool:
    """Recognize a native container before interpreting acquisition conventions."""
    if "tomojax_format" in file.attrs:
        return True
    geometry = file.get("entry/geometry")
    entry = file.get("entry")
    detector = file.get("entry/instrument/detector")
    if (
        "entry/processing/tomojax" in file
        or (isinstance(entry, h5py.Group) and "grid_meta_json" in entry.attrs)
        or (isinstance(detector, h5py.Group) and "detector_meta_json" in detector.attrs)
        or (
            isinstance(geometry, h5py.Group)
            and (
                "geometry_meta_json" in geometry.attrs
                or (
                    attr_to_str(geometry.attrs.get("NX_class")) == "NXcollection"
                    and "type" in geometry.attrs
                    and isinstance(entry, h5py.Group)
                    and attr_to_str(entry.attrs.get("definition")) == "NXtomo"
                    and "data/projections" in entry
                )
            )
        )
    ):
        raise ValueError(
            f"{file.filename} is an rc6 TomoJAX file. Open and re-save it with "
            "the format-2 migration release (build 4378e691) before opening it here."
        )
    return False


def load_detector_metadata(out: LoadedDataset, entry: h5py.Group) -> None:
    detector = child_group(entry, "instrument/detector")
    out["geometry_meta"] = {
        "detector_center_status": "not_recorded",
        "detector_pixel_size": [_pixel_size(detector, k) for k in ("x_pixel_size", "y_pixel_size")],
    }


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


def load_sample_metadata(out: LoadedDataset, entry: h5py.Group) -> None:
    sample_grp = child_group(entry, "sample")
    name = None if sample_grp is None else child_dataset(sample_grp, "name")
    if name is not None:
        out["sample_name"] = attr_to_str(name[()], default=None)
