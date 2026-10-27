"""Content-discovered external NXtomo and Data Exchange raw acquisitions.

Keep partial calibration as metadata, not a numerically plausible Detector.
In particular, pixel pitch does not establish the rotation-axis position.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np

from tomojax._data import locate_frames

from ._acquisition import RawAcquisition
from ._datasets import ProjectionDataset

_MM = {
    "m": 1000.0,
    "mm": 1.0,
    "cm": 10.0,
    "um": 0.001,
    "µm": 0.001,
    "μm": 0.001,
    "micron": 0.001,
    "microns": 0.001,
    "micrometer": 0.001,
    "micrometers": 0.001,
    "nm": 1e-6,
}


def _text(value: Any) -> str:
    value = np.asarray(value).reshape(-1)[0]
    return value.decode() if isinstance(value, bytes) else str(value)


def _length(file: Any, path: str) -> float | None:
    if path not in file:
        return None
    data = file[path]
    unit = _text(data.attrs.get("units", data.attrs.get("unit", ""))).strip().lower()
    # Without a recorded length unit, a physical pitch is not established.
    if unit not in _MM:
        return None
    value = float(np.asarray(data[()]).item()) * _MM[unit]
    return value if np.isfinite(value) else None


def _entries(file: Any) -> list[str]:
    """Acquisition containers under an NXtomo application definition."""
    definitions: list[str] = []
    containers: list[str] = []

    def visit(name: str) -> None:
        group = file.get(name)
        if not isinstance(group, h5py.Group):
            return
        entry = cast(Any, group)  # h5py stubs omit dataset scalar indexing.
        definition = entry.get("definition")
        label = definition[()] if definition is not None else entry.attrs.get("definition", "")
        if _text(label) == "NXtomo":
            definitions.append("/" + name)
        # An NXsubentry may inherit the application definition from its parent.
        if "instrument/detector" in entry:
            containers.append("/" + name)

    file.visit_links(visit)
    return [
        parent
        for parent in containers
        if any(parent == entry or parent.startswith(entry + "/") for entry in definitions)
    ] or definitions


def _layout(file: Any, data_path: str | None) -> tuple[str, str, str, str] | None:
    """Return format, stack, key and angle paths for a recognised acquisition."""
    if "exchange/data" in file:
        where = data_path or "/exchange/data"
        return "data_exchange", where, "", ""
    entries = _entries(file)
    if not entries:
        return None
    if data_path is not None:
        entries = [entry for entry in entries if data_path.startswith(entry + "/")]
    if len(entries) != 1:
        raise ValueError("Several NXtomo entries: pass data_path=... for the detector stack")
    entry = entries[0]
    detector = file.get(entry + "/instrument/detector")
    # TomoJAX's saved metadata/poses are owned by its dataset reader, not raw IO.
    if entry + "/geometry" in file or (
        detector is not None and "detector_meta_json" in detector.attrs
    ):
        return None
    where = data_path or next(
        (
            entry + suffix
            for suffix in ("/instrument/detector/data", "/data/data")
            if entry + suffix in file
        ),
        entry + "/instrument/detector/data",
    )
    key = entry + "/instrument/detector/image_key"
    angles = next(
        (
            entry + "/" + p
            for p in (
                "sample/rotation_angle",
                "sample/transformations/rotation_angle",
                "data/rotation_angle",
            )
            if entry + "/" + p in file
        ),
        "",
    )
    return "nxtomo", where, key if key in file else "", angles


def _metadata(file: Any, layout: str, where: str) -> dict[str, Any]:
    acquisition: dict[str, Any] = {"layout": layout, "data_path": where, "length_unit": "mm"}
    if layout == "data_exchange":
        prefix = "/measurement/instrument/detector/"
        names = ("actual_pixel_size_x", "actual_pixel_size_y")
        for field, path in (
            ("beamline", "/measurement/instrument/source/beamline"),
            ("source", "/measurement/instrument/source/name"),
        ):
            if path in file:
                acquisition[field] = _text(file[path][()])
    else:
        prefix = (
            where.rsplit("/", 2)[0] + "/instrument/detector/"
            if where.endswith("/data/data")
            else where.rsplit("/", 1)[0] + "/"
        )
        names = ("x_pixel_size", "y_pixel_size")
    acquisition["detector_pixel_size"] = [_length(file, prefix + name) for name in names]
    acquisition["pixel_size_paths"] = [prefix + name for name in names if prefix + name in file]
    return acquisition


def read_hdf5_acquisition(
    path: Path,
    *,
    data_path: str | None = None,
    image_key_path: str | None = None,
    angles_path: str | None = None,
) -> RawAcquisition | None:
    """Recognise NXtomo/Data Exchange by their contents, never the file suffix."""
    if not path.is_file() or not h5py.is_hdf5(path):
        return None
    with h5py.File(path, "r") as handle:
        file: Any = handle
        layout = _layout(file, data_path)
        if layout is None:
            return None
        name, where, key, angles = layout
        located = locate_frames(
            str(path),
            data_path=where,
            image_key_path=image_key_path or key or None,
            angles_path=angles_path or angles or None,
        )
        # Recognition describes the layout, not completeness of its calibration.
        # Missing flats/darks still allow explicit caller replacements, even when
        # the acquisition has been renamed to a non-HDF5 suffix.
        if name == "nxtomo" and located.image_key is None:
            # NXtomo may interleave flats/darks with projections; unlike Data
            # Exchange's data array, its stack is not implicitly sample-only.
            raise ValueError(
                "NXtomo has no frame labels: pass image_key_path=... "
                "to identify its sample/flat/dark frames"
            )
        acquisition = _metadata(file, name, where)
        acquisition["angles_path"] = located.angles_path
        acquisition["image_key_path"] = located.image_key_path
        if located.image_key is not None:
            labels, counts = np.unique(located.image_key, return_counts=True)
            acquisition["image_key_counts"] = {
                str(label): int(count) for label, count in zip(labels, counts, strict=True)
            }
        sample_path = (
            "/measurement/sample/name"
            if name == "data_exchange"
            else "/" + where.strip("/").split("/")[0] + "/sample/name"
        )
        sample_name = _text(file[sample_path][()]) if sample_path in file else None
    stack = located.stack
    key_values = located.image_key
    key_values = np.zeros(len(stack), np.int32) if key_values is None else key_values
    sample = key_values == 0
    if not sample.any():
        raise ValueError(f"{path} holds no sample frames (image_key 0)")
    counts = stack.frames_at(sample)
    flats = located.flats
    if np.any(key_values == 1):
        flats = stack.frames_at(key_values == 1)
    darks = located.darks
    if np.any(key_values == 2):
        darks = stack.frames_at(key_values == 2)
    record = ProjectionDataset(
        np.broadcast_to(np.float32(0), counts.shape),
        np.empty(0, np.float32) if located.angles is None else located.angles[sample],
        detector=None,
        geometry_metadata={"acquisition": acquisition, "detector_center_status": "not_recorded"},
        sample_name=sample_name,
        source_path=str(path),
        source_format=name,
    )
    return RawAcquisition(
        record,
        counts,
        flats=flats,
        darks=darks,
        flat_positions=np.cumsum(sample)[key_values == 1] if np.any(key_values == 1) else None,
    )
