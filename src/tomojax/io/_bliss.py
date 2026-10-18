"""Raw ESRF Bliss step scans: sequence-owned metadata and lazy image blocks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from tomojax._data import Hdf5Frames

from ._datasets import ProjectionDataset
from ._frame_blocks import FrameBlocks

if TYPE_CHECKING:
    from ._acquisition import RawAcquisition

_ROTATIONS = (
    "rotm",
    "mhsrot",
    "hsrot",
    "mrsrot",
    "hrsrot",
    "srot",
    "srot_eh2",
    "diffrz",
    "hrrz_trig",
    "rot",
)
_LENGTH_MM: dict[str, float] = {
    "m": 1000,
    "meter": 1000,
    "mm": 1,
    "millimeter": 1,
    "um": 0.001,
    "micrometer": 0.001,
    "µm": 0.001,
    "μm": 0.001,
    "nm": 1e-6,
    "nanometer": 1e-6,
    "cm": 10,
}


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _value(group: Any, key: str, default: Any = None) -> Any:
    return group[key][()] if key in group else default


def _length(group: Any, key: str, *, default_unit: str) -> np.ndarray | None:
    if key not in group:
        return None
    data = group[key]
    unit = _text(data.attrs.get("unit", data.attrs.get("units", default_unit)))
    return np.asarray(data[()], np.float64) * _LENGTH_MM[unit]


def _category(entry: Any) -> int | None:
    if "technique/image_key" in entry:
        return int(entry["technique/image_key"][()])
    title = _text(_value(entry, "title", ""))
    for key, starts in (
        (2, ("dark",)),
        (1, ("reference images", "flat", "ref")),
        (0, ("projections",)),
        (-1, ("static images",)),
    ):
        if title.startswith(starts):
            return key
    return None


def _sequence(file: Any, data_path: str | None) -> tuple[Any, list[Any]] | None:
    entries: dict[str, Any] = {k: g for k, g in file.items() if isinstance(g, h5py.Group)}
    roots: dict[str, Any] = {
        k: g
        for k, g in entries.items()
        if "technique/scan_category" in g or _text(_value(g, "title", "")).startswith("tomo:")
    }
    if not roots:
        return None
    selected = (data_path or _text(file.attrs.get("default", ""))).strip("/")
    if selected in roots:
        root = roots[selected]
    elif data_path or len(roots) != 1:
        raise ValueError(f"{file.filename}: select a Bliss sequence with data_path: {tuple(roots)}")
    else:
        root = next(iter(roots.values()))

    title = _text(_value(root, "title", ""))
    if title not in {"tomo:basic", "tomo:fullturn", "tomo:fulltomo", "tomo:halfturn"}:
        raise ValueError(f"{root.name}: unsupported Bliss sequence {title!r}")
    if "measurement/scan_numbers" in root:
        steps = [root.file[f"{int(n)}.1"] for n in root["measurement/scan_numbers"][()]]
    else:
        # Older Bliss did not record child scan numbers. The next initialization
        # closes the sequence; numeric scan order is not lexicographic order.
        def number(g: Any) -> float:
            return float(g.name.rsplit("/", 1)[-1])

        sibling_entries: list[Any] = [
            g
            for g in root.file.values()
            if isinstance(g, h5py.Group) and _text(_value(g, "title", ""))
        ]
        sibling_roots: list[Any] = [
            g
            for g in sibling_entries
            if "technique/scan_category" in g or _text(_value(g, "title", "")).startswith("tomo:")
        ]
        start = number(root)
        stop = min((number(g) for g in sibling_roots if number(g) > start), default=float("inf"))
        steps = sorted((g for g in sibling_entries if start < number(g) < stop), key=number)
    return root, steps


def _detector_stack(entry: Any, detector_name: str | None) -> Hdf5Frames:
    data: Any
    found: list[Any]
    if detector_name:
        paths = (f"instrument/{detector_name}/data", f"measurement/{detector_name}")
        found = [entry[p] for p in paths if p in entry]
        data = found[0] if found else None
    else:
        found = [
            g["data"]
            for g in entry.get("instrument", {}).values()
            if isinstance(g, h5py.Group)
            and _text(g.attrs.get("NX_class", "")) == "NXdetector"
            and "data" in g
        ]
        if len(found) != 1:
            raise ValueError(f"{entry.name}: no unique Bliss NXdetector")
        data = found[0]
    if data is None or data.ndim != 3:
        raise ValueError(f"{entry.name}: missing Bliss detector stack")
    return Hdf5Frames.open(str(data.file.filename), data.name)


def _angles(entry: Any, root: Any, n: int) -> np.ndarray:
    declared = np.atleast_1d(_value(root, "technique/tomoconfig/rotation", []))
    names = tuple(_text(v) for v in declared) or _ROTATIONS
    # Prefer converted commands across all locations before considering motors.
    # A static positioner or raw channel must not hide a measured angular series.
    for name in (*("calc_" + v for v in names), *names):
        for prefix in ("instrument/positioners", "", "measurement"):
            path = f"{prefix}/{name}".strip("/")
            if path not in entry:
                continue
            data = entry[path]
            angles = np.asarray(data[()], np.float64)
            if angles.size not in (1, n):
                continue
            unit = _text(data.attrs.get("unit", data.attrs.get("units", ""))).lower()
            if unit in ("", "channel"):
                # Encoder channels and unlabelled motors have no established
                # degree scale; leave angles incomplete for a caller override.
                continue
            if unit in ("rad", "radian", "radians"):
                angles = np.rad2deg(angles)
            elif unit not in ("degree", "degrees", "deg"):
                raise ValueError(f"{data.name}: unsupported rotation unit {unit!r}")
            if bool(_value(root, "technique/tomoconfig/rotation_is_clockwise", False)):
                angles = -angles
            return np.broadcast_to(angles, (n,)).astype(np.float32)
    return np.empty(0, np.float32)


def _collect_steps(root: Any, steps: list[Any]):
    name = _value(root, "technique/detector/name")
    if name is None:
        names = np.atleast_1d(_value(root, "technique/tomoconfig/detector", []))
        name = names[0] if len(names) == 1 else None
    name = _text(name) if name is not None else None
    blocks: dict[int, list[Any]] = {0: [], 1: [], 2: []}
    angles, positions, n = [], [], 0
    for entry in steps:
        kind = _category(entry)
        if kind not in blocks:
            continue
        stack = _detector_stack(entry, name)
        blocks[kind].append(stack)
        if kind == 0:
            angles.append(_angles(entry, root, len(stack)))
            n += len(stack)
        elif kind == 1:
            positions.extend([n] * len(stack))
    if n != int(root["technique/scan/tomo_n"][()]):
        raise ValueError(f"{root.name}: projection count does not match Bliss tomo_n")
    return blocks, np.concatenate(angles), np.asarray(positions, np.float64)


def _read_sequence(root: Any, steps: list[Any]) -> RawAcquisition:
    from ._acquisition import RawAcquisition

    blocks, angles, positions = _collect_steps(root, steps)
    counts = FrameBlocks.join(blocks[0])
    pixel = _length(root, "technique/optic/sample_pixel_size", default_unit="um")
    metadata: dict[str, Any] = {"layout": "esrf_bliss", "data_path": root.name, "length_unit": "mm"}
    for key in ("sample_detector_distance", "source_sample_distance"):
        value = _length(root, f"technique/scan/{key}", default_unit="mm")
        if value is not None:
            metadata[key] = float(value)
    # The effective sample pitch sets projection scale; the camera pitch does not.
    metadata["detector_pixel_size"] = (
        [None, None] if pixel is None else np.broadcast_to(pixel, (2,)).tolist()
    )
    camera_pixel = _length(root, "technique/detector/pixel_size", default_unit="um")
    if camera_pixel is not None:
        metadata["camera_pixel_size"] = np.broadcast_to(camera_pixel, (2,)).tolist()
    record = ProjectionDataset(
        projections=np.broadcast_to(np.float32(0), counts.shape),
        angles=angles,
        detector=None,
        source_path=str(root.file.filename),
        source_format="esrf_bliss",
        geometry_metadata={"acquisition": metadata, "detector_center_status": "not_recorded"},
    )
    return RawAcquisition(
        record,
        counts,
        flats=FrameBlocks.join(blocks[1]) if blocks[1] else None,
        darks=FrameBlocks.join(blocks[2]) if blocks[2] else None,
        flat_positions=positions if len(positions) else None,
    )


def read_bliss(path: Path, *, data_path: str | None = None) -> RawAcquisition | None:
    with h5py.File(path, "r") as file:
        selected = _sequence(file, data_path)
        return _read_sequence(*selected) if selected is not None else None
