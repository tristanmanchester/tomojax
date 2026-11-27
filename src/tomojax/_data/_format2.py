"""The format-2 NeXus tree and its memory-mapped numerical members."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from enum import StrEnum
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
from h5py._objects import phil
import msgspec
import numpy as np

from tomojax import progress
from tomojax.core.geometry.base import Detector, Grid
from tomojax.core.host_volume import release_planes
from tomojax.core.input_files import input_hdf5
from tomojax.core.registration import (
    pose_columns,
    projection_values,
    scan_axes,
    volume_axes,
    volume_values,
)
from tomojax.geometry.api import nominal_axis_unit_from_inputs
from tomojax.wire.arrays import Array, Axis, Coordinates, Hdf5, Uniform

from ._io_acquisition import angles_in_degrees, checked_image_key
from ._io_frames import PinnedFrames, linked_dataset
from ._io_helpers import child_dataset, ensure_group
from ._io_types import LoadedNXTomo, NXTomoMetadata, SourceInfo
from ._record import array_record, metadata_record
from .geometry_meta import normalize_geometry_type

if TYPE_CHECKING:
    import jax


_TILE_BYTES = 32 << 20


class Member(StrEnum):
    """Numerical members of a saved TomoJAX object."""

    VOLUME = "volume"
    IMAGE = "image"
    PHANTOM = "phantom"
    PROJECTIONS = "projections"


def volume_member(meta: Mapping[str, object]) -> Member:
    if meta.get("volume_role") == "phantom":
        return Member.PHANTOM
    if "alignment_info" in meta and "reconstruction_method" not in meta:
        return Member.IMAGE
    return Member.VOLUME


def contiguous(
    group: h5py.Group, name: str, shape: tuple[int, ...], dtype: np.dtype
) -> h5py.Dataset:
    """Allocate the bytes at creation so slab consumers can map the advertised offset."""
    creation = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    creation.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
    return group.create_dataset(
        name, shape=shape, dtype=dtype.newbyteorder("<"), fill_time="never", dcpl=creation
    )


def signal_at(file: h5py.Group, path: str, *, argument: str | None = None) -> h5py.Dataset:
    """Read a required numerical member by the path stored in its record."""
    dataset = linked_dataset(file, path)
    if dataset is None:
        if argument is not None:
            raise KeyError(f"{argument}: {file.file.filename} has no dataset {path!r}")
        raise ValueError(f"missing dataset at {path}")
    if dataset.file.id != file.file.id:
        input_hdf5(dataset.file)
    return dataset


def same_signal(file: h5py.File, selected: h5py.Dataset, recorded: str | None) -> bool:
    """An alternate path inherits units and domain only if it aliases the saved signal."""
    original = file.get(recorded) if recorded else None
    return isinstance(original, h5py.Dataset) and original.id == selected.id


def copy_pixels(target: h5py.Dataset, source: np.ndarray | jax.Array) -> None:
    plane = math.prod(target.shape[1:]) * target.dtype.itemsize
    depth = max(1, _TILE_BYTES // max(1, plane))
    with progress.task("write", str(target.name), math.ceil(target.shape[0] / depth), "slabs"):
        for first in range(0, target.shape[0], depth):
            stop = min(first + depth, target.shape[0])
            target[first:stop] = np.asarray(source[first:stop])
            if isinstance(source, np.ndarray):
                release_planes(source, first, stop)
            progress.step("slabs")
            progress.check_canceled()


def nxdata(
    group: h5py.Group,
    signal: h5py.Dataset,
    axes: Sequence[Axis],
    *,
    unit: str,
    domain: str,
    coordinates: Mapping[str, h5py.Dataset] | None = None,
) -> None:
    """Register the signal and its voxel-center coordinates with NeXus viewers."""
    name = str(signal.name).rsplit("/", 1)[1]
    group.attrs.update(NX_class="NXdata", signal=name)
    if name not in group:
        group[name] = signal
    signal.attrs.update(units=unit, long_name=domain)
    names = []
    for n, axis in zip(signal.shape, axes, strict=True):
        linked = (coordinates or {}).get(axis.name)
        label = str(linked.name).rsplit("/", 1)[1] if linked is not None else axis.name
        names.append(label)
        if linked is not None:
            if label not in group:
                group[label] = linked
        elif isinstance(axis.coordinates, Uniform):
            values = (
                axis.coordinates.origin + np.arange(n, dtype=np.float64) * axis.coordinates.spacing
            )
            group.create_dataset(label, data=values).attrs["units"] = axis.unit
        else:
            raise ValueError(f"missing coordinate table for {axis.name}")
    group.attrs["axes"] = names


def saved_volume_grid(
    scan_grid: Grid | Mapping[str, Any] | None, geometry_meta: Mapping[str, Any] | None
) -> Grid | None:
    """Resolve the voxel grid of a saved reconstruction, phantom or working image.

    Parameters
    ----------
    scan_grid : Grid, mapping or None
        Grid of the saved acquisition.
    geometry_meta : mapping or None
        Result metadata, including a distinct volume or alignment working grid.

    Returns
    -------
    Grid or None
        Grid in the recorded length unit, or None if the file gives none.
    """
    meta = geometry_meta or {}
    raw = meta.get("volume_grid", scan_grid)
    if volume_member(meta) is Member.IMAGE:
        alignment = meta.get("alignment_info")
        if isinstance(alignment, Mapping):
            raw = alignment.get("working_grid") or raw
    if isinstance(raw, Grid):
        return raw
    return Grid.from_dict(raw) if raw else None


def spatial_axes(meta: NXTomoMetadata, shape: tuple[int, ...], unit: str) -> tuple[Axis, ...]:
    grid = saved_volume_grid(meta.grid, meta.geometry_meta) or Grid(
        shape[2], shape[1], shape[0], 1, 1, 1
    )
    if grid.shape != shape:
        raise ValueError(f"volume shape {shape} differs from grid {grid.shape}")
    return volume_axes(grid, unit)


def write_tables(
    record: dict[str, Any],
    meta: NXTomoMetadata,
    sample: h5py.Group,
    det: h5py.Group,
    processing: h5py.Group,
) -> None:
    """Replace inline tables with references to the numerical datasets."""
    tables = record["tables"]
    for key, value, group, name in (
        ("angles", meta.angles, sample, "rotation_angle"),
        ("image_key", meta.image_key, det, "image_key"),
        ("poses", meta.align_params, processing, "poses"),
        ("angle_offset", meta.angle_offset, processing, "angle_offset"),
        ("true_poses", tables.get("true_poses", {}).get("values"), processing, "true_poses"),
    ):
        if value is not None:
            values = np.asarray(value)
            d = group.create_dataset(name, data=values, dtype=values.dtype.newbyteorder("<"))
            table = tables[key]
            table.pop("values")
            table["path"] = d.name
            d.attrs["units"] = table["unit"]
            if key == "angles":
                inputs = dict(meta.geometry_meta or {})
                family = normalize_geometry_type(meta.geometry_type)
                inputs["geometry_type"] = "lamino" if family == "cone" else family
                if family == "parallel" or "tilt" in inputs or "axis_unit_lab" in inputs:
                    d.attrs["transformation_type"] = "rotation"
                    d.attrs["vector"] = nominal_axis_unit_from_inputs(inputs)

            if "columns" in table:
                d.attrs["columns"] = [c["name"] for c in table["columns"]]
                d.attrs["units"] = [c["unit"] for c in table["columns"]]


def measurement_domain(recorded: str | None, key: np.ndarray | None) -> str | None:
    """Resolve the declared quantity and the frame labels of the selected stack."""
    calibration_frames = key is not None and np.isin(key, (1, 2)).any()
    if recorded == "line_integrals" and calibration_frames:
        raise ValueError("image_key_path labels darks or flats in a stack of line integrals")
    if recorded is None:
        return "counts" if calibration_frames else None
    if recorded not in {"line_integrals", "counts"}:
        raise ValueError(f"unknown projection domain {recorded!r}")
    return recorded


def write_projections(
    projections: np.ndarray,
    meta: NXTomoMetadata,
    group: h5py.Group,
    det: h5py.Group,
    angle: h5py.Dataset | None,
    tables: dict,
    unit: str,
    path: Path,
) -> dict:
    """Write the detector signal once and link the default scan plot to its physical axes."""
    shape = tuple(projections.shape)
    for key, values in (("angles", meta.angles), ("image_key", meta.image_key)):
        if values is not None and values.shape != shape[:1]:
            raise ValueError(f"{key} must be shape (n_views,)")
    if "image_key" not in det:
        d = det.create_dataset("image_key", data=np.zeros(shape[0], np.int32))
        tables["image_key"] = {"path": d.name, "unit": "1"}
    fields = meta.detector.to_dict() if isinstance(meta.detector, Detector) else meta.detector or {}
    detector = Detector.from_dict(
        {"nu": shape[2], "nv": shape[1], "du": 1.0, "dv": 1.0, "det_center": [0.0, 0.0], **fields}
    )
    if (detector.nv, detector.nu) != shape[1:]:
        raise ValueError(
            f"projection shape {shape[1:]} differs from detector {(detector.nv, detector.nu)}"
        )
    view = (
        Axis("view", "acquisition", "degree", Coordinates(Hdf5(str(path), str(angle.name))))
        if angle is not None
        else Axis("view", "acquisition", "index", Uniform(0, 1))
    )
    view, v, u = scan_axes(detector, view, unit)
    # A partial detector can give a pitch without a center, or only one pitch.
    # An uncalibrated axis counts pixels from the middle of the panel.
    axes = (
        view,
        v if "dv" in fields else Axis("v", "detector", "pixel", Uniform(-(shape[1] - 1) / 2, 1)),
        u if "du" in fields else Axis("u", "detector", "pixel", Uniform(-(shape[2] - 1) / 2, 1)),
    )
    declared = measurement_domain(meta.projection_domain, meta.image_key)
    counts = (
        declared == "counts" if declared is not None else np.dtype(projections.dtype).kind in "iu"
    )
    value_unit, domain = projection_values(counts=counts)
    d = contiguous(det, "data", shape, np.dtype(projections.dtype))
    copy_pixels(d, projections)
    nxdata(
        group,
        d,
        axes,
        unit=value_unit,
        domain=domain.replace("_", " "),
        coordinates={"view": angle} if angle is not None else {},
    )
    group["image_key"] = det["image_key"]
    description = array_record(d.shape, d.dtype.name, axes, value_unit, domain)
    return {"path": d.name, **description}


def write_instrument(entry: h5py.Group, meta: NXTomoMetadata, unit: str) -> h5py.Group:
    """Create the detector with its measured pitches and record the beam source."""
    instrument = ensure_group(entry, "instrument", "NXinstrument")
    det = ensure_group(instrument, "detector", "NXdetector")
    source = ensure_group(instrument, "source", "NXsource")
    for k, value in (
        ("name", meta.source_name),
        ("type", meta.source_type),
        ("probe", meta.source_probe),
    ):
        if value is not None:
            source[k] = value
    det_info = meta.detector.to_dict() if isinstance(meta.detector, Detector) else meta.detector
    if det_info:
        for key, name in (("du", "x_pixel_size"), ("dv", "y_pixel_size")):
            if key in det_info:
                pitch = det.create_dataset(name, data=det_info[key])
                if unit != "scan_length":
                    pitch.attrs["units"] = unit
    return det


def write_record(
    file: h5py.File,
    path: Path,
    meta: NXTomoMetadata,
    projections: np.ndarray | None,
    volume: h5py.Dataset | None,
) -> None:
    n_views = projections.shape[0] if projections is not None else None
    if n_views is None and meta.angles is not None:
        n_views = len(meta.angles)
    for name, values in (("poses", meta.align_params), ("angle_offset", meta.angle_offset)):
        if values is not None:
            expected = (n_views, 6) if name == "poses" else (n_views,)
            if values.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {values.shape}")
    record = metadata_record(meta)
    geom = meta.geometry_meta or {}
    unit = str(record["length_unit"])
    arrays = {}
    tables = record["tables"]
    assert isinstance(tables, dict)
    member = volume_member(geom)
    entry = ensure_group(file, "entry", "NXentry")
    det = write_instrument(entry, meta, unit)
    sample = ensure_group(entry, "sample", "NXsample")
    processing = ensure_group(entry, "tomojax", "NXcollection")
    if meta.sample_name is not None:
        sample.create_dataset("name", data=meta.sample_name)
        entry["title"] = sample["name"]
    write_tables(record, meta, sample, det, processing)
    if projections is not None:
        group = ensure_group(entry, "data", "NXdata")
        arrays["projections"] = write_projections(
            projections,
            meta,
            group,
            det,
            child_dataset(sample, "rotation_angle"),
            tables,
            unit,
            path,
        )
    if volume is not None:
        value_unit, domain = volume_values(volume.dtype, unit, geom)
        axes = spatial_axes(meta, tuple(volume.shape), unit)
        nxdata(volume.parent, volume, axes, unit=value_unit, domain=domain)
        arrays[member] = {
            "path": volume.name,
            **array_record(volume.shape, volume.dtype.name, axes, value_unit, domain),
        }
    kind = (
        {Member.PHANTOM: "simulation", Member.IMAGE: "alignment", Member.VOLUME: "reconstruction"}[
            member
        ]
        if volume is not None or member == Member.IMAGE
        else "scan"
    )
    record.update(format=2, kind=kind, arrays=arrays)
    default = member if volume is not None else "data"
    entry.attrs["default"] = str(default)
    provenance = record["provenance"]
    file.attrs.update(
        default="entry",
        creator="tomojax",
        tomojax_format=2,
        creator_version=provenance["tomojax"],
        file_time=provenance["written"],
        file_name=path.name,
    )
    entry.create_dataset(
        "record",
        data=json.dumps(record, ensure_ascii=False, allow_nan=False),
        dtype=h5py.string_dtype(),
    )


def open_record(file: h5py.File) -> dict:
    number = file.attrs.get("tomojax_format")
    if number != 2:
        version = file.attrs.get("creator_version", "unknown")
        raise ValueError(
            f"{file.filename}: TomoJAX format {number} written by {version}; supports format 2"
        )
    record = json.loads(signal_at(file, "/entry/record")[()])
    if record.get("format") != number:
        raise ValueError(f"{file.filename}: inconsistent format in root and record")
    return record


def registered_members(file: h5py.File, path: Path) -> dict[str, Array]:
    """Resolve the registered arrays and coordinate tables in this open generation.

    Parameters
    ----------
    file : h5py.File
        Open format-2 container.
    path : Path
        File path to place in storage references.

    Returns
    -------
    dict
        Registered arrays keyed by member name.
    """
    record = open_record(file)
    members = {}
    for name, description in record["arrays"].items():
        if name not in Member:
            continue
        axes = []
        for axis in description["axes"]:
            coords = axis["coordinates"]
            if table := coords.get("table"):
                source = Hdf5(str(path), record["tables"][table]["path"])
                coords = msgspec.to_builtins(Coordinates(source))
            axes.append(msgspec.convert({**axis, "coordinates": coords}, type=Axis))
        dataset = signal_at(file, description["path"])
        members[name] = Array(
            Hdf5(str(path), description["path"]),
            dataset.shape,
            dataset.dtype.str,
            tuple(axes),
            description["unit"],
            description["domain"],
        )
    poses = record["tables"].get("poses")
    if record["kind"] == "alignment" and poses is not None:
        signal = signal_at(file, poses["path"])
        members["poses"] = Array(
            Hdf5(str(path), poses["path"]),
            signal.shape,
            signal.dtype.str,
            (members["projections"].axes[0], Axis("column", "pose", "index", Uniform(0, 1))),
            "mixed",
            "poses",
            columns=pose_columns(record["length_unit"]),
        )
    return members


def local_map(file: h5py.File, field: h5py.Dataset) -> np.memmap | None:
    """Map contiguous pixels through the master descriptor while it names this generation."""
    offset = field.id.get_offset()
    if (
        file.driver != "sec2"
        or field.file.id != file.id
        or offset is None
        or field.id.get_create_plist().get_external_count()
    ):
        return None
    # sec2 caches the position of its descriptor. NumPy seeks to EOF when it maps
    # a file object. Other HDF5 readers can share this open file description.
    # Hold h5py's library lock until its cached position matches the descriptor again.
    with phil, os.fdopen(file.id.get_vfd_handle(), "rb", buffering=0, closefd=False) as stream:
        position = stream.tell()
        try:
            pixels = np.memmap(stream, field.dtype, "r", offset=offset, shape=field.shape)
        finally:
            stream.seek(position)
    pixels.filename = str(Path(file.filename).absolute())
    return pixels


def pixels_of(file: h5py.File, field: h5py.Dataset, *, layout_only: bool) -> np.ndarray:
    if field.dtype.kind not in "biuf":
        raise ValueError(f"{field.name}: expected numerical pixels")
    if layout_only:
        return np.broadcast_to(np.zeros((), field.dtype), field.shape)
    mapped = local_map(file, field)
    if mapped is not None:
        return mapped
    from tomojax.core.files import validate_virtual_pixels

    validate_virtual_pixels(field)
    pixels = field[...]
    pixels.flags.writeable = False
    return pixels


def read_tables(
    file: h5py.File,
    descriptions: Mapping[str, dict],
    n: int | None,
    *,
    image_key_path: str | None,
    angles_path: str | None,
) -> dict[str, np.ndarray]:
    """Read per-view metadata for the same stack and explicit paths as the selected pixels."""
    paths = {
        k: d["path"]
        for k, d in descriptions.items()
        if k in ("angles", "poses", "true_poses", "angle_offset", "image_key")
    }
    overrides = {"image_key": image_key_path, "angles": angles_path}
    paths.update({key: path for key, path in overrides.items() if path is not None})
    fields = {
        key: signal_at(
            file, path, argument=f"{key}_path" if overrides.get(key) is not None else None
        )
        for key, path in paths.items()
    }
    tables = {k: np.asarray(field) for k, field in fields.items()}
    if "image_key" in tables and n is not None:
        tables["image_key"] = checked_image_key(
            tables["image_key"], frames=n, path=f"image_key_path {paths['image_key']}"
        )
    if "angles" in tables:
        angles = fields["angles"]
        saved_angles = descriptions.get("angles", {})
        units = angles.attrs.get("units", angles.attrs.get("unit"))
        if units is None:
            recorded = saved_angles.get("path")
            if same_signal(file, angles, recorded):
                units = saved_angles.get("unit", "degree")
            else:
                raise ValueError(f"angles_path {angles.name!r} has no angular units")
        tables["angles"] = angles_in_degrees(
            tables["angles"],
            units.decode() if isinstance(units, bytes) else str(units),
            frames=n if n is not None else tables["angles"].size,
            path=f"angles_path {paths['angles']}",
            image_key=tables.get("image_key"),
        )
    for key in ("poses", "true_poses"):
        if key in tables and (
            tables[key].ndim != 2
            or tables[key].shape[1] != 6
            or (n is not None and tables[key].shape[0] != n)
        ):
            raise ValueError(
                f"{paths[key]}: expected one pose row per view with exactly 6 columns, "
                f"got shape {tables[key].shape}"
            )

    return tables


def read_format_2(
    file: h5py.File,
    *,
    data_path: str | None = None,
    image_key_path: str | None = None,
    angles_path: str | None = None,
    include_volume: bool = True,
    lazy_projections: bool = False,
    layout_only: bool = False,
) -> LoadedNXTomo:
    record = open_record(file)
    geom = record["geometry"]
    arrays = record["arrays"]
    projections = None
    domain = None
    if "projections" in arrays or data_path is not None:
        saved_path = arrays.get("projections", {}).get("path")
        signal = signal_at(
            file, data_path or saved_path, argument="data_path" if data_path is not None else None
        )
        if signal.ndim != 3:
            raise ValueError(f"data_path {signal.name}: projections must be (view, v, u)")
        if data_path is None or same_signal(file, signal, saved_path):
            domain = arrays["projections"]["domain"]
        projections = (
            PinnedFrames.of(signal)
            if lazy_projections
            else pixels_of(file, signal, layout_only=layout_only)
        )

    tables = read_tables(
        file,
        record["tables"],
        None if projections is None else projections.shape[0],
        image_key_path=image_key_path,
        angles_path=angles_path,
    )
    meta = deepcopy(geom["meta"])
    if "true_poses" in tables:
        simulation = meta.setdefault("simulation_info", {}).setdefault("simulation", {})
        simulation["true_poses"] = tables["true_poses"].tolist()
    source = record.get("source", {})
    metadata = NXTomoMetadata(
        angles=tables.get("angles"),
        projection_domain=measurement_domain(domain, tables.get("image_key")),
        image_key=tables.get("image_key"),
        grid=geom["grid"],
        detector=geom["detector"],
        geometry_type=normalize_geometry_type(geom["type"]),
        geometry_meta=meta,
        align_params=tables.get("poses"),
        align_gauge=record["alignment"]["gauge"],
        geometry_calibration=record["calibration"],
        angle_offset=tables.get("angle_offset"),
        misalign_spec=record["simulation"]["misalign_spec"],
        simulation_artifacts=record["simulation"]["artifacts"],
        frame=record["frame"],
        sample_name=record["sample"]["name"],
        source_name=source.get("name"),
        source_type=source.get("type"),
        source_probe=source.get("probe"),
        history=[*record.get("history", []), record["provenance"]],
    )
    if include_volume:
        for member in (Member.VOLUME, Member.IMAGE, Member.PHANTOM):
            if member in arrays:
                signal = signal_at(file, arrays[member]["path"])
                metadata.volume = pixels_of(file, signal, layout_only=layout_only)
                break
    return LoadedNXTomo(
        projections,
        metadata,
        SourceInfo(name=source.get("name"), type=source.get("type"), probe=source.get("probe")),
        "zyx" if metadata.volume is not None else None,
        record,
    )
