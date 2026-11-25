from __future__ import annotations

from contextlib import contextmanager
from itertools import product
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

import h5py
import numpy as np

from tomojax import progress
from tomojax.core.files import replaced_when_complete
from tomojax.geometry.api import nominal_axis_unit_from_inputs

from ._io_acquisition import ROTATION_ANGLE_PATHS, angle_scale_degrees
from ._io_frames import Hdf5Frames
from ._io_helpers import (
    attr_to_str,
    detector_group,
    ensure_group,
    load_detector_metadata,
    load_geometry_metadata,
    load_grid_metadata,
    load_image_key,
    load_processing_metadata,
    load_rotation_angles,
    load_sample_metadata,
    load_source_metadata,
    sizes_of,
    write_string_attr,
)
from ._io_types import LoadedDataset, LoadedNXTomo, NXTomoMetadata, ValidationReport
from .geometry_meta import normalize_geometry_type

if TYPE_CHECKING:
    from collections.abc import Iterator
    from os import PathLike

    import jax

# Where a saved reconstruction keeps its volume, and where a file keeps its frames.
_VOLUME = "/entry/processing/tomojax/volume"
_FRAMES = ("/entry/instrument/detector/data", "/entry/data/projections", "/entry/projections")


def save_nxtomo(
    path: str,
    projections: np.ndarray | None,
    *,
    metadata: NXTomoMetadata | None = None,
    compression: str = "lzf",
    overwrite: bool = True,
) -> None:
    """Write a dataset to an HDF5 file using NXtomo + TomoJAX extras.

    `metadata` owns the persistence contract: geometry, detector/grid definitions,
    optional reconstructed volume, and TomoJAX-specific extras. Volume inputs in
    `metadata.volume` use `(z, y, x)` order, on disk and in memory.

    Parameters
    ----------
    path : str
        NXtomo output file; parent directories are created.
    projections : numpy.ndarray or None
        Projection stack in (view, v, u) order; None for a reconstruction saved without
        its scan, which then needs ``metadata.volume`` and keeps the geometry only.
    metadata : NXTomoMetadata or None, optional
        Acquisition and reconstruction metadata; absent angles stay absent.
        None writes projections without detector or angular calibration.
    compression : str, optional
        HDF5 dataset compression filter; defaults to lzf.
    overwrite : bool, optional
        Replace an existing file; False requires a new path.

    Returns
    -------
    None
        Write projections and metadata to the NXtomo container.

    Raises
    ------
    ValueError
        If projection shape or volume-axis metadata are invalid.
    OSError
        If output cannot be written or already exists with overwrite=False.
    """
    meta = metadata or NXTomoMetadata()
    if not overwrite and Path(path).exists():
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")

    proj = None if projections is None else np.asarray(projections)
    if proj is None and meta.volume is None:
        raise ValueError("a dataset without projections must hold a volume")
    if proj is not None and proj.ndim != 3:
        raise ValueError("projections must be (n_views, nv, nu)")
    if meta.volume is not None and meta.volume.ndim != 3:
        raise ValueError("volume must be 3D (nz, ny, nx)")
    with (
        replaced_when_complete(path, sizes_of(proj, meta.volume)) as temporary,
        h5py.File(temporary, "w") as f,
    ):
        entry = _write_entry(f, meta, proj, compression)
        _write_volume_section(entry, meta, compression=compression)


@contextmanager
def slab_volume(
    path: Path, meta: NXTomoMetadata, shape: tuple[int, int, int]
) -> Iterator[h5py.Dataset]:
    """Open a NeXus reconstruction for single-pass slab writes.

    Parameters
    ----------
    path : Path
        Private, staged output file.
    meta : NXTomoMetadata
        Acquisition geometry and volume grid, including physical origin and units.
    shape : tuple of int
        Full volume dimensions in (z, y, x) order.

    Yields
    ------
    h5py.Dataset
        Registered float32 attenuation dataset, writable until context exit.
    """
    with h5py.File(path, "w") as f:
        entry = _write_entry(f, meta, None, "lzf")
        yield _volume_group(entry, meta).create_dataset(
            "volume", shape=shape, dtype=np.float32, chunks=True, compression="lzf"
        )


def _write_entry(
    f: h5py.File, meta: NXTomoMetadata, proj: np.ndarray | None, compression: str
) -> h5py.Group:
    n_views = len(meta.angles if meta.angles is not None else ()) if proj is None else proj.shape[0]
    geometry_type_norm = normalize_geometry_type(meta.geometry_type)
    entry = ensure_group(f, "entry", "NXentry")
    # NXtomo names its application definition in a field; the attribute
    # remains for readers of earlier TomoJAX files.
    entry.create_dataset("definition", data=np.bytes_(b"NXtomo"))
    write_string_attr(entry, "definition", "NXtomo")
    inst = ensure_group(entry, "instrument", "NXinstrument")

    _write_geometry_section(entry, meta, geometry_type_norm)
    det = _write_detector_section(inst, meta, proj, n_views=n_views, compression=compression)
    _write_sample_section(entry, meta, n_views=n_views)

    if proj is not None:  # NXdata linking for the default plot
        data_grp = ensure_group(entry, "data", "NXdata")
        # store projections also under /entry/data/projections for convenience
        data_grp["projections"] = det["data"]
        write_string_attr(data_grp, "signal", "projections")

    _write_source_section(inst, meta)
    _write_grid_metadata(entry, meta)
    _write_tomojax_metadata_sections(entry, meta, compression=compression)
    return entry


def _write_geometry_section(
    entry: h5py.Group,
    meta: NXTomoMetadata,
    geometry_type_norm: str,
) -> None:
    geom = ensure_group(entry, "geometry", "NXcollection")
    write_string_attr(geom, "type", geometry_type_norm)
    if meta.geometry_meta:
        geom.attrs["geometry_meta_json"] = json.dumps(meta.geometry_meta)


def _write_detector_section(
    inst: h5py.Group,
    meta: NXTomoMetadata,
    proj: np.ndarray | None,
    *,
    n_views: int,
    compression: str,
) -> h5py.Group:
    det = ensure_group(inst, "detector", "NXdetector")
    if proj is not None:  # a reconstruction saved without its scan has only the geometry
        _, nv, nu = proj.shape
        det.create_dataset(
            "data",
            data=proj,
            chunks=(1, min(256, nv), min(256, nu)),
            compression=compression,
        )
        det["data"].attrs["long_name"] = "projections"
        im_key = np.zeros((n_views,), np.int32) if meta.image_key is None else meta.image_key
        im_key = np.asarray(im_key, dtype=np.int32)
        if im_key.shape != (n_views,):
            raise ValueError("image_key must be shape (n_views,)")
        det.create_dataset("image_key", data=im_key, dtype=np.int32)

    if meta.detector is not None:
        det_dict = meta.detector if isinstance(meta.detector, dict) else meta.detector.to_dict()
        det.attrs["detector_meta_json"] = json.dumps(det_dict)
        # Pitches are in the geometry's length unit; label it only when recorded.
        acquisition = (meta.geometry_meta or {}).get("acquisition")
        unit = acquisition.get("length_unit") if isinstance(acquisition, dict) else None
        for name, key in (("x_pixel_size", "du"), ("y_pixel_size", "dv")):
            det.create_dataset(name, data=np.asarray(det_dict[key]))
            if isinstance(unit, str):
                det[name].attrs["units"] = unit
    return det


def _write_sample_section(entry: h5py.Group, meta: NXTomoMetadata, *, n_views: int) -> None:
    sample = ensure_group(entry, "sample", "NXsample")
    sample_name_str = "sample" if meta.sample_name is None else str(meta.sample_name)
    sample.create_dataset(
        "name",
        data=np.array(sample_name_str, dtype=h5py.string_dtype(encoding="utf-8")),
    )
    if meta.angles is None:
        return  # A conversion must not turn absent measurements into zero angles.
    angles = np.asarray(meta.angles, dtype=np.float32)
    if angles.shape != (n_views,):
        raise ValueError("angles must be (n_views,)")
    trans = ensure_group(sample, "transformations", "NXtransformations")
    d_angle = trans.create_dataset("rotation_angle", data=angles)
    d_angle.attrs["units"] = "degree"
    write_string_attr(d_angle, "transformation_type", "rotation")
    # The public NX fields must agree with the saved tilted/arbitrary-axis model.
    # If that model is incomplete, do not manufacture an axial direction.
    inputs = dict(meta.geometry_meta or {})
    gtype = normalize_geometry_type(meta.geometry_type)
    inputs["geometry_type"] = "lamino" if gtype == "cone" else gtype
    if gtype == "parallel" or "tilt" in inputs or "axis_unit_lab" in inputs:
        axis = nominal_axis_unit_from_inputs(inputs)
        trans.create_dataset("rotation_axis", data=np.asarray(axis, dtype=np.float32))
    write_string_attr(trans, "depends_on", "rotation_angle")


def _write_source_section(inst: h5py.Group, meta: NXTomoMetadata) -> None:
    src_grp = ensure_group(inst, "SOURCE", "NXsource")
    if meta.source_name is not None:
        src_grp.create_dataset(
            "name",
            data=np.array(str(meta.source_name), dtype=h5py.string_dtype(encoding="utf-8")),
        )
    if meta.source_type is not None:
        src_grp.create_dataset(
            "type",
            data=np.array(str(meta.source_type), dtype=h5py.string_dtype(encoding="utf-8")),
        )
    if meta.source_probe is not None:
        src_grp.create_dataset(
            "probe",
            data=np.array(str(meta.source_probe), dtype=h5py.string_dtype(encoding="utf-8")),
        )


def _write_grid_metadata(entry: h5py.Group, meta: NXTomoMetadata) -> None:
    if meta.grid is not None:
        gdict = meta.grid if isinstance(meta.grid, dict) else meta.grid.to_dict()
        entry.attrs["grid_meta_json"] = json.dumps(gdict)


def _write_volume_section(
    entry: h5py.Group,
    meta: NXTomoMetadata,
    *,
    compression: str,
) -> None:
    if meta.volume is None:
        return
    tj = _volume_group(entry, meta)
    direct = isinstance(meta.volume, np.ndarray) and meta.volume.flags.c_contiguous
    vol = tj.create_dataset(
        "volume",
        shape=meta.volume.shape,
        dtype=meta.volume.dtype,
        data=meta.volume if direct else None,
        chunks=True,
        compression=compression,
    )
    if not direct:
        _write_in_tiles(vol, meta.volume)
    vol.attrs["long_name"] = "volume"


def _volume_group(entry: h5py.Group, meta: NXTomoMetadata) -> h5py.Group:
    processing = ensure_group(entry, "processing", "NXprocess")
    tj = ensure_group(processing, "tomojax", "NXcollection")
    write_string_attr(tj, "volume_axes_order", "zyx")
    if meta.frame is not None:
        write_string_attr(tj, "frame", str(meta.frame))

    return tj


# Host memory one tile of a saved volume may take.
_TILE_BYTES = 64 * 1024**2


def _write_in_tiles(target: h5py.Dataset, source: np.ndarray | jax.Array) -> None:
    """Write ``source`` in tiles of whole chunks, at most 64 MiB each.

    Device and noncontiguous host arrays are read one tile at a time, never
    as a second copy of the whole volume.
    """
    chunks = target.chunks
    assert chunks is not None  # created chunked
    budget = _TILE_BYTES // target.dtype.itemsize
    tile = list(chunks)
    for axis in reversed(range(3)):  # grow along the fast disk axes first
        others = int(np.prod([n for i, n in enumerate(tile) if i != axis]))
        tile[axis] = min(
            target.shape[axis], max(1, budget // others // chunks[axis]) * chunks[axis]
        )
    starts = tuple(range(0, n, step) for n, step in zip(target.shape, tile, strict=True))
    with progress.task("write", _VOLUME, math.prod(map(len, starts)), "tiles"):
        for start in product(*starts):
            region = tuple(
                slice(lo, min(lo + step, n))
                for lo, step, n in zip(start, tile, target.shape, strict=True)
            )
            target[region] = np.asarray(source[region])
            progress.step("tiles")
            progress.check_cancelled()


def _write_tomojax_metadata_sections(
    entry: h5py.Group,
    meta: NXTomoMetadata,
    *,
    compression: str,
) -> None:
    has_alignment_metadata = (
        meta.align_params is not None
        or meta.align_gauge is not None
        or meta.angle_offset is not None
        or meta.misalign_spec is not None
    )
    if has_alignment_metadata or meta.geometry_calibration is not None:
        processing = ensure_group(entry, "processing", "NXprocess")
        tj = ensure_group(processing, "tomojax", "NXcollection")
        if has_alignment_metadata:
            _write_alignment_metadata_section(tj, meta, compression=compression)
        if meta.geometry_calibration is not None:
            calibration_grp = ensure_group(tj, "calibration", "NXcollection")
            calibration_grp.attrs["geometry_calibration_json"] = json.dumps(
                meta.geometry_calibration
            )

    if meta.simulation_artefacts is not None:
        processing = ensure_group(entry, "processing", "NXprocess")
        tj = ensure_group(processing, "tomojax", "NXcollection")
        simulation_grp = ensure_group(tj, "simulation", "NXcollection")
        simulation_grp.attrs["artefacts_json"] = json.dumps(meta.simulation_artefacts)


def _write_alignment_metadata_section(
    tj: h5py.Group,
    meta: NXTomoMetadata,
    *,
    compression: str,
) -> None:
    align_grp = ensure_group(tj, "align", "NXcollection")
    if meta.align_params is not None:
        dset = align_grp.create_dataset(
            "thetas",
            data=np.asarray(meta.align_params, dtype=np.float32),
            chunks=True,
            compression=compression,
        )
        names = ["alpha", "beta", "phi", "x", "z", "y"][: int(np.shape(meta.align_params)[1])]
        dset.attrs["columns"] = np.array(names, dtype=h5py.string_dtype())
    if meta.align_gauge is not None:
        align_grp.attrs["gauge_fix_json"] = json.dumps(meta.align_gauge)
    if meta.angle_offset is not None:
        align_grp.create_dataset(
            "angle_offset",
            data=np.asarray(meta.angle_offset, dtype=np.float32),
            chunks=True,
            compression=compression,
        )
    if meta.misalign_spec is not None:
        align_grp.attrs["misalign_spec_json"] = json.dumps(meta.misalign_spec)


def _read_projections(stack: Hdf5Frames) -> np.ndarray:
    views, rows, cols = stack.shape
    with progress.task("load", Path(stack.path).name, views, "views"):
        active = progress.for_unit("views")
        if active is None:
            return np.asarray(stack)
        step = max(1, _TILE_BYTES // max(1, rows * cols * stack.dtype.itemsize))
        if step >= views:
            result = np.asarray(stack)
            active.advance(views, {})
            progress.check_cancelled()
            return result
        result = np.empty(stack.shape, stack.dtype)
        for start in range(0, views, step):
            stop = min(views, start + step)
            result[start:stop] = stack[start:stop]
            active.advance(stop, {})
            progress.check_cancelled()
        return result


def load_nxtomo(path: str, *, lazy: bool = False) -> LoadedNXTomo:
    """Load an NXtomo dataset and TomoJAX extras.

    Returns a ``LoadedNXTomo`` payload with typed accessors for projections and
    persisted metadata. ``grid`` is present only when the file stores grid
    metadata; a saved volume does not establish physical voxels. Volumes are returned in
    ``(z, y, x)`` order. With ``lazy`` the projections are an
    :class:`~tomojax._data._io_frames.Hdf5Frames`, read only as indexed.

    Parameters
    ----------
    path : str
        NXtomo/HDF5 input file.
    lazy : bool, optional
        Keep projections lazy rather than materializing them; defaults to False.

    Returns
    -------
    LoadedNXTomo
        Projection payload and metadata, with any saved volume in (z, y, x) order.

    Raises
    ------
    KeyError
        If required NXtomo groups or projection data are absent.
    ValueError
        If per-frame metadata or volume-axis conventions are invalid.
    OSError
        If the input cannot be opened.
    """
    return read_nxtomo(path, lazy=lazy, layout_only=False)


def read_nxtomo(path: str, *, lazy: bool, layout_only: bool) -> LoadedNXTomo:
    out: LoadedDataset = {}
    volume_np = None

    with h5py.File(path, "r") as f:
        entry = cast("h5py.Group", f["/entry"])
        det_grp = detector_group(entry)
        if det_grp is not None and "data" in det_grp:
            where = "/entry/instrument/detector/data"
        elif "data" in entry and "projections" in entry["data"]:
            where = "/entry/data/projections"
        elif "projections" in entry:
            where = "/entry/projections"
        elif _without_frames(f):  # a reconstruction saved without its scan
            where = None
        else:
            raise KeyError("Could not find projections dataset under /entry")
        if where is None:
            out["projections"] = None
            angles = next((f[p] for p in ROTATION_ANGLE_PATHS if p in f), None)
            n_views = 0 if angles is None else int(cast("h5py.Dataset", angles).shape[0])
            image_key = np.zeros(n_views, np.int32)
        else:
            stack = Hdf5Frames.open(path, where)
            proj = stack if lazy else _read_projections(stack)
            out["projections"] = proj
            n_views = proj.shape[0]
            image_key = load_image_key(entry, n_views=n_views, path=path)
            out["image_key"] = image_key
        out["angles"] = load_rotation_angles(entry, n_views=n_views, path=path, image_key=image_key)
        load_geometry_metadata(out, entry, path=path)
        load_grid_metadata(out, entry, path=path)
        load_detector_metadata(out, entry, path=path)
        load_source_metadata(out, entry)
        volume, volume_axes_attr = load_processing_metadata(out, entry, path=path)
        load_sample_metadata(out, entry)

        if volume is not None:
            volume_np, disk_order = _volume_in_internal_axes(
                volume, volume_axes_attr, path=path, layout_only=layout_only
            )
            out["disk_volume_axes_order"] = disk_order
    payload = LoadedNXTomo.from_dataset(out)
    payload.metadata.volume = volume_np
    return payload


def _volume_in_internal_axes(
    volume: h5py.Dataset, volume_axes_attr: str | None, *, path: str, layout_only: bool
) -> tuple[np.ndarray, str]:
    """A saved volume in (z, y, x) order, with its disk order."""
    disk_axes = saved_volume_axes(volume_axes_attr, path)
    volume_np = (
        np.broadcast_to(np.zeros((), volume.dtype), volume.shape) if layout_only else volume[...]
    )
    if disk_axes != "zyx":
        volume_np = volume_np.transpose(_volume_permutation(disk_axes))
        if not layout_only:
            volume_np = np.ascontiguousarray(volume_np)
    return volume_np, disk_axes


def saved_without_frames(path: str | PathLike[str]) -> bool:
    """Whether ``path`` is a reconstruction saved without its scan: a volume and no frames.

    Parameters
    ----------
    path : str or path-like
        HDF5 file to look in.

    Returns
    -------
    bool
        True when the file holds a saved volume and no stack of frames.
    """
    with h5py.File(os.fspath(path), "r") as file:
        return _without_frames(file)


def _without_frames(file: h5py.File) -> bool:
    return isinstance(file.get(_VOLUME), h5py.Dataset) and not any(p in file for p in _FRAMES)


def saved_volume_axes(attribute: str | None, path: object) -> str:
    """The disk order of a saved volume, from its ``volume_axes_order`` attribute.

    TomoJAX records the order with every volume it writes; the shape alone cannot
    say it (a cube fits every order), so a volume without the attribute is refused.

    Parameters
    ----------
    attribute : str or None
        The volume's ``volume_axes_order`` attribute, if any.
    path : object
        The file, for messages.

    Returns
    -------
    str
        The order, such as ``"zyx"``.

    Raises
    ------
    ValueError
        If the attribute is absent or not a permutation of ``xyz``.
    """
    if not attribute:
        raise ValueError(
            f"{path}: the saved volume does not record its axis order; "
            "set its volume_axes_order attribute (for example 'zyx')"
        )
    try:
        _volume_permutation(attribute)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return attribute.lower()


def _volume_permutation(axes: str) -> tuple[int, int, int]:
    """Reader-only permutation from a declared storage order to zyx."""
    axes = axes.lower()
    if len(axes) != 3 or set(axes) != {"x", "y", "z"}:
        raise ValueError(f"volume axes must be a permutation of xyz, got {axes!r}")
    return axes.index("z"), axes.index("y"), axes.index("x")


def validate_nxtomo(path: str) -> ValidationReport:
    """Check the lightweight NXtomo schema and report any issues.

    Parameters
    ----------
    path : str
        NXtomo file to check.

    Returns
    -------
    dict
        Schema report; an empty issues list means no problems were found.
    """
    report: ValidationReport = {"issues": []}
    try:
        with h5py.File(path, "r") as f:
            if "/entry" not in f:
                report["issues"].append("Missing /entry")
                return report
            e = cast("h5py.Group", f["/entry"])
            issues, angle_path, n_views = _required_entry_issues(f, e)
            report["issues"].extend(issues)
            if "instrument/detector/image_key" in e:
                report["issues"].extend(
                    _image_key_issues(
                        cast("h5py.Dataset", e["instrument/detector/image_key"]), frames=n_views
                    )
                )
            if angle_path is not None:
                report["issues"].extend(
                    _rotation_angle_issues(
                        cast("h5py.Dataset", f[angle_path]), frames=n_views, path=path
                    )
                )
    except OSError as exc:
        report["issues"].append(f"Unable to read HDF5 file: {exc}")
    return report


def _required_entry_issues(f: h5py.File, e: h5py.Group) -> tuple[list[str], str | None, int | None]:
    """Missing or misshapen required NXtomo members, the angle path and the view count."""
    issues = []
    field = e.get("definition")
    stored = np.asarray(field).tolist() if isinstance(field, h5py.Dataset) else None
    stored = stored.decode() if isinstance(stored, bytes) else stored
    if "NXtomo" not in (stored, attr_to_str(e.attrs.get("definition"))):
        issues.append("entry/definition must be 'NXtomo'")
    data = e.get("instrument/detector/data")
    if data is None:
        issues.append("Missing instrument/detector/data")
    angle_path = next((p for p in ROTATION_ANGLE_PATHS if p in f), None)
    if angle_path is None:
        issues.append("Missing sample rotation_angle")
    if "instrument/detector/image_key" not in e:
        issues.append("Missing instrument/detector/image_key")
    if "sample" not in e or "name" not in e["sample"]:
        issues.append("Missing sample/name")
    n_views = None
    if data is not None:
        if not isinstance(data, h5py.Dataset) or data.ndim not in (2, 3):
            issues.append("instrument/detector/data must be (n_views, nu) or (n_views, nv, nu)")
        else:
            n_views = data.shape[0]
    return issues, angle_path, n_views


def _image_key_issues(image_key: h5py.Dataset, *, frames: int | None) -> list[str]:
    """Shape, dtype and value issues in an image_key dataset."""
    issues = []
    if image_key.ndim != 1:
        issues.append("instrument/detector/image_key must be 1D (n_views,)")
    elif frames is not None and image_key.shape[0] != frames:
        issues.append("image_key length must match #views in detector/data")
    if image_key.dtype.kind not in {"i", "u"}:
        issues.append("image_key must use integer dtype")
    elif image_key.ndim == 1:
        values = np.asarray(image_key[...])
        allowed = np.asarray([0, 1, 2, 3], dtype=values.dtype)
        invalid_values, invalid_counts = np.unique(
            values[~np.isin(values, allowed)], return_counts=True
        )
        if invalid_values.size > 0:
            formatted = ", ".join(
                f"{int(value)} ({int(count)} frame{'s' if int(count) != 1 else ''})"
                for value, count in zip(invalid_values, invalid_counts, strict=True)
            )
            issues.append("image_key values must be in {0, 1, 2, 3}; found " + formatted)
    return issues


def _rotation_angle_issues(angles: h5py.Dataset, *, frames: int | None, path: str) -> list[str]:
    """Shape and unit issues in a rotation-angle dataset."""
    issues = []
    if angles.ndim != 1:
        issues.append("rotation_angle must be 1D (n_views,)")
    elif frames is not None and angles.shape[0] != frames:
        issues.append("rotation_angle length must match #views in detector/data")
    units = attr_to_str(angles.attrs.get("units"))
    try:
        angle_scale_degrees(units or "", path=path)
    except ValueError as exc:
        issues.append(str(exc))
    return issues
