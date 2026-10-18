from __future__ import annotations

from itertools import product
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import h5py
import numpy as np

from tomojax.geometry.api import (
    DISK_VOLUME_AXES,
    INTERNAL_VOLUME_AXES,
    VOLUME_AXES_ATTR,
    axes_to_perm,
    infer_disk_axes,
    nominal_axis_unit_from_inputs,
    transpose_volume,
)

from ._io_acquisition import ROTATION_ANGLE_PATHS, angle_scale_degrees
from ._io_frames import Hdf5Frames
from ._io_helpers import (
    _attr_to_str,
    _axes_log_warning,
    _detector_group,
    _ensure_group,
    _load_detector_metadata,
    _load_geometry_metadata,
    _load_grid_metadata,
    _load_image_key,
    _load_processing_metadata,
    _load_rotation_angles,
    _load_sample_metadata,
    _load_source_metadata,
    _write_string_attr,
)
from ._io_types import LoadedDataset, LoadedNXTomo, NXTomoMetadata, ValidationReport
from .geometry_meta import normalize_geometry_type

if TYPE_CHECKING:
    import jax


def save_nxtomo(
    path: str,
    projections: np.ndarray,
    *,
    metadata: NXTomoMetadata | None = None,
    compression: str = "lzf",
    overwrite: bool = True,
) -> None:
    """Write a dataset to an HDF5 file using NXtomo + TomoJAX extras.

    `metadata` owns the persistence contract: geometry, detector/grid definitions,
    optional reconstructed volume, and TomoJAX-specific extras. Volume inputs in
    `metadata.volume` are always expected in internal `xyz` order; set
    `metadata.volume_axes_order` to control how that internal volume is serialized
    on disk.

    Parameters
    ----------
    path : str
        NXtomo output file; parent directories are created.
    projections : numpy.ndarray
        Projection stack in (view, v, u) order.
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
    parent = Path(path).parent
    if str(parent) not in {"", "."}:
        parent.mkdir(parents=True, exist_ok=True)

    meta = metadata or NXTomoMetadata()
    mode = "w" if overwrite else "x"

    proj = np.asarray(projections)
    if proj.ndim != 3:
        raise ValueError("projections must be (n_views, nv, nu)")
    n_views, nv, nu = proj.shape
    geometry_type_norm = normalize_geometry_type(meta.geometry_type)

    disk_axes = meta.volume_axes_order.lower()
    _validate_volume_save_options(meta, disk_axes=disk_axes)

    with h5py.File(path, mode) as f:
        entry = _ensure_group(f, "entry", "NXentry")
        _write_string_attr(entry, "definition", "NXtomo")
        inst = _ensure_group(entry, "instrument", "NXinstrument")

        _write_geometry_section(entry, meta, geometry_type_norm)
        det = _write_detector_section(
            inst,
            meta,
            proj,
            n_views=n_views,
            nv=nv,
            nu=nu,
            compression=compression,
        )
        _write_sample_section(entry, meta, n_views=n_views)

        # NXdata linking for default plot (optional)
        data_grp = _ensure_group(entry, "data", "NXdata")
        # store projections also under /entry/data/projections for convenience
        data_grp["projections"] = det["data"]
        _write_string_attr(data_grp, "signal", "projections")

        _write_source_section(inst, meta)
        _write_grid_metadata(entry, meta)
        _write_volume_section(entry, meta, disk_axes=disk_axes, compression=compression)
        _write_tomojax_metadata_sections(entry, meta, compression=compression)


def _write_geometry_section(
    entry: h5py.Group,
    meta: NXTomoMetadata,
    geometry_type_norm: str,
) -> None:
    geom = _ensure_group(entry, "geometry", "NXcollection")
    _write_string_attr(geom, "type", geometry_type_norm)
    if meta.geometry_meta:
        geom.attrs["geometry_meta_json"] = json.dumps(meta.geometry_meta)


def _write_detector_section(
    inst: h5py.Group,
    meta: NXTomoMetadata,
    proj: np.ndarray,
    *,
    n_views: int,
    nv: int,
    nu: int,
    compression: str,
) -> h5py.Group:
    det = _ensure_group(inst, "detector", "NXdetector")
    det.create_dataset(
        "data",
        data=proj,
        chunks=(1, min(256, nv), min(256, nu)),
        compression=compression,
    )
    det["data"].attrs["long_name"] = "projections"
    if meta.image_key is None:
        im_key = np.zeros((n_views,), dtype=np.int32)
    else:
        im_key = np.asarray(meta.image_key, dtype=np.int32)
        if im_key.shape != (n_views,):
            raise ValueError("image_key must be shape (n_views,)")
    det.create_dataset("image_key", data=im_key, dtype=np.int32)

    if meta.detector is not None:
        det_dict = meta.detector if isinstance(meta.detector, dict) else meta.detector.to_dict()
        det.attrs["detector_meta_json"] = json.dumps(det_dict)
        det.create_dataset("x_pixel_size", data=np.asarray(det_dict["du"]))
        det["x_pixel_size"].attrs["units"] = "pixel"
        det.create_dataset("y_pixel_size", data=np.asarray(det_dict["dv"]))
        det["y_pixel_size"].attrs["units"] = "pixel"
    return det


def _write_sample_section(entry: h5py.Group, meta: NXTomoMetadata, *, n_views: int) -> None:
    sample = _ensure_group(entry, "sample", "NXsample")
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
    trans = _ensure_group(sample, "transformations", "NXtransformations")
    d_angle = trans.create_dataset("rotation_angle", data=angles)
    d_angle.attrs["units"] = "degree"
    _write_string_attr(d_angle, "transformation_type", "rotation")
    # The public NX fields must agree with the saved tilted/arbitrary-axis model.
    # If that model is incomplete, do not manufacture an axial direction.
    inputs = dict(meta.geometry_meta or {})
    gtype = normalize_geometry_type(meta.geometry_type)
    inputs["geometry_type"] = "lamino" if gtype == "cone" else gtype
    if gtype == "parallel" or "tilt_deg" in inputs or "axis_unit_lab" in inputs:
        axis = nominal_axis_unit_from_inputs(inputs)
        trans.create_dataset("rotation_axis", data=np.asarray(axis, dtype=np.float32))
    _write_string_attr(trans, "depends_on", "rotation_angle")


def _write_source_section(inst: h5py.Group, meta: NXTomoMetadata) -> None:
    src_grp = _ensure_group(inst, "SOURCE", "NXsource")
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
    disk_axes: str,
    compression: str,
) -> None:
    if meta.volume is None:
        return
    _validate_volume_save_options(meta, disk_axes=disk_axes)
    processing = _ensure_group(entry, "processing", "NXprocess")
    tj = _ensure_group(processing, "tomojax", "NXcollection")
    axes = axes_to_perm(INTERNAL_VOLUME_AXES, disk_axes)
    direct = isinstance(meta.volume, np.ndarray) and axes == (0, 1, 2)
    vol = tj.create_dataset(
        "volume",
        shape=tuple(meta.volume.shape[axis] for axis in axes),
        dtype=meta.volume.dtype,
        data=meta.volume if direct else None,
        chunks=True,
        compression=compression,
    )
    if not direct:
        _write_in_tiles(vol, meta.volume, axes)
    vol.attrs["long_name"] = "volume"
    _write_string_attr(tj, VOLUME_AXES_ATTR, disk_axes)
    if meta.frame is not None:
        _write_string_attr(tj, "frame", str(meta.frame))


# Host memory one tile of a saved volume may take.
_TILE_BYTES = 64 * 1024**2


def _write_in_tiles(
    target: h5py.Dataset, source: np.ndarray | jax.Array, axes: tuple[int, ...]
) -> None:
    """Write ``source`` transposed by ``axes`` in tiles of whole chunks, at most 64 MiB each.

    A device array or memmap is read, and a transposed volume reordered, a tile at a
    time, never as a second copy of the whole volume.
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
    starts = (range(0, n, step) for n, step in zip(target.shape, tile, strict=True))
    for start in product(*starts):
        region = tuple(
            slice(lo, min(lo + step, n))
            for lo, step, n in zip(start, tile, target.shape, strict=True)
        )
        # Disk axis k is source axis axes[k].
        part = source[tuple(region[axes.index(j)] for j in range(3))]
        target[region] = np.asarray(part).transpose(axes)


def _validate_volume_save_options(meta: NXTomoMetadata, *, disk_axes: str) -> None:
    if meta.volume is None:
        return
    if disk_axes == "unknown":
        raise ValueError(
            "Cannot save volume with unknown axis orientation. "
            "The loaded file lacks axis metadata. "
            "Set metadata.volume_axes_order before saving."
        )
    try:
        axes_to_perm(INTERNAL_VOLUME_AXES, disk_axes)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(
            "metadata.volume_axes_order must be a permutation of 'xyz', "
            f"got {meta.volume_axes_order!r}"
        ) from exc
    if meta.volume.ndim != 3:
        raise ValueError("volume must be 3D (nx, ny, nz)")


def _write_tomojax_metadata_sections(
    entry: h5py.Group,
    meta: NXTomoMetadata,
    *,
    compression: str,
) -> None:
    has_alignment_metadata = (
        meta.align_params is not None
        or meta.align_gauge is not None
        or meta.angle_offset_deg is not None
        or meta.misalign_spec is not None
    )
    if has_alignment_metadata or meta.geometry_calibration is not None:
        processing = _ensure_group(entry, "processing", "NXprocess")
        tj = _ensure_group(processing, "tomojax", "NXcollection")
        if has_alignment_metadata:
            _write_alignment_metadata_section(tj, meta, compression=compression)
        if meta.geometry_calibration is not None:
            calibration_grp = _ensure_group(tj, "calibration", "NXcollection")
            calibration_grp.attrs["geometry_calibration_json"] = json.dumps(
                meta.geometry_calibration
            )

    if meta.simulation_artefacts is not None:
        processing = _ensure_group(entry, "processing", "NXprocess")
        tj = _ensure_group(processing, "tomojax", "NXcollection")
        simulation_grp = _ensure_group(tj, "simulation", "NXcollection")
        simulation_grp.attrs["artefacts_json"] = json.dumps(meta.simulation_artefacts)


def _write_alignment_metadata_section(
    tj: h5py.Group,
    meta: NXTomoMetadata,
    *,
    compression: str,
) -> None:
    align_grp = _ensure_group(tj, "align", "NXcollection")
    if meta.align_params is not None:
        dset = align_grp.create_dataset(
            "thetas",
            data=np.asarray(meta.align_params, dtype=np.float32),
            chunks=True,
            compression=compression,
        )
        names = ["alpha", "beta", "phi", "dx", "dz", "dy"][: int(np.shape(meta.align_params)[1])]
        dset.attrs["columns"] = np.array(names, dtype=h5py.string_dtype())
    if meta.align_gauge is not None:
        align_grp.attrs["gauge_fix_json"] = json.dumps(meta.align_gauge)
    if meta.angle_offset_deg is not None:
        align_grp.create_dataset(
            "angle_offset_deg",
            data=np.asarray(meta.angle_offset_deg, dtype=np.float32),
            chunks=True,
            compression=compression,
        )
    if meta.misalign_spec is not None:
        align_grp.attrs["misalign_spec_json"] = json.dumps(meta.misalign_spec)


def load_nxtomo(path: str, *, lazy: bool = False) -> LoadedNXTomo:
    """Load an NXtomo dataset and TomoJAX extras.

    Returns a ``LoadedNXTomo`` payload with typed accessors for projections and
    persisted metadata. ``grid`` is present only when the file stores grid
    metadata; a saved volume does not establish physical voxels. Volumes are returned in
    internal ``xyz`` order. With ``lazy`` the projections are an
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
        Projection payload and metadata, with any saved volume in internal xyz order.

    Raises
    ------
    KeyError
        If required NXtomo groups or projection data are absent.
    ValueError
        If per-frame metadata or volume-axis conventions are invalid.
    OSError
        If the input cannot be opened.
    """
    out: LoadedDataset = {}
    volume_raw: np.ndarray | None = None
    volume_axes_attr: str | None = None

    with h5py.File(path, "r") as f:
        entry = f["/entry"]
        det_grp = _detector_group(entry)
        if det_grp is not None and "data" in det_grp:
            where = "/entry/instrument/detector/data"
        elif "data" in entry and "projections" in entry["data"]:
            where = "/entry/data/projections"
        elif "projections" in entry:
            where = "/entry/projections"
        else:
            raise KeyError("Could not find projections dataset under /entry")
        stack = Hdf5Frames.open(path, where)
        proj = stack if lazy else np.asarray(stack)
        out["projections"] = proj
        n_views = proj.shape[0]
        image_key = _load_image_key(entry, n_views=n_views, path=path)
        out["image_key"] = image_key
        out["angles"] = _load_rotation_angles(
            entry, n_views=n_views, path=path, image_key=image_key
        )
        _load_geometry_metadata(out, entry, path=path)
        _load_grid_metadata(out, entry, path=path)
        _load_detector_metadata(out, entry, path=path)
        _load_source_metadata(out, entry)
        volume_raw, volume_axes_attr = _load_processing_metadata(out, entry, path=path)
        _load_sample_metadata(out, entry)

        # Normalize volume axes if present
        if volume_raw is not None:
            grid_hint = out.get("grid")
            attr_norm = None
            if volume_axes_attr:
                try:
                    axes_to_perm(INTERNAL_VOLUME_AXES, volume_axes_attr)
                    attr_norm = volume_axes_attr.lower()
                except ValueError:
                    _axes_log_warning(
                        "Ignoring malformed volume_axes_order=%r on %s",
                        volume_axes_attr,
                        path,
                    )
            disk_axes = attr_norm or infer_disk_axes(volume_raw.shape, grid_hint)
            source = "attr" if attr_norm else "heuristic"
            volume_np = np.asarray(volume_raw)
            disk_order: str
            if disk_axes == DISK_VOLUME_AXES:
                if source == "heuristic":
                    _axes_log_warning(
                        "load_nxtomo: inferred disk volume axes zyx for %s; "
                        "transposing to internal xyz",
                        path,
                    )
                volume_np = np.asarray(
                    transpose_volume(volume_np, DISK_VOLUME_AXES, INTERNAL_VOLUME_AXES)
                )
                disk_order = DISK_VOLUME_AXES
            elif disk_axes == INTERNAL_VOLUME_AXES:
                if source == "heuristic":
                    _axes_log_warning(
                        "load_nxtomo: inferred disk volume axes xyz for %s",
                        path,
                    )
                disk_order = INTERNAL_VOLUME_AXES
            elif disk_axes is None:
                disk_order = "unknown"
                _axes_log_warning(
                    "load_nxtomo: unable to infer volume axis order for %s (shape %s)",
                    path,
                    volume_raw.shape,
                )
            else:
                volume_np = np.asarray(transpose_volume(volume_np, disk_axes, INTERNAL_VOLUME_AXES))
                disk_order = disk_axes
            out["volume"] = volume_np
            out["volume_axes_order"] = None
            out["disk_volume_axes_order"] = disk_order
            out["volume_axes_source"] = source
    return LoadedNXTomo.from_dataset(out)


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
            e = f["/entry"]
            definition = _attr_to_str(e.attrs.get("definition"))
            if definition != "NXtomo":
                report["issues"].append("entry/definition must be 'NXtomo'")
            # Required datasets
            if "instrument/detector/data" not in e:
                report["issues"].append("Missing instrument/detector/data")
            angle_path = next((p for p in ROTATION_ANGLE_PATHS if p in f), None)
            if angle_path is None:
                report["issues"].append("Missing sample rotation_angle")
            if "instrument/detector/image_key" not in e:
                report["issues"].append("Missing instrument/detector/image_key")
            if "sample" not in e or "name" not in e["sample"]:
                report["issues"].append("Missing sample/name")
            # Basic shapes
            n_views = None
            if "instrument/detector/data" in e:
                d = e["instrument/detector/data"]
                if d.ndim != 3:
                    report["issues"].append("instrument/detector/data must be 3D (n_views, nv, nu)")
                else:
                    n_views = d.shape[0]
            if "instrument/detector/image_key" in e:
                ik = e["instrument/detector/image_key"]
                if ik.ndim != 1:
                    report["issues"].append("instrument/detector/image_key must be 1D (n_views,)")
                elif n_views is not None and ik.shape[0] != n_views:
                    report["issues"].append("image_key length must match #views in detector/data")
                if ik.dtype.kind not in {"i", "u"}:
                    report["issues"].append("image_key must use integer dtype")
                elif ik.ndim == 1:
                    invalid_values, invalid_counts = np.unique(
                        ik[~np.isin(ik, np.asarray([0, 1, 2, 3], dtype=ik.dtype))],
                        return_counts=True,
                    )
                    if invalid_values.size > 0:
                        formatted = ", ".join(
                            f"{int(value)} ({int(count)} frame{'s' if int(count) != 1 else ''})"
                            for value, count in zip(invalid_values, invalid_counts, strict=True)
                        )
                        report["issues"].append(
                            "image_key values must be in {0, 1, 2, 3}; found " + formatted
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


def _rotation_angle_issues(angles: h5py.Dataset, *, frames: int | None, path: str) -> list[str]:
    """Shape and unit issues in a rotation-angle dataset."""
    issues = []
    if angles.ndim != 1:
        issues.append("rotation_angle must be 1D (n_views,)")
    elif frames is not None and angles.shape[0] != frames:
        issues.append("rotation_angle length must match #views in detector/data")
    units = _attr_to_str(angles.attrs.get("units"))
    try:
        angle_scale_degrees(units or "", path=path)
    except ValueError as exc:
        issues.append(str(exc))
    return issues
