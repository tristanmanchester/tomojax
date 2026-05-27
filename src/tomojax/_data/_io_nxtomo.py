from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from tomojax.geometry import (
    DISK_VOLUME_AXES,
    INTERNAL_VOLUME_AXES,
    VOLUME_AXES_ATTR,
    axes_to_perm,
    infer_disk_axes,
    transpose_volume,
)

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
    _normalize_geometry_type,
    _write_string_attr,
)
from ._io_types import LoadedDataset, LoadedNXTomo, NXTomoMetadata, ValidationReport


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
    geometry_type_norm = _normalize_geometry_type(meta.geometry_type)

    disk_axes = meta.volume_axes_order.lower()

    with h5py.File(path, mode) as f:
        entry = _ensure_group(f, "entry", "NXentry")
        _write_string_attr(entry, "definition", "NXtomo")

        # Geometry metadata (mark as NXcollection so common viewers display it)
        geom = _ensure_group(entry, "geometry", "NXcollection")
        _write_string_attr(geom, "type", geometry_type_norm)
        if meta.geometry_meta:
            geom.attrs["geometry_meta_json"] = json.dumps(meta.geometry_meta)

        # Instrument / detector
        inst = _ensure_group(entry, "instrument", "NXinstrument")
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

        # Pixel geometry if provided
        if meta.detector is not None:
            det_dict = meta.detector if isinstance(meta.detector, dict) else meta.detector.to_dict()
            det.attrs["detector_meta_json"] = json.dumps(det_dict)
            # Store basic pixel sizes (units arbitrary for sims)
            det.create_dataset("x_pixel_size", data=np.asarray(det_dict.get("du", 1.0)))
            det["x_pixel_size"].attrs["units"] = "pixel"
            det.create_dataset("y_pixel_size", data=np.asarray(det_dict.get("dv", 1.0)))
            det["y_pixel_size"].attrs["units"] = "pixel"

        sample = _ensure_group(entry, "sample", "NXsample")
        trans = _ensure_group(sample, "transformations", "NXtransformations")
        if meta.thetas_deg is None:
            thetas_deg = np.zeros((n_views,), dtype=np.float32)
        else:
            thetas_deg = np.asarray(meta.thetas_deg, dtype=np.float32)
            if thetas_deg.shape != (n_views,):
                raise ValueError("thetas_deg must be (n_views,)")
        d_angle = trans.create_dataset("rotation_angle", data=thetas_deg)
        d_angle.attrs["units"] = "degree"
        _write_string_attr(d_angle, "transformation_type", "rotation")
        # Rotation around +z by default (matching phi in current code)
        trans.create_dataset("rotation_axis", data=np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
        _write_string_attr(trans, "depends_on", "rotation_angle")
        sample_name_str = "sample" if meta.sample_name is None else str(meta.sample_name)
        sample.create_dataset(
            "name",
            data=np.array(sample_name_str, dtype=h5py.string_dtype(encoding="utf-8")),
        )

        # NXdata linking for default plot (optional)
        data_grp = _ensure_group(entry, "data", "NXdata")
        # store projections also under /entry/data/projections for convenience
        data_grp["projections"] = det["data"]
        _write_string_attr(data_grp, "signal", "projections")

        # Source metadata (optional but recommended)
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

        # Grid metadata
        if meta.grid is not None:
            gdict = meta.grid if isinstance(meta.grid, dict) else meta.grid.to_dict()
            entry.attrs["grid_meta_json"] = json.dumps(gdict)

        # Optional GT / reconstructed volume and TomoJAX metadata
        if meta.volume is not None:
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
            processing = _ensure_group(entry, "processing", "NXprocess")
            tj = _ensure_group(processing, "tomojax", "NXcollection")
            vol_data = np.asarray(meta.volume)
            if vol_data.ndim != 3:
                raise ValueError("volume must be 3D (nx, ny, nz)")
            if disk_axes != INTERNAL_VOLUME_AXES:
                vol_data = np.asarray(transpose_volume(vol_data, INTERNAL_VOLUME_AXES, disk_axes))
            vol = tj.create_dataset(
                "volume",
                data=vol_data,
                chunks=True,
                compression=compression,
            )
            vol.attrs["long_name"] = "ground_truth_volume"
            _write_string_attr(tj, VOLUME_AXES_ATTR, disk_axes)
            # Persist volume frame metadata if provided (e.g., 'sample' or 'lab')
            if meta.frame is not None:
                _write_string_attr(tj, "frame", str(meta.frame))

        # Optional alignment and geometry-calibration metadata
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
                align_grp = _ensure_group(tj, "align", "NXcollection")
                if meta.align_params is not None:
                    dset = align_grp.create_dataset(
                        "thetas",
                        data=np.asarray(meta.align_params, dtype=np.float32),
                        chunks=True,
                        compression=compression,
                    )
                    dset.attrs["columns"] = np.array(
                        ["alpha", "beta", "phi", "dx", "dz"], dtype=h5py.string_dtype()
                    )
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

            if meta.geometry_calibration is not None:
                calibration_grp = _ensure_group(tj, "calibration", "NXcollection")
                calibration_grp.attrs["geometry_calibration_json"] = json.dumps(
                    meta.geometry_calibration
                )

        # Optional simulation artefact metadata
        if meta.simulation_artefacts is not None:
            processing = _ensure_group(entry, "processing", "NXprocess")
            tj = _ensure_group(processing, "tomojax", "NXcollection")
            simulation_grp = _ensure_group(tj, "simulation", "NXcollection")
            simulation_grp.attrs["artefacts_json"] = json.dumps(meta.simulation_artefacts)


def load_nxtomo(path: str) -> LoadedNXTomo:
    """Load an NXtomo dataset and TomoJAX extras.

    Returns a ``LoadedNXTomo`` payload with typed accessors for projections and
    persisted metadata. ``grid`` is present when the file stores grid metadata
    or when a saved volume allows a unit-grid fallback. Volumes are returned in
    internal ``xyz`` order.
    """
    out: LoadedDataset = {}
    volume_raw: np.ndarray | None = None
    volume_axes_attr: str | None = None

    with h5py.File(path, "r") as f:
        entry = f["/entry"]
        proj = None
        det_grp = _detector_group(entry)
        if det_grp is not None and "data" in det_grp:
            proj = det_grp["data"][...]
        elif "data" in entry and "projections" in entry["data"]:
            proj = entry["data/projections"][...]
        elif "projections" in entry:
            proj = entry["projections"][...]
        if proj is None:
            raise KeyError("Could not find projections dataset under /entry")
        out["projections"] = proj
        n_views = proj.shape[0]
        out["image_key"] = _load_image_key(entry, n_views=n_views, path=path)
        out["thetas_deg"] = _load_rotation_angles(entry, n_views=n_views, path=path)
        _load_geometry_metadata(out, entry, path=path)
        _load_grid_metadata(out, entry, path=path)
        _load_detector_metadata(out, entry, proj, path=path)
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
    """Lightweight schema checks. Returns a report dict; empty `issues` means OK."""
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
            if "sample/transformations/rotation_angle" not in e:
                report["issues"].append("Missing sample/transformations/rotation_angle")
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
            if "sample/transformations/rotation_angle" in e:
                ang = e["sample/transformations/rotation_angle"]
                if ang.ndim != 1:
                    report["issues"].append("rotation_angle must be 1D (n_views,)")
                elif n_views is not None and ang.shape[0] != n_views:
                    report["issues"].append(
                        "rotation_angle length must match #views in detector/data"
                    )
                units = _attr_to_str(ang.attrs.get("units"))
                if units != "degree":
                    report["issues"].append("rotation_angle units attr should be 'degree'")
    except OSError as exc:
        report["issues"].append(f"Unable to read HDF5 file: {exc}")
    return report
