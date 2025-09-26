"""HDF5/NXtomo IO for TomoJAX v2.

Provides utilities to read/write datasets in HDF5 using the NeXus (NXtomo)
conventions with TomoJAX-specific extras under /entry/processing/tomojax.

This module focuses on accessibility at beamlines and interop with existing
pipelines, while keeping simulation-friendly metadata.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import h5py

from ..core.geometry.base import Grid, Detector
from ..utils.axes import (
    DISK_VOLUME_AXES,
    INTERNAL_VOLUME_AXES,
    VOLUME_AXES_ATTR,
    axes_to_perm,
    infer_disk_axes,
    transpose_volume,
)


LOG = logging.getLogger(__name__)
_AXES_SILENCE = os.environ.get("TOMOJAX_AXES_SILENCE", "").lower() in {"1", "true", "yes", "on"}


def _axes_log_warning(message: str, *args: Any) -> None:
    if not _AXES_SILENCE:
        LOG.warning(message, *args)


def _attr_to_str(v: Any, default: str | None = None) -> str | None:
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
    except Exception:
        return default


def _ensure_group(root: h5py.Group, name: str, nx_class: Optional[str] = None) -> h5py.Group:
    g = root.require_group(name)
    if nx_class:
        g.attrs["NX_class"] = nx_class
    return g


def _write_string_attr(obj: h5py.Group | h5py.Dataset, key: str, value: str) -> None:
    obj.attrs[key] = np.array(value, dtype=h5py.string_dtype(encoding="utf-8"))


def save_nxtomo(
    path: str,
    projections: np.ndarray,
    *,
    thetas_deg: Optional[np.ndarray] = None,
    grid: Optional[Grid | Dict[str, Any]] = None,
    detector: Optional[Detector | Dict[str, Any]] = None,
    geometry_type: str = "parallel",
    geometry_meta: Optional[Dict[str, Any]] = None,
    volume: Optional[np.ndarray] = None,
    align_params: Optional[np.ndarray] = None,
    angle_offset_deg: Optional[np.ndarray] = None,
    misalign_spec: Optional[Dict[str, Any]] = None,
    frame: Optional[str] = None,
    compression: str = "lzf",
    overwrite: bool = True,
    volume_axes_order: str = DISK_VOLUME_AXES,
) -> None:
    """Write a dataset to an HDF5 file using NXtomo + TomoJAX extras.

    Volume inputs are expected in internal `xyz` order. Set `volume_axes_order="xyz"`
    to bypass the transpose when interoperating with legacy datasets.

    - projections: (n_views, nv, nu) float32
    - thetas_deg: (n_views,) rotation angles in degrees (around +z by default)
    - grid/detector: voxel and detector definitions
    - volume: optional ground-truth volume (nx, ny, nz) in the declared frame
    - align_params: optional (n_views, 5) [alpha,beta,phi,dx,dz] in radians/world units
    - geometry_type: "parallel" | "lamino" | "custom"
    - geometry_meta: Optional dict with extra geometry metadata (e.g., lamino tilt)
    """
    # Ensure parent directory exists
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if overwrite:
        mode = "w"
    else:
        mode = "x"

    proj = np.asarray(projections)
    assert proj.ndim == 3, "projections must be (n_views, nv, nu)"
    n_views, nv, nu = proj.shape

    disk_axes = volume_axes_order.lower()
    try:
        axes_to_perm(INTERNAL_VOLUME_AXES, disk_axes)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"volume_axes_order must be a permutation of 'xyz', got {volume_axes_order!r}"
        ) from exc

    with h5py.File(path, mode) as f:
        entry = _ensure_group(f, "entry", "NXentry")
        _write_string_attr(entry, "definition", "NXtomo")

        # Geometry metadata (mark as NXcollection so common viewers display it)
        geom = _ensure_group(entry, "geometry", "NXcollection")
        _write_string_attr(geom, "type", geometry_type)
        if geometry_meta:
            geom.attrs["geometry_meta_json"] = json.dumps(geometry_meta)

        # Instrument / detector
        inst = _ensure_group(entry, "instrument", "NXinstrument")
        det = _ensure_group(inst, "detector", "NXdetector")
        det.create_dataset("data", data=proj, chunks=(1, min(256, nv), min(256, nu)), compression=compression)
        det["data"].attrs["long_name"] = "projections"

        # Pixel geometry if provided
        det_meta: Dict[str, Any] = {}
        if detector is not None:
            det_dict = detector if isinstance(detector, dict) else detector.to_dict()
            det_meta = det_dict
            det.attrs["detector_meta_json"] = json.dumps(det_dict)
            # Store basic pixel sizes (units arbitrary for sims)
            det.create_dataset("x_pixel_size", data=np.asarray(det_dict.get("du", 1.0)))
            det["x_pixel_size"].attrs["units"] = "pixel"
            det.create_dataset("y_pixel_size", data=np.asarray(det_dict.get("dv", 1.0)))
            det["y_pixel_size"].attrs["units"] = "pixel"

        # Sample and transformations (angles)
        sample = _ensure_group(entry, "sample", "NXsample")
        trans = _ensure_group(sample, "transformations", "NXtransformations")
        if thetas_deg is None:
            thetas_deg = np.zeros((n_views,), dtype=np.float32)
        else:
            thetas_deg = np.asarray(thetas_deg, dtype=np.float32)
            assert thetas_deg.shape == (n_views,), "thetas_deg must be (n_views,)"
        d_angle = trans.create_dataset("rotation_angle", data=thetas_deg)
        d_angle.attrs["units"] = "degree"
        _write_string_attr(d_angle, "transformation_type", "rotation")
        # Rotation around +z by default (matching phi in current code)
        trans.create_dataset("rotation_axis", data=np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
        _write_string_attr(trans, "depends_on", "rotation_angle")

        # NXdata linking for default plot (optional)
        data_grp = _ensure_group(entry, "data", "NXdata")
        # store projections also under /entry/data/projections for convenience
        data_grp["projections"] = det["data"]
        _write_string_attr(data_grp, "signal", "projections")

        # Grid metadata
        grid_meta: Dict[str, Any] = {}
        if grid is not None:
            gdict = grid if isinstance(grid, dict) else grid.to_dict()
            grid_meta = gdict
            entry.attrs["grid_meta_json"] = json.dumps(gdict)

        # Optional GT / reconstructed volume and TomoJAX metadata
        if volume is not None:
            processing = _ensure_group(entry, "processing", "NXprocess")
            tj = _ensure_group(processing, "tomojax", "NXcollection")
            vol_data = np.asarray(volume)
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
            if frame is not None:
                _write_string_attr(tj, "frame", str(frame))

        # Optional alignment params and misalignment metadata
        if align_params is not None or angle_offset_deg is not None or misalign_spec is not None:
            processing = _ensure_group(entry, "processing", "NXprocess")
            tj = _ensure_group(processing, "tomojax", "NXcollection")
            align_grp = _ensure_group(tj, "align", "NXcollection")
            if align_params is not None:
                dset = align_grp.create_dataset(
                    "thetas",
                    data=np.asarray(align_params, dtype=np.float32),
                    chunks=True,
                    compression=compression,
                )
                dset.attrs["columns"] = np.array(["alpha", "beta", "phi", "dx", "dz"], dtype=h5py.string_dtype())
            if angle_offset_deg is not None:
                align_grp.create_dataset(
                    "angle_offset_deg",
                    data=np.asarray(angle_offset_deg, dtype=np.float32),
                    chunks=True,
                    compression=compression,
                )
            if misalign_spec is not None:
                align_grp.attrs["misalign_spec_json"] = json.dumps(misalign_spec)


def load_nxtomo(path: str) -> Dict[str, Any]:
    """Load an NXtomo dataset and TomoJAX extras.

    Returns a dict with keys: projections, thetas_deg, geometry_type, grid, detector,
    and optional volume, align_params. Volumes are returned in internal `xyz` order.
    """
    out: Dict[str, Any] = {}
    volume_raw: Optional[np.ndarray] = None
    volume_axes_attr: Optional[str] = None

    with h5py.File(path, "r") as f:
        entry = f["/entry"]
        # Projections with fallbacks
        proj = None
        try:
            proj = entry["instrument/detector/data"][...]
        except Exception:
            try:
                proj = entry["data/projections"][...]
            except Exception:
                if "projections" in entry:
                    proj = entry["projections"][...]
        if proj is None:
            raise KeyError("Could not find projections dataset under /entry")
        out["projections"] = proj
        n_views = proj.shape[0]
        # Angles with fallback
        thetas = None
        try:
            thetas = entry["sample/transformations/rotation_angle"][...]
        except Exception:
            thetas = np.zeros((n_views,), dtype=np.float32)
        out["thetas_deg"] = thetas
        # Geometry type
        geom = entry.get("geometry")
        geom_type = "parallel"
        if geom is not None and "type" in geom.attrs:
            s = _attr_to_str(geom.attrs.get("type"), default="parallel")
            if s:
                geom_type = s
        out["geometry_type"] = geom_type
        if geom is not None:
            meta_attr = geom.attrs.get("geometry_meta_json")
            if meta_attr is not None:
                s = _attr_to_str(meta_attr)
                if s:
                    try:
                        meta_dict = json.loads(s)
                    except Exception:
                        meta_dict = None
                    if isinstance(meta_dict, dict):
                        out["geometry_meta"] = meta_dict
                        for key, val in meta_dict.items():
                            out.setdefault(key, val)

        # Grid / Detector metadata (JSON blobs if present)
        grid_meta = entry.attrs.get("grid_meta_json")
        if grid_meta is not None:
            s = _attr_to_str(grid_meta)
            if s:
                out["grid"] = json.loads(s)

        # Detector metadata (fallback to defaults if missing)
        try:
            det_grp = entry["instrument/detector"]
            det_meta = det_grp.attrs.get("detector_meta_json")
            if det_meta is not None:
                s = _attr_to_str(det_meta)
                if s:
                    out["detector"] = json.loads(s)
        except Exception:
            pass
        if "detector" not in out:
            nv, nu = int(proj.shape[1]), int(proj.shape[2])
            out["detector"] = {
                "nu": nu,
                "nv": nv,
                "du": 1.0,
                "dv": 1.0,
                "det_center": [0.0, 0.0],
            }

        # Optional volume (used also to infer grid fallback)
        try:
            if "processing" in entry and "tomojax" in entry["processing"]:
                tj = entry["processing/tomojax"]
                volume_axes_attr = _attr_to_str(tj.attrs.get(VOLUME_AXES_ATTR))
                if "volume" in tj:
                    volume_raw = tj["volume"][...]
                # Frame metadata for the volume if present
                fr_attr = tj.attrs.get("frame")
                if fr_attr is not None:
                    s = _attr_to_str(fr_attr)
                    if s:
                        out["frame"] = s
                if "align" in tj:
                    if "thetas" in tj["align"]:
                        out["align_params"] = tj["align/thetas"][...]
                    if "angle_offset_deg" in tj["align"]:
                        out["angle_offset_deg"] = tj["align/angle_offset_deg"][...]
                    spec_attr = tj["align"].attrs.get("misalign_spec_json")
                    if spec_attr is not None:
                        s = _attr_to_str(spec_attr)
                        if s:
                            try:
                                out["misalign_spec"] = json.loads(s)
                            except Exception:
                                pass
        except Exception:
            pass

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
                        "Ignoring malformed volume_axes_order=%r on %s", volume_axes_attr, path
                    )
            disk_axes = attr_norm or infer_disk_axes(volume_raw.shape, grid_hint)
            source = "attr" if attr_norm else "heuristic"
            volume_np = np.asarray(volume_raw)
            disk_order: str
            if disk_axes == DISK_VOLUME_AXES:
                if source == "heuristic":
                    _axes_log_warning(
                        "load_nxtomo: inferred disk volume axes zyx for %s; transposing to internal xyz",
                        path,
                    )
                volume_np = np.asarray(transpose_volume(volume_np, DISK_VOLUME_AXES, INTERNAL_VOLUME_AXES))
                disk_order = DISK_VOLUME_AXES
            elif disk_axes == INTERNAL_VOLUME_AXES:
                if source == "heuristic":
                    _axes_log_warning("load_nxtomo: assuming legacy xyz disk volume order for %s", path)
                disk_order = INTERNAL_VOLUME_AXES if source == "attr" else "xyz_legacy"
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
            out["volume_axes_order"] = INTERNAL_VOLUME_AXES
            out["disk_volume_axes_order"] = disk_order
            out["volume_axes_source"] = source

        # Grid fallback from volume if grid meta missing
        if "grid" not in out and volume_raw is not None and "volume" in out:
            vol = out["volume"]
            nx, ny, nz = int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])
            out["grid"] = {"nx": nx, "ny": ny, "nz": nz, "vx": 1.0, "vy": 1.0, "vz": 1.0}
    return out


def validate_nxtomo(path: str) -> Dict[str, Any]:
    """Lightweight schema checks. Returns a report dict; empty `issues` means OK."""
    report: Dict[str, Any] = {"issues": []}
    try:
        with h5py.File(path, "r") as f:
            if "/entry" not in f:
                report["issues"].append("Missing /entry")
                return report
            e = f["/entry"]
            # Required datasets
            if "instrument/detector/data" not in e:
                report["issues"].append("Missing instrument/detector/data")
            if "sample/transformations/rotation_angle" not in e:
                report["issues"].append("Missing sample/transformations/rotation_angle")
            # Basic shapes
            if "instrument/detector/data" in e:
                d = e["instrument/detector/data"]
                if d.ndim != 3:
                    report["issues"].append("projections must be 3D (n_views,nv,nu)")
    except Exception as exc:  # pragma: no cover (defensive)
        report["issues"].append(f"Exception during validation: {exc}")
    return report


def save_npz(path: str, projections: np.ndarray, **meta: Any) -> None:
    """Simple NPZ saver for tiny tests or interop."""
    np.savez_compressed(path, projections=projections, **meta)


def load_npz(path: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def convert(in_path: str, out_path: str) -> None:
    """Convert between .npz and .nxs based on file extension."""
    if in_path.endswith(".npz") and out_path.endswith((".nxs", ".h5", ".hdf5")):
        data = load_npz(in_path)
        projections = data["projections"]
        thetas_deg = data.get("thetas_deg")
        grid = data.get("grid")
        detector = data.get("detector")
        volume = data.get("volume")
        align_params = data.get("align_params")
        save_nxtomo(
            out_path,
            projections,
            thetas_deg=thetas_deg,
            grid=grid,
            detector=detector,
            volume=volume,
            align_params=align_params,
        )
    elif in_path.endswith((".nxs", ".h5", ".hdf5")) and out_path.endswith(".npz"):
        data = load_nxtomo(in_path)
        save_npz(out_path, **data)
    else:
        raise ValueError("Unsupported conversion. Use .npz <-> .nxs/.h5/.hdf5")
