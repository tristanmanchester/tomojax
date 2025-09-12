"""HDF5/NXtomo IO for TomoJAX v2.

Provides utilities to read/write datasets in HDF5 using the NeXus (NXtomo)
conventions with TomoJAX-specific extras under /entry/processing/tomojax.

This module focuses on accessibility at beamlines and interop with existing
pipelines, while keeping simulation-friendly metadata.
"""

from __future__ import annotations

import json
from dataclasses import asdict
import os
from typing import Any, Dict, Optional

import numpy as np
import h5py

from ..core.geometry.base import Grid, Detector


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
    volume: Optional[np.ndarray] = None,
    align_params: Optional[np.ndarray] = None,
    compression: str = "lzf",
    overwrite: bool = True,
) -> None:
    """Write a dataset to an HDF5 file using NXtomo + TomoJAX extras.

    - projections: (n_views, nv, nu) float32
    - thetas_deg: (n_views,) rotation angles in degrees (around +z by default)
    - grid/detector: voxel and detector definitions
    - volume: optional ground-truth volume (nz, ny, nx)
    - align_params: optional (n_views, 5) [alpha,beta,phi,dx,dz] in radians/world units
    - geometry_type: "parallel" | "lamino" | "custom"
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

    with h5py.File(path, mode) as f:
        entry = _ensure_group(f, "entry", "NXentry")
        _write_string_attr(entry, "definition", "NXtomo")

        # Geometry metadata
        geom = entry.require_group("geometry")
        _write_string_attr(geom, "type", geometry_type)

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

        # Optional GT volume
        if volume is not None:
            processing = _ensure_group(entry, "processing", "NXprocess")
            tj = _ensure_group(processing, "tomojax", "NXcollection")
            vol = tj.create_dataset(
                "volume",
                data=np.asarray(volume),
                chunks=True,
                compression=compression,
            )
            vol.attrs["long_name"] = "ground_truth_volume"

        # Optional alignment params
        if align_params is not None:
            processing = _ensure_group(entry, "processing", "NXprocess")
            tj = _ensure_group(processing, "tomojax", "NXcollection")
            align_grp = _ensure_group(tj, "align", "NXcollection")
            dset = align_grp.create_dataset(
                "thetas",
                data=np.asarray(align_params, dtype=np.float32),
                chunks=True,
                compression=compression,
            )
            dset.attrs["columns"] = np.array(["alpha", "beta", "phi", "dx", "dz"], dtype=h5py.string_dtype())


def load_nxtomo(path: str) -> Dict[str, Any]:
    """Load an NXtomo dataset and TomoJAX extras.

    Returns a dict with keys: projections, thetas_deg, geometry_type, grid, detector,
    and optional volume, align_params.
    """
    out: Dict[str, Any] = {}
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
                if "volume" in tj:
                    out["volume"] = tj["volume"][...]
                if "align" in tj and "thetas" in tj["align"]:
                    out["align_params"] = tj["align/thetas"][...]
        except Exception:
            pass

        # Grid fallback from volume if grid meta missing
        if "grid" not in out and "volume" in out:
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
