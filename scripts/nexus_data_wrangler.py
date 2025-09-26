#!/usr/bin/env python3
"""
Preprocess laminography data: flat/dark correction -> absorption (-log)
and save to a NeXus-compatible HDF5 file matching the requested layout.

Output structure:
  /entry (NXentry)
    @definition = "NXtomo"
    @grid_meta_json = {"nx": ..., "ny": ..., "nz": ..., "vx": ..., "vy": ..., "vz": ...}
    geometry (NXcollection)
      @type = "lamino"
      @geometry_meta_json = {"tilt_deg": ..., "tilt_about": "..."}
    instrument (NXinstrument)
      detector (NXdetector)
        data  (float32, [n_proj, ny, nx], chunks [1, ny, nx], compression=lzf, @long_name="projections")
        x_pixel_size (scalar, units="pixel")
        y_pixel_size (scalar, units="pixel")
        @detector_meta_json = {"nu": nx, "nv": ny, "du": 1.0, "dv": 1.0, "det_center": [0.0, 0.0]}
    sample (NXsample)
      transformations (NXtransformations)
        depends_on = "/entry/sample/transformations/rotation_angle"
        rotation_angle (float32, [n_proj], units="degree", @summary={start, step, count, endpoint})
        rotation_axis  (float32, [3])  # keep [0,0,1] for compatibility
    data (NXdata)
      @signal = "projections"
      projections -> hard link to /entry/instrument/detector/data
    processing (NXprocess)
      tomojax (NXcollection)
        @frame = "sample"
        @volume_axes_order = "zyx"
        volume (float32, [nz, ny, nx], chunks [16,16,32], compression=lzf, @long_name="ground_truth_volume")
"""

import json
import numpy as np
import h5py
from pathlib import Path

from tomojax.utils.axes import DISK_VOLUME_AXES, VOLUME_AXES_ATTR


def load_raw(input_path, proj_path, ang_path, key_path):
    with h5py.File(input_path, "r") as f:
        data = f[proj_path][...]            # shape: [N, ny, nx]
        angles_all = f[ang_path][...]       # shape: [N] or [>=N]
        image_key = f[key_path][...]        # 0=proj, 1=flat, 2=dark
    return data, angles_all, image_key


def flat_dark_correct_to_absorption(data, image_key, min_intensity=1e-6):
    """
    data: [N, ny, nx] raw detector counts
    image_key: [N] with 0=projection, 1=flat, 2=dark
    returns: absorption_projs [P, ny, nx]
    """
    is_proj = (image_key == 0)
    is_flat = (image_key == 1)
    is_dark = (image_key == 2)

    proj = data[is_proj]
    flats = data[is_flat]
    darks = data[is_dark]

    if flats.size == 0:
        raise RuntimeError("No flat fields found (image_key==1). Cannot normalise.")
    if darks.size == 0:
        # Allow missing darks by using zeros, but warn user
        print("WARNING: No dark fields found (image_key==2). Using zeros for dark correction.")
        dark_avg = 0.0
    else:
        dark_avg = np.mean(darks.astype(np.float32), axis=0)

    flat_avg = np.mean(flats.astype(np.float32), axis=0)

    proj = proj.astype(np.float32)
    # Normalise: (I - D) / (F - D)
    denom = flat_avg - dark_avg
    denom = np.maximum(denom, min_intensity)
    norm = (proj - dark_avg) / denom

    # Clip and convert to absorption contrast: -log(norm)
    norm = np.clip(norm, min_intensity, None)
    absorption = -np.log(norm, dtype=np.float32)

    # Clean up any residual NaNs/Infs
    absorption[~np.isfinite(absorption)] = 0.0
    return absorption


def _spatial_bin(arr: np.ndarray, bin_y: int = 1, bin_x: int = 1) -> np.ndarray:
    """Bin last two spatial dims by (bin_y, bin_x) using mean. Crops remainders.

    - If `arr.ndim == 3`, expects shape [N, ny, nx].
    - If `arr.ndim == 2`, expects shape [ny, nx].
    """
    bin_y = int(max(1, bin_y))
    bin_x = int(max(1, bin_x))
    if bin_y == 1 and bin_x == 1:
        return arr.astype(np.float32, copy=False)

    if arr.ndim == 3:
        n, ny, nx = arr.shape
        ny_c = (ny // bin_y) * bin_y
        nx_c = (nx // bin_x) * bin_x
        if ny_c != ny or nx_c != nx:
            arr = arr[:, :ny_c, :nx_c]
        arr = arr.reshape(n, ny_c // bin_y, bin_y, nx_c // bin_x, bin_x)
        return arr.mean(axis=(2, 4), dtype=np.float32)
    elif arr.ndim == 2:
        ny, nx = arr.shape
        ny_c = (ny // bin_y) * bin_y
        nx_c = (nx // bin_x) * bin_x
        if ny_c != ny or nx_c != nx:
            arr = arr[:ny_c, :nx_c]
        arr = arr.reshape(ny_c // bin_y, bin_y, nx_c // bin_x, bin_x)
        return arr.mean(axis=(1, 3), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported array ndim for binning: {arr.ndim}")


def _pad_to_multiples(arr: np.ndarray, mult_y: int | None, mult_x: int | None, mode: str = "edge") -> np.ndarray:
    """Symmetrically pad last two spatial dims to be multiples of (mult_y, mult_x).

    - If multiplier is None or <=1, no padding is applied on that axis.
    - Uses np.pad with given mode (default 'edge' to avoid introducing zeros in absorption).
    """
    if mult_y is None and mult_x is None:
        return arr
    if arr.ndim not in (2, 3):
        raise ValueError("Padding expects 2D or 3D arrays")

    ny = arr.shape[-2]
    nx = arr.shape[-1]
    target_ny = ny
    target_nx = nx
    if mult_y and int(mult_y) > 1:
        m = int(mult_y)
        r = target_ny % m
        if r != 0:
            target_ny += (m - r)
    if mult_x and int(mult_x) > 1:
        m = int(mult_x)
        r = target_nx % m
        if r != 0:
            target_nx += (m - r)
    pad_y = max(0, target_ny - ny)
    pad_x = max(0, target_nx - nx)
    if pad_y == 0 and pad_x == 0:
        return arr
    py0 = pad_y // 2
    py1 = pad_y - py0
    px0 = pad_x // 2
    px1 = pad_x - px0
    if arr.ndim == 3:
        pad_width = [(0, 0), (py0, py1), (px0, px1)]
    else:
        pad_width = [(py0, py1), (px0, px1)]
    return np.pad(arr, pad_width, mode=mode)


def summarize_angles(angles_deg):
    """Return a small summary dict like in the example."""
    out = {
        "start_deg": float(angles_deg[0]) if angles_deg.size > 0 else 0.0,
        "count": int(angles_deg.size),
        "endpoint": False,
    }
    if angles_deg.size >= 2:
        steps = np.diff(angles_deg)
        # Use median step; declare uniform if spread is tiny
        step = float(np.median(steps))
        out["step_deg"] = step
    else:
        out["step_deg"] = 0.0
    return out


def write_nexus_h5(
    output_path,
    projections,
    angles_deg,
    pixel_size_pixels_x=1.0,
    pixel_size_pixels_y=1.0,
    tilt_deg=35.0,
    tilt_about="x",
    grid=(128, 128, 128),
    voxels=(1.0, 1.0, 1.0),
):
    """
    projections: float32 [P, ny, nx] absorption
    angles_deg: float32 [P]
    grid: (nx, ny, nz) for the 'processing/tomojax/volume' placeholder and grid_meta_json
    voxels: (vx, vy, vz) physical voxel sizes in same units as pixel_size_pixels_*
    """
    P, ny, nx = projections.shape
    nx_grid, ny_grid, nz_grid = grid
    vx, vy, vz = voxels

    # Chunks: [1, ny, nx] for fast angle slicing
    proj_chunks = (1, ny, nx)

    with h5py.File(output_path, "w") as f:
        # /entry
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["definition"] = "NXtomo"
        entry.attrs["grid_meta_json"] = json.dumps(
            {"nx": int(nx_grid), "ny": int(ny_grid), "nz": int(nz_grid),
             "vx": float(vx), "vy": float(vy), "vz": float(vz)}
        )

        # /entry/geometry (NXcollection)
        geom = entry.create_group("geometry")
        geom.attrs["NX_class"] = "NXcollection"
        geom.attrs["type"] = "lamino"
        geom.attrs["geometry_meta_json"] = json.dumps(
            {"tilt_deg": float(tilt_deg), "tilt_about": str(tilt_about)}
        )

        # /entry/instrument/detector
        instr = entry.create_group("instrument")
        instr.attrs["NX_class"] = "NXinstrument"
        det = instr.create_group("detector")
        det.attrs["NX_class"] = "NXdetector"

        dset = det.create_dataset(
            "data",
            data=projections.astype(np.float32),
            dtype="float32",
            chunks=proj_chunks,
            compression="lzf",
            shuffle=False,
            fletcher32=False,
        )
        dset.attrs["long_name"] = "projections"

        # Pixel sizes (kept as 'pixel' units per your example)
        xps = det.create_dataset("x_pixel_size", data=np.array([pixel_size_pixels_x], dtype=np.float32))
        xps.attrs["units"] = "pixel"
        yps = det.create_dataset("y_pixel_size", data=np.array([pixel_size_pixels_y], dtype=np.float32))
        yps.attrs["units"] = "pixel"

        det.attrs["detector_meta_json"] = json.dumps(
            {"nu": int(nx), "nv": int(ny), "du": float(pixel_size_pixels_x), "dv": float(pixel_size_pixels_y), "det_center": [0.0, 0.0]}
        )

        # /entry/sample/transformations
        sample = entry.create_group("sample")
        sample.attrs["NX_class"] = "NXsample"
        trans = sample.create_group("transformations")
        trans.attrs["NX_class"] = "NXtransformations"

        # NeXus expects depends_on as an attribute pointing to the top transform
        trans.attrs["depends_on"] = "rotation_angle"

        ang = trans.create_dataset("rotation_angle", data=angles_deg.astype(np.float32))
        ang.attrs["units"] = "degree"
        ang.attrs["summary"] = json.dumps(summarize_angles(angles_deg))

        axis = trans.create_dataset("rotation_axis", data=np.array([0.0, 0.0, 1.0], dtype=np.float32))

        # /entry/data (NXdata) with hard link to detector data
        nxdata = entry.create_group("data")
        nxdata.attrs["NX_class"] = "NXdata"
        nxdata.attrs["signal"] = "projections"
        # Hard-link the projections dataset
        nxdata["projections"] = dset

        # /entry/processing/tomojax (NXcollection) with placeholder volume
        proc = entry.create_group("processing")
        proc.attrs["NX_class"] = "NXprocess"
        tj = proc.create_group("tomojax")
        tj.attrs["NX_class"] = "NXcollection"
        tj.attrs["frame"] = "sample"
        tj.attrs[VOLUME_AXES_ATTR] = np.array(DISK_VOLUME_AXES, dtype=h5py.string_dtype(encoding="utf-8"))

        vol = tj.create_dataset(
            "volume",
            shape=(int(nz_grid), int(ny_grid), int(nx_grid)),
            dtype="float32",
            chunks=(16, 16, 32),
            compression="lzf",
            shuffle=False,
            fletcher32=False,
        )
        vol[...] = 0.0  # lightweight (128^3 ~ 8 MB as float32)
        vol.attrs["long_name"] = "ground_truth_volume"

    print(f"Wrote corrected absorption data to: {output_path}")


def main():
    import argparse

    p = argparse.ArgumentParser(description="Flat/dark correct a NeXus file and export TomoJAX-compatible NXtomo.")
    p.epilog = "Volumes are saved on disk in (nz, ny, nx) order with @volume_axes_order="zyx"; TomoJAX APIs always return (nx, ny, nz)."
    p.add_argument("--in", dest="input_path", required=True, help="Input .nxs path")
    p.add_argument("--out", dest="output_path", required=True, help="Output .nxs path")
    p.add_argument("--proj-path", default="/entry/imaging/data", help="H5 path to raw frames [N, ny, nx]")
    p.add_argument("--angles-path", default="/entry/imaging_sum/smaract_zrot_value_set", help="H5 path to rotation angles [N]")
    p.add_argument("--image-key-path", default="/entry/instrument/EtherCAT/image_key", help="H5 path to image_key [N] with 0=proj,1=flat,2=dark")
    p.add_argument("--tilt-deg", type=float, default=35.0, help="Laminography tilt in degrees")
    p.add_argument("--tilt-about", choices=["x", "z"], default="x", help="Axis about which the tilt is applied")
    p.add_argument("--pixel-size", type=float, default=1.0, help="Base detector pixel size (kept as 'pixel' units)")
    p.add_argument("--bin", type=int, default=1, help="Spatial binning factor applied to both y and x")
    p.add_argument("--bin-y", type=int, default=None, help="Override binning factor along y (rows)")
    p.add_argument("--bin-x", type=int, default=None, help="Override binning factor along x (cols)")
    p.add_argument("--grid", type=int, nargs=3, metavar=("NX","NY","NZ"), default=None, help="Override volume grid size (nx ny nz). Default: derive from detector (nu,nu,nv)")
    p.add_argument("--no-grid-from-det", dest="grid_from_det", action="store_false", default=True, help="Disable deriving grid dims from detector (nu,nu,nv)")
    p.add_argument("--voxels", type=float, nargs=3, metavar=("VX","VY","VZ"), default=None, help="Override voxel sizes for metadata. Default: (du,du,dv)")
    p.add_argument("--min-intensity", type=float, default=1e-6, help="Clamp for normalisation denominator and log")
    p.add_argument("--pad-y-multiple", type=int, default=None, help="Pad rows to nearest multiple (after binning)")
    p.add_argument("--pad-x-multiple", type=int, default=None, help="Pad cols to nearest multiple (after binning)")
    p.add_argument("--pad-mode", choices=["edge","constant","reflect"], default="edge", help="Padding mode (default: edge)")
    args = p.parse_args()

    # ---- LOAD ----
    data, angles_all, image_key = load_raw(
        args.input_path,
        args.proj_path,
        args.angles_path,
        args.image_key_path,
    )
    is_proj = (image_key == 0)
    angles = angles_all[is_proj].astype(np.float32)

    # ---- CORRECT -> ABSORPTION ----
    absorption = flat_dark_correct_to_absorption(
        data=data, image_key=image_key, min_intensity=float(args.min_intensity)
    )

    # ---- OPTIONAL SPATIAL BINNING ----
    by = int(args.bin_y) if args.bin_y is not None else int(args.bin)
    bx = int(args.bin_x) if args.bin_x is not None else int(args.bin)
    if by > 1 or bx > 1:
        absorption = _spatial_bin(absorption, by, bx)
        print(f"Applied spatial binning: by={by}, bx={bx} -> new shape {absorption.shape}")

    # ---- OPTIONAL PADDING TO MULTIPLES (post-binning) ----
    if args.pad_y_multiple or args.pad_x_multiple:
        before = absorption.shape
        absorption = _pad_to_multiples(absorption, args.pad_y_multiple, args.pad_x_multiple, mode=str(args.pad_mode))
        after = absorption.shape
        if after != before:
            print(f"Padded to multiples (y={args.pad_y_multiple}, x={args.pad_x_multiple}): {before} -> {after}")

    # Sanity: angles length must match #projections
    if absorption.shape[0] != angles.shape[0]:
        raise RuntimeError(
            f"Projection count mismatch: absorption {absorption.shape[0]} vs angles {angles.shape[0]}"
        )

    # ---- WRITE NX/H5 ----
    # Choose grid dims: default to detector-based (nu, nu, nv)
    _, ny_pix, nx_pix = absorption.shape
    if args.grid is not None:
        grid = tuple(map(int, args.grid))
    elif args.grid_from_det:
        grid = (int(nx_pix), int(nx_pix), int(ny_pix))  # (nx, ny, nz)
    else:
        grid = (128, 128, 128)

    # Choose voxel sizes: default to detector pixel sizes (du,du,dv)
    if args.voxels is not None:
        vox = tuple(map(float, args.voxels))
    else:
        vox = (float(args.pixel_size) * float(bx), float(args.pixel_size) * float(bx), float(args.pixel_size) * float(by))

    write_nexus_h5(
        output_path=args.output_path,
        projections=absorption,
        angles_deg=angles,
        pixel_size_pixels_x=float(args.pixel_size) * float(bx),
        pixel_size_pixels_y=float(args.pixel_size) * float(by),
        tilt_deg=float(args.tilt_deg),
        tilt_about=str(args.tilt_about),
        grid=grid,
        voxels=vox,
    )


if __name__ == "__main__":
    main()
