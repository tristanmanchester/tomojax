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

from dataclasses import dataclass
import json

import h5py
import numpy as np

from tomojax.geometry import DISK_VOLUME_AXES, VOLUME_AXES_ATTR
from tomojax.io import (
    flat_dark_correct_frames_to_absorption,
    pad_to_multiples as _pad_to_multiples,
    spatial_bin as _spatial_bin,
    summarize_angles,
    volume_chunks as _volume_chunks,
)


def load_raw(input_path, proj_path, ang_path, key_path):
    with h5py.File(input_path, "r") as f:
        data = f[proj_path][...]            # shape: [N, ny, nx]
        angles_all = f[ang_path][...]       # shape: [N] or [>=N]
        image_key = f[key_path][...]        # 0=proj, 1=flat, 2=dark
    return data, angles_all, image_key


def flat_dark_correct_to_absorption(
    data,
    image_key,
    min_intensity=1e-6,
    assume_dark_field: float | None = None,
):
    """
    data: [N, ny, nx] raw detector counts
    image_key: [N] with 0=projection, 1=flat, 2=dark
    returns: absorption_projs [P, ny, nx]
    """
    return flat_dark_correct_frames_to_absorption(
        data=np.asarray(data),
        image_key=np.asarray(image_key),
        min_intensity=float(min_intensity),
        assume_dark_field=assume_dark_field,
    )


def _write_entry_metadata(f, *, grid, voxels, tilt_deg, tilt_about):
    nx_grid, ny_grid, nz_grid = grid
    vx, vy, vz = voxels
    entry = f.create_group("entry")
    entry.attrs["NX_class"] = "NXentry"
    entry.attrs["definition"] = "NXtomo"
    entry.attrs["grid_meta_json"] = json.dumps(
        {
            "nx": int(nx_grid),
            "ny": int(ny_grid),
            "nz": int(nz_grid),
            "vx": float(vx),
            "vy": float(vy),
            "vz": float(vz),
        }
    )

    geom = entry.create_group("geometry")
    geom.attrs["NX_class"] = "NXcollection"
    geom.attrs["type"] = "lamino"
    geom.attrs["geometry_meta_json"] = json.dumps(
        {"tilt_deg": float(tilt_deg), "tilt_about": str(tilt_about)}
    )
    return entry


def _write_detector(entry, *, projections, image_key_arr, pixel_size_pixels_x, pixel_size_pixels_y):
    _, ny, nx = projections.shape
    instr = entry.create_group("instrument")
    instr.attrs["NX_class"] = "NXinstrument"
    det = instr.create_group("detector")
    det.attrs["NX_class"] = "NXdetector"

    dset = det.create_dataset(
        "data",
        data=projections.astype(np.float32),
        dtype="float32",
        chunks=(1, ny, nx),
        compression="lzf",
        shuffle=False,
        fletcher32=False,
    )
    dset.attrs["long_name"] = "projections"
    det.create_dataset("image_key", data=image_key_arr.astype(np.int32))

    xps = det.create_dataset(
        "x_pixel_size",
        data=np.array([pixel_size_pixels_x], dtype=np.float32),
    )
    xps.attrs["units"] = "pixel"
    yps = det.create_dataset(
        "y_pixel_size",
        data=np.array([pixel_size_pixels_y], dtype=np.float32),
    )
    yps.attrs["units"] = "pixel"

    det.attrs["detector_meta_json"] = json.dumps(
        {
            "nu": int(nx),
            "nv": int(ny),
            "du": float(pixel_size_pixels_x),
            "dv": float(pixel_size_pixels_y),
            "det_center": [0.0, 0.0],
        }
    )
    return instr, dset


def _write_source(instr, *, source_name, source_type, source_probe):
    src = instr.create_group("SOURCE")
    src.attrs["NX_class"] = "NXsource"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    if source_name is not None:
        src.create_dataset("name", data=np.array(str(source_name), dtype=string_dtype))
    if source_type is not None:
        src.create_dataset("type", data=np.array(str(source_type), dtype=string_dtype))
    if source_probe is not None:
        src.create_dataset("probe", data=np.array(str(source_probe), dtype=string_dtype))


def _write_sample_transforms(entry, *, sample_name, angles_deg):
    sample = entry.create_group("sample")
    sample.attrs["NX_class"] = "NXsample"
    sample.create_dataset(
        "name",
        data=np.array(str(sample_name), dtype=h5py.string_dtype(encoding="utf-8")),
    )
    trans = sample.create_group("transformations")
    trans.attrs["NX_class"] = "NXtransformations"
    trans.attrs["depends_on"] = "rotation_angle"

    ang = trans.create_dataset("rotation_angle", data=angles_deg.astype(np.float32))
    ang.attrs["units"] = "degree"
    ang.attrs["summary"] = json.dumps(summarize_angles(angles_deg))
    trans.create_dataset("rotation_axis", data=np.array([0.0, 0.0, 1.0], dtype=np.float32))


def _write_nxdata(entry, projections_dataset):
    nxdata = entry.create_group("data")
    nxdata.attrs["NX_class"] = "NXdata"
    nxdata.attrs["signal"] = "projections"
    nxdata["projections"] = projections_dataset


def _write_tomojax_processing(entry, *, grid):
    nx_grid, ny_grid, nz_grid = grid
    proc = entry.create_group("processing")
    proc.attrs["NX_class"] = "NXprocess"
    tj = proc.create_group("tomojax")
    tj.attrs["NX_class"] = "NXcollection"
    tj.attrs["frame"] = "sample"
    tj.attrs[VOLUME_AXES_ATTR] = np.array(
        DISK_VOLUME_AXES,
        dtype=h5py.string_dtype(encoding="utf-8"),
    )

    volume_shape = (int(nz_grid), int(ny_grid), int(nx_grid))
    vol = tj.create_dataset(
        "volume",
        shape=volume_shape,
        dtype="float32",
        chunks=_volume_chunks(volume_shape),
        compression="lzf",
        shuffle=False,
        fletcher32=False,
    )
    vol[...] = 0.0
    vol.attrs["long_name"] = "ground_truth_volume"


@dataclass(frozen=True)
class PreparedWranglerData:
    projections: np.ndarray
    angles_deg: np.ndarray
    image_key: np.ndarray
    pixel_size_pixels_x: float
    pixel_size_pixels_y: float
    grid: tuple[int, int, int]
    voxels: tuple[float, float, float]


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
    image_key=None,
    sample_name="sample",
    source_name="TomoJAX source",
    source_type=None,
    source_probe="x-ray",
):
    """
    projections: float32 [P, ny, nx] absorption
    angles_deg: float32 [P]
    grid: (nx, ny, nz) for the 'processing/tomojax/volume' placeholder and grid_meta_json
    voxels: (vx, vy, vz) physical voxel sizes in same units as pixel_size_pixels_*
    """
    P, ny, nx = projections.shape
    if image_key is None:
        image_key_arr = np.zeros((P,), dtype=np.int32)
    else:
        image_key_arr = np.asarray(image_key, dtype=np.int32)
        if image_key_arr.shape != (P,):
            raise ValueError("image_key must have shape (n_views,)")
    nx_grid, ny_grid, nz_grid = grid
    vx, vy, vz = voxels

    with h5py.File(output_path, "w") as f:
        entry = _write_entry_metadata(
            f,
            grid=(nx_grid, ny_grid, nz_grid),
            voxels=(vx, vy, vz),
            tilt_deg=tilt_deg,
            tilt_about=tilt_about,
        )
        instr, projections_dataset = _write_detector(
            entry,
            projections=projections,
            image_key_arr=image_key_arr,
            pixel_size_pixels_x=pixel_size_pixels_x,
            pixel_size_pixels_y=pixel_size_pixels_y,
        )
        _write_source(
            instr,
            source_name=source_name,
            source_type=source_type,
            source_probe=source_probe,
        )
        _write_sample_transforms(entry, sample_name=sample_name, angles_deg=angles_deg)
        _write_nxdata(entry, projections_dataset)
        _write_tomojax_processing(entry, grid=(nx_grid, ny_grid, nz_grid))

    print(f"Wrote corrected absorption data to: {output_path}")


def _prepare_wrangler_data(args) -> PreparedWranglerData:
    data, angles_all, image_key = load_raw(
        args.input_path,
        args.proj_path,
        args.angles_path,
        args.image_key_path,
    )
    is_proj = image_key == 0
    angles = angles_all[is_proj].astype(np.float32)
    absorption = flat_dark_correct_to_absorption(
        data=data,
        image_key=image_key,
        min_intensity=float(args.min_intensity),
        assume_dark_field=args.assume_dark_field,
    )

    by = int(args.bin_y) if args.bin_y is not None else int(args.bin)
    bx = int(args.bin_x) if args.bin_x is not None else int(args.bin)
    if by > 1 or bx > 1:
        absorption = _spatial_bin(absorption, by, bx)
        print(f"Applied spatial binning: by={by}, bx={bx} -> new shape {absorption.shape}")

    if args.pad_y_multiple or args.pad_x_multiple:
        before = absorption.shape
        absorption = _pad_to_multiples(
            absorption,
            args.pad_y_multiple,
            args.pad_x_multiple,
            mode=str(args.pad_mode),
        )
        after = absorption.shape
        if after != before:
            print(
                f"Padded to multiples (y={args.pad_y_multiple}, x={args.pad_x_multiple}): "
                f"{before} -> {after}"
            )

    if absorption.shape[0] != angles.shape[0]:
        raise RuntimeError(
            f"Projection count mismatch: absorption {absorption.shape[0]} vs angles {angles.shape[0]}"
        )

    _, ny_pix, nx_pix = absorption.shape
    if args.grid is not None:
        grid = tuple(map(int, args.grid))
    elif args.grid_from_det:
        grid = (int(nx_pix), int(nx_pix), int(ny_pix))
    else:
        grid = (128, 128, 128)

    pixel_size_x = float(args.pixel_size) * float(bx)
    pixel_size_y = float(args.pixel_size) * float(by)
    vox = (
        tuple(map(float, args.voxels))
        if args.voxels is not None
        else (pixel_size_x, pixel_size_x, pixel_size_y)
    )
    return PreparedWranglerData(
        projections=absorption,
        angles_deg=angles,
        image_key=np.zeros((angles.shape[0],), dtype=np.int32),
        pixel_size_pixels_x=pixel_size_x,
        pixel_size_pixels_y=pixel_size_y,
        grid=grid,
        voxels=vox,
    )


def main():
    import argparse

    p = argparse.ArgumentParser(description="Flat/dark correct a NeXus file and export TomoJAX-compatible NXtomo.")
    p.epilog = "Volumes are saved on disk in (nz, ny, nx) order with @volume_axes_order=\"zyx\"; TomoJAX APIs always return (nx, ny, nz)."
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
    p.add_argument(
        "--assume-dark-field",
        type=float,
        default=None,
        help=(
            "Explicit constant dark-field value to use when image_key contains no dark "
            "frames. By default missing dark fields fail instead of silently using zero."
        ),
    )
    p.add_argument("--pad-y-multiple", type=int, default=None, help="Pad rows to nearest multiple (after binning)")
    p.add_argument("--pad-x-multiple", type=int, default=None, help="Pad cols to nearest multiple (after binning)")
    p.add_argument("--pad-mode", choices=["edge","constant","reflect"], default="edge", help="Padding mode (default: edge)")
    args = p.parse_args()

    prepared = _prepare_wrangler_data(args)
    write_nexus_h5(
        output_path=args.output_path,
        projections=prepared.projections,
        angles_deg=prepared.angles_deg,
        pixel_size_pixels_x=prepared.pixel_size_pixels_x,
        pixel_size_pixels_y=prepared.pixel_size_pixels_y,
        tilt_deg=float(args.tilt_deg),
        tilt_about=str(args.tilt_about),
        grid=prepared.grid,
        voxels=prepared.voxels,
        image_key=prepared.image_key,
        sample_name="sample",
        source_name="TomoJAX pipeline",
        source_type="experiment",
        source_probe="x-ray",
    )


if __name__ == "__main__":
    main()
