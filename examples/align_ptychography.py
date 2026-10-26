"""Prepare, align and reconstruct PSI ptychography with joint projection matching.

Usage: python examples/align_ptychography.py example_data.mat --binning 4
Also accepts porous_glass_dataset.mat (physical refractive-index output).
The measured porous-glass stack is already aligned; the simulation has motion.
Projection matching can fit noise or model mismatch as motion. Inspect poses and
reconstructed image quality; convergence does not imply physical accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, cast

import h5py
import numpy as np
from scipy.io import loadmat

import tomojax as tj
from tomojax.corrections import remove_phase_ramp, unwrap_phase

if TYPE_CHECKING:
    from tomojax.alignment import ProjectionMatchingConfig

# Measured-data detector ROI and known air strips, not fitted poses.
PSI_ROWS = slice(212, 628)
PSI_COLUMNS = slice(197, 1509)
PSI_AIR_COLUMNS = 50


def read_psi_objects(
    path: str | Path, *, rows: slice | None = None, columns: slice | None = None
) -> tuple[np.ndarray, np.ndarray, float | None, float | None]:
    """Read (view,v,u) objects, degrees, wavelength and pixel size from PSI examples.

    MATLAB v5 ``example_data.mat`` holds ``stack_object`` and ``theta`` with
    no physical scale: wavelength and pixel size are returned as None.
    The v7.3 porous-glass file reverses the stored axes and includes those
    lengths in metres. Its default crop excludes unreliable detector borders;
    the smaller simulated example uses the full detector by default.
    Explicit ``rows`` and ``columns`` override either default crop.

    Raw HDF5 chunks span all views, so reading one view at a time would
    repeatedly decompress the scan. No aligned phase or reference is read.
    """
    if not h5py.is_hdf5(path):
        mat = loadmat(path, variable_names=["stack_object", "theta"])
        objects = np.asarray(mat["stack_object"]).transpose(2, 0, 1)
        objects = objects[:, rows or slice(None), columns or slice(None)]
        return objects, np.asarray(mat["theta"], dtype=np.float64).ravel(), None, None
    with h5py.File(path, "r") as source:
        dataset = cast("h5py.Dataset", source["stack_object"])
        raw = np.asarray(dataset[:, columns or PSI_COLUMNS, rows or PSI_ROWS])
        objects = raw["real"] + 1j * raw["imag"] if raw.dtype.names else raw
        angles = np.asarray(source["rotation_angle"], dtype=np.float64).ravel()
        wavelength = float(np.asarray(source["lambda"]).item())
        pixel_size = float(np.asarray(source["pixel_size"]).item())
    return objects.transpose(0, 2, 1), angles, wavelength, pixel_size


def prepare_phase(objects: np.ndarray, air_mask: np.ndarray, *, batch_size: int = 16) -> np.ndarray:
    """Return positive line-integral phase in radians, with bounded device batches.

    ``objects`` is (view,v,u); ``air_mask`` marks known air on the detector.
    The negative transmission phase is the positive tomographic signal.
    Complex intensity weights the air-plane fit, not the sample itself.
    """
    phase = np.empty(objects.shape, dtype=np.float32)
    for first in range(0, len(objects), batch_size):
        batch = objects[first : first + batch_size]
        corrected = remove_phase_ramp(unwrap_phase(batch), air_mask=air_mask * np.abs(batch) ** 2)
        phase[first : first + batch_size] = -np.asarray(corrected)
    return phase


def make_scan(phase: np.ndarray, angles: np.ndarray, *, binning: int = 1) -> tj.Scan:
    """Use original-pixel units, with a grid matching the binned detector."""
    _, nv, nu = phase.shape
    grid = tj.Grid(nu, nu, nv, 1, 1, 1)
    detector = tj.Detector(nu, nv, 1, 1)
    scan = tj.Scan(phase, tj.ParallelGeometry(grid, detector, angles.tolist())).binned(binning)
    detector = scan.detector
    grid = tj.Grid(
        detector.nu,
        detector.nu,
        detector.nv,
        detector.du,
        detector.du,
        detector.dv,
        vol_center=(0, 0, detector.center[1]),
    )
    return tj.Scan(scan.projections, tj.ParallelGeometry(grid, detector, angles.tolist()))


def alignment_config(*, iterations: int = 20) -> ProjectionMatchingConfig:
    """The fixed sparse-joint translation estimator."""
    from tomojax.alignment import ProjectionMatchingConfig

    return ProjectionMatchingConfig(iterations=iterations)


def run_example(
    path: str | Path, *, binning: int = 1, iterations: int = 20
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    """Return a volume, detector-frame poses and timings with explicit units.

    Binning averages unwrapped phase, not complex transmission, and retains
    original-pixel geometry and pose units. The grid follows the binned
    detector. Values are delta when the file supplies physical calibration,
    otherwise phase per original pixel (rad/pixel), not refractive index.
    """
    start = time.perf_counter()
    objects, angles, wavelength, pixel_size = read_psi_objects(path)
    timings: dict[str, float | str] = {"read_seconds": time.perf_counter() - start}
    air = np.zeros(objects.shape[-2:], dtype=np.float32)
    air[:, :PSI_AIR_COLUMNS] = air[:, -PSI_AIR_COLUMNS:] = 1
    before = time.perf_counter()
    phase = prepare_phase(objects, air)
    del objects
    timings["phase_seconds"] = time.perf_counter() - before
    scan = make_scan(phase, angles, binning=binning)
    grid = scan.grid

    before = time.perf_counter()
    aligned = tj.align(scan, mode="pose", grid=grid, config=alignment_config(iterations=iterations))
    timings["alignment_seconds"] = time.perf_counter() - before
    timings["estimation_seconds"] = float(cast("float", aligned.info["estimation_seconds"]))
    corrected_scan, poses = aligned.scan, np.asarray(aligned.poses).copy()
    del aligned  # Release alignment's image before the final full-grid inverse.
    before = time.perf_counter()
    volume = np.asarray(tj.reconstruct(corrected_scan, "fourier").volume)
    timings["value_unit"] = "rad/pixel"
    timings["voxel_size_pixels"] = grid.vx
    if wavelength is not None and pixel_size is not None:
        # Geometry remains in original pixels, including after binning.
        volume = volume * (wavelength / (2 * np.pi * pixel_size))
        timings["value_unit"] = "delta"
        timings["pixel_size_m"] = pixel_size
        timings["voxel_size_m"] = pixel_size * grid.vx
    timings["reconstruction_seconds"] = time.perf_counter() - before
    timings["whole_seconds"] = time.perf_counter() - start
    return volume, poses, timings


def main() -> None:
    parser = argparse.ArgumentParser(description=cast("str", __doc__).split("\n\n")[0])
    parser.add_argument("path", type=Path, help="PSI example_data.mat or porous_glass_dataset.mat")
    parser.add_argument(
        "-o", "--output", type=Path, help="Optional NPZ of central slices and poses"
    )
    parser.add_argument(
        "--binning", type=int, default=1, help="Detector phase binning (4 for a CPU preview)"
    )
    parser.add_argument("--iterations", type=int, default=20, help="Joint alignment iterations")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    volume, poses, timings = run_example(
        args.path, binning=args.binning, iterations=args.iterations
    )
    print(json.dumps(timings, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        nz, ny, nx = volume.shape
        quantity = "delta" if timings["value_unit"] == "delta" else "phase_per_pixel"
        slices = {
            f"{quantity}_xy": volume[nz // 2],
            f"{quantity}_xz": volume[:, ny // 2, :],
            f"{quantity}_yz": volume[:, :, nx // 2],
        }
        np.savez_compressed(
            args.output,
            allow_pickle=False,
            **slices,
            poses=poses,
            value_unit=timings["value_unit"],
            voxel_size_pixels=timings["voxel_size_pixels"],
            pixel_size_m=timings.get("pixel_size_m", np.nan),
        )


if __name__ == "__main__":
    main()
