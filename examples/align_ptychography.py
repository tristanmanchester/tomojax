"""Prepare, align and reconstruct PSI ptychography with joint projection matching.

Usage: python examples/align_ptychography.py porous_glass_dataset.mat
The released stack is already aligned, but projection matching can still fit
noise or model mismatch as motion. Inspect poses and reconstructed image quality;
convergence does not imply physical accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, cast

import h5py
import numpy as np

import tomojax as tj
from tomojax.corrections import remove_phase_ramp, unwrap_phase

if TYPE_CHECKING:
    from tomojax.alignment import ProjectionMatchingConfig

# This dataset's reliable detector ROI and known air strips, not fitted poses.
PSI_ROWS = slice(212, 628)
PSI_COLUMNS = slice(197, 1509)
PSI_AIR_COLUMNS = 50


def read_psi_objects(
    path: str | Path, *, rows: slice = PSI_ROWS, columns: slice = PSI_COLUMNS
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Read the valid ROI once, returning (view,v,u), degrees, wavelength, pixel size.

    MATLAB v7.3 reverses the stored axes. The raw HDF5 chunks span all views,
    so reading one view at a time would repeatedly decompress the whole scan.
    No supplied aligned phase or reference reconstruction is read.
    """
    with h5py.File(path, "r") as source:
        dataset = cast("h5py.Dataset", source["stack_object"])
        raw = np.asarray(dataset[:, columns, rows])
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


def make_scan(phase: np.ndarray, angles: np.ndarray) -> tj.Scan:
    """Use pixel units throughout alignment and reconstruction."""
    _, nv, nu = phase.shape
    grid = tj.Grid(nu, nu, nv, 1, 1, 1)
    detector = tj.Detector(nu, nv, 1, 1)
    return tj.Scan(phase, tj.ParallelGeometry(grid, detector, angles.tolist()))


def alignment_config() -> ProjectionMatchingConfig:
    """The fixed sparse-joint translation estimator."""
    from tomojax.alignment import ProjectionMatchingConfig

    return ProjectionMatchingConfig(iterations=20)


def run_example(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return physical delta, detector-frame poses and workflow timings."""
    start = time.perf_counter()
    objects, angles, wavelength, pixel_size = read_psi_objects(path)
    timings: dict[str, float] = {"read_seconds": time.perf_counter() - start}
    air = np.zeros(objects.shape[-2:], dtype=np.float32)
    air[:, :PSI_AIR_COLUMNS] = air[:, -PSI_AIR_COLUMNS:] = 1
    before = time.perf_counter()
    phase = prepare_phase(objects, air)
    del objects
    timings["phase_seconds"] = time.perf_counter() - before
    scan = make_scan(phase, angles)

    before = time.perf_counter()
    aligned = tj.align(scan, mode="pose", config=alignment_config())
    timings["alignment_seconds"] = time.perf_counter() - before
    timings["estimation_seconds"] = float(cast("float", aligned.info["estimation_seconds"]))
    corrected_scan, poses = aligned.scan, np.asarray(aligned.poses).copy()
    del aligned  # Release alignment's image before the final full-grid inverse.
    before = time.perf_counter()
    delta = np.asarray(tj.reconstruct(corrected_scan, "fourier").volume)
    # Geometry was in pixels; wavelength and pixel_size are both in metres.
    delta *= wavelength / (2 * np.pi * pixel_size)
    timings["reconstruction_seconds"] = time.perf_counter() - before
    timings["whole_seconds"] = time.perf_counter() - start
    timings["pixel_size_m"] = pixel_size
    return delta, poses, timings


def main() -> None:
    parser = argparse.ArgumentParser(description=cast("str", __doc__).split("\n\n")[0])
    parser.add_argument("path", type=Path, help="PSI porous_glass_dataset.mat")
    parser.add_argument(
        "-o", "--output", type=Path, help="Optional NPZ of central slices and poses"
    )
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    delta, poses, timings = run_example(args.path)
    print(json.dumps(timings, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        nx, ny, nz = delta.shape
        np.savez_compressed(
            args.output,
            delta_xy=delta[:, :, nz // 2],
            delta_xz=delta[:, ny // 2, :],
            delta_yz=delta[nx // 2, :, :],
            poses=poses,
            pixel_size_m=cast("float", timings["pixel_size_m"]),
        )


if __name__ == "__main__":
    main()
