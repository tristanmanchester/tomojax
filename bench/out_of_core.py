"""Iterative cone-beam reconstruction of volumes larger than the GPU, or than memory.

CuO (a Zeiss ``.txrm`` scan, every view)::

    python bench/out_of_core.py cuo SCAN.txrm --binning 2 --iterations 3 --subsets 8

corrects the projections into memory with ``Frames.corrected`` and runs
OS-SART on the binned grid; the solver sweeps its volume through the device in
z slabs. With ``--scratch DIR`` (no binning) the projections are instead
corrected from the raw frames a band of detector rows at a time, as each sweep
needs them, and the volume is a memory-mapped file in ``DIR``, deleted at the
end; it starts as zero. ``--planes FILE.npz`` saves its central xy and xz planes.

A synthetic volume (spheres in a cylinder, a CuO-like cone)::

    python bench/out_of_core.py synthetic --size 1300 --views 180 --method cgls --scratch DIR

simulates its projections with the same slab sweep and reconstructs them with
CGLS or SIRT; with ``--scratch`` the iterate is a memmap there and CGLS keeps
its work volumes in unnamed files beside it.

Prints one JSON record: stage times, peak host memory (sampled anonymous
memory and ``ru_maxrss``), peak GPU memory, bytes moved to and from the device
and read from or written to disk, and the data residual on every eighth view
(in-memory CuO) or the image error (synthetic).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import resource
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _measure import GpuSampler, environment

import tomojax as tj
from tomojax.corrections import corrected_row_source
from tomojax.geometry import ConeBeam, ConeGeometry, Detector, Grid
from tomojax.recon import CGLSConfig, SIRTConfig, cgls, sirt


class _Memory:
    """Peak anonymous host memory, sampled every 0.2 s; progress on stderr every 2 min."""

    def __init__(self) -> None:
        self.peak = 0
        threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self) -> None:
        start = time.perf_counter()
        for count in itertools.count(1):
            with Path("/proc/self/status").open() as status:
                for line in status:
                    if line.startswith("RssAnon:"):
                        self.peak = max(self.peak, int(line.split()[1]) * 1024)
            if count % 600 == 0:
                progress = {"s": round(time.perf_counter() - start), "peak_anon": self.peak}
                print(json.dumps({**progress, **_disk()}), file=sys.stderr, flush=True)
            time.sleep(0.2)


def _disk() -> dict[str, int]:
    with Path("/proc/self/io").open() as io:
        values = {key: int(value) for key, value in (line.split(": ") for line in io)}
    return {"read": values["read_bytes"], "written": values["write_bytes"]}


def _cuo(args: argparse.Namespace, record: dict, memory: _Memory) -> None:
    frames = tj.load_frames(args.path, darks=0)
    if args.scratch is not None:
        args.binning = 1
    config = SIRTConfig(iterations=args.iterations, subsets=args.subsets)
    if args.scratch is None:
        start = time.perf_counter()
        scan = frames.corrected(binning=args.binning)
        grid = scan.grid.resampled((scan.detector.nu,) * 3)
        record["correct_s"] = time.perf_counter() - start
        start = time.perf_counter()
        result = tj.reconstruct(scan, "sirt", grid=grid, config=config)
        record["solve_s"] = time.perf_counter() - start
        record["info"] = {k: v for k, v in result.info.items() if isinstance(v, int | float | bool)}
        record["peak_host_anonymous_bytes_solving"] = memory.peak
        subset, volume = scan.selected(slice(0, None, 8)), result.volume
        del scan, result  # scoring needs only every eighth view
        predicted = np.array(tj.project(subset.geometry, volume, grid=grid))
        predicted -= subset.projections
        error = np.linalg.norm(predicted) / np.linalg.norm(subset.projections)
        record["residual_every_eighth_view"] = float(error)
        record["grid"] = [grid.nx, grid.ny, grid.nz]
        return
    from tomojax.recon._cone_slabs import RowReader

    geometry, grid = frames.geometry, frames.geometry.grid
    path = Path(args.scratch) / "volume.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    volume = np.lib.format.open_memmap(
        path, "w+", np.float32, (grid.nz, grid.ny, grid.nx), fortran_order=False
    )
    try:
        with corrected_row_source(
            frames.counts, flats=frames.flats, flat_positions=frames.flat_positions,
            darks=frames.darks, white_level=frames.white_level, steps=(),
            view_positions=frames.view_positions, exposure_times=frames.exposure_times,
            flat_exposure_times=frames.flat_exposure_times, host=True,
        ) as source:  # fmt: skip
            rows = RowReader(lambda first, stop: source(first, stop), frames.counts.shape)
            start, disk = time.perf_counter(), _disk()
            _, info = sirt(geometry, grid, geometry.detector, rows, out=volume, config=config)
            after = _disk()
        record["solve_s"] = time.perf_counter() - start
        record["disk"] = {key: after[key] - disk[key] for key in disk}
        record["info"] = {k: v for k, v in info.items() if isinstance(v, int | float | bool)}
        record["grid"] = [grid.nx, grid.ny, grid.nz]
        record["central_plane_mean"] = float(np.mean(volume[grid.nz // 2]))
        if args.planes:  # (a yz plane would read a page per voxel)
            np.savez(args.planes, xy=volume[grid.nz // 2], xz=volume[:, grid.ny // 2])
    finally:
        del volume
        path.unlink(missing_ok=True)


def _spheres(size: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centres, radii and attenuations (voxel units) of 24 spheres inside a larger one."""
    rng = np.random.default_rng(seed)
    centres = np.vstack([np.zeros((1, 3)), rng.uniform(-0.6, 0.6, (24, 3)) * size / 2])
    radii = np.concatenate([[0.8 * size / 2], rng.uniform(0.04, 0.18, 24) * size / 2])
    values = np.concatenate([[0.1], rng.uniform(0.2, 1.0, 24)])
    return centres, radii, values


def _sphere_planes(size: int, z0: int, z1: int) -> np.ndarray:
    """Planes ``[z0, z1)`` of the spheres sampled at voxel centres, each only in its box."""
    axis = np.arange(size, dtype=np.float32) - (size - 1) / 2
    z = np.arange(z0, z1, dtype=np.float32) - (size - 1) / 2
    out = np.zeros((z1 - z0, size, size), np.float32)
    for centre, radius, value in zip(*_spheres(size), strict=True):
        lo = np.searchsorted(axis, centre - radius)
        hi = np.searchsorted(axis, centre + radius, side="right")
        zl, zh = np.searchsorted(z, centre[2] - radius), np.searchsorted(z, centre[2] + radius)
        inside = (
            (axis[None, None, lo[0] : hi[0]] - centre[0]) ** 2
            + (axis[None, lo[1] : hi[1], None] - centre[1]) ** 2
            + (z[zl:zh, None, None] - centre[2]) ** 2
        ) < radius**2
        out[zl:zh, lo[1] : hi[1], lo[0] : hi[0]] += np.float32(value) * inside
    return out


def _sphere_projections(geometry: ConeGeometry) -> np.ndarray:
    """Exact cone-beam line integrals of the spheres: chord lengths times attenuation."""
    detector, beam = geometry.detector, geometry.beam
    centre, u_dir, v_dir = beam.detector_frame(detector)
    u = (np.arange(detector.nu) - (detector.nu - 1) / 2) * detector.du
    v = (np.arange(detector.nv) - (detector.nv - 1) / 2) * detector.dv
    source = beam.source()
    rays = centre + u[None, :, None] * u_dir + v[:, None, None] * v_dir - source
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    poses = geometry.poses()
    data = np.zeros((len(poses), detector.nv, detector.nu), np.float32)
    for view, pose in enumerate(poses):
        for c, radius, value in zip(*_spheres(detector.nu), strict=True):
            offset = pose[:3, :3] @ c + pose[:3, 3] - source
            along = rays @ offset
            gap = radius**2 - (offset @ offset - along**2)
            data[view] += (2 * value * np.sqrt(np.maximum(gap, 0))).astype(np.float32)
    return data


def _synthetic(args: argparse.Namespace, record: dict, memory: _Memory) -> None:
    size, magnification = args.size, 3.03
    distance = 10279 * size / 2036  # the CuO scan's cone angle
    grid = Grid(size, size, size, 1.0, 1.0, 1.0)
    detector = Detector(size, size, magnification, magnification)
    angles = np.linspace(0, 360, args.views, endpoint=False)
    geometry = ConeGeometry(grid, detector, angles, ConeBeam(distance, magnification * distance))
    start = time.perf_counter()
    data = _sphere_projections(geometry)
    record["simulate_s"] = time.perf_counter() - start
    out, path = None, None
    if args.scratch is not None:
        path = Path(args.scratch) / "volume.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        shape = (size, size, size)
        out = np.lib.format.open_memmap(path, "w+", np.float32, shape, fortran_order=False)
    try:
        start, disk = time.perf_counter(), _disk()
        if args.method == "cgls":
            config = CGLSConfig(iterations=args.iterations, rtol=0)
            volume, info = cgls(geometry, grid, detector, data, config=config, out=out)
        else:
            config = SIRTConfig(iterations=args.iterations)
            volume, info = sirt(geometry, grid, detector, data, config=config, out=out)
        after = _disk()
        record["solve_s"] = time.perf_counter() - start
        record["disk"] = {key: after[key] - disk[key] for key in disk}
        record["info"] = {k: v for k, v in info.items() if isinstance(v, int | float | bool)}
        record["peak_host_anonymous_bytes_solving"] = memory.peak
        error = norm = 0.0
        for z0 in range(0, size, 64):
            truth = _sphere_planes(size, z0, min(size, z0 + 64))
            error += float(np.sum((np.asarray(volume[z0 : z0 + 64]) - truth) ** 2))
            norm += float(np.sum(truth.astype(np.float64) ** 2))
        record["relative_image_error"] = (error / norm) ** 0.5
    finally:
        del out
        if path is not None:
            path.unlink(missing_ok=True)


def main() -> None:
    """Run one case and print its record."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="case", required=True)
    cuo = sub.add_parser("cuo")
    cuo.add_argument("path")
    cuo.add_argument("--binning", type=int, default=2, help="ignored with --scratch")
    cuo.add_argument("--iterations", type=int, default=3)
    cuo.add_argument("--subsets", type=int, default=8)
    cuo.add_argument("--scratch")
    cuo.add_argument("--planes", help="save the central planes to this .npz (with --scratch)")
    synthetic = sub.add_parser("synthetic")
    synthetic.add_argument("--size", type=int, default=1300)
    synthetic.add_argument("--views", type=int, default=180)
    synthetic.add_argument("--method", choices=("cgls", "sirt"), default="cgls")
    synthetic.add_argument("--iterations", type=int, default=3)
    synthetic.add_argument("--scratch")
    args = parser.parse_args()
    record: dict = {"environment": environment(), "case": vars(args)}
    gpu, memory, start = GpuSampler(period=0.25), _Memory(), time.perf_counter()
    try:
        (_cuo if args.case == "cuo" else _synthetic)(args, record, memory)
    finally:
        record["gpu"] = gpu.window(start, time.perf_counter())
        gpu.close()
    record["peak_host_anonymous_bytes"] = memory.peak
    record["ru_maxrss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    record["total_s"] = time.perf_counter() - start
    record.pop("environment")
    print(json.dumps(record, indent=1, default=str))


if __name__ == "__main__":
    main()
