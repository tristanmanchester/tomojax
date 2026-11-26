r"""Parallel-beam reconstruction of a real scan against ASTRA, TomoPy and tomocupy.

    uv run --no-sync python bench/compare_parallel.py RAW.nxs WORK --axis 1417.5 \
        --baselines-python BASELINES/bin/python

``RAW.nxs`` is an NXtomo file of raw counts with flats and darks (the Diamond I12
cut ``full_rows1048-1111.nxs`` of Zenodo 3339629 in docs/measurements.md);
``--axis`` is the rotation axis as a detector column. ``BASELINES`` is a
separate environment holding TomoPy and tomocupy, whose NumPy pins conflict
with TomoJAX's. ``WORK`` receives the prepared inputs and the outputs.

Once, untimed: the corrected line integrals (``lines.npy``, ``angles.npy``) and a
Data Exchange copy of the raw counts (``dx.h5``) that tomocupy reads.

In memory: TomoJAX's default ``tj.reconstruct`` (the Fourier-slice inverse for an
axial half turn), TomoJAX FBP, ASTRA ``FBP_CUDA`` slice by slice and TomoPy
gridrec, each in a fresh process from the line integrals in host memory to the
volume in host memory, all on the same axis and a ``nu`` x ``nu`` slice per row.
Cold is imports plus the first call; warm is the median of three repeats.

End to end: TomoJAX's default and FBP (``tj.load_frames``, ``corrected()``,
``tj.reconstruct``, ``np.save``) and ``tomocupy recon`` (fourierrec, ramp
filter), each a fresh process from the raw file on disk to a volume on disk,
five interleaved repeats with ``sync`` before each, outside the timer.

Each result is printed as a JSON line; agreement is the correlation of each
middle slice with TomoJAX FBP's after a 2-pixel Gaussian blur, inside the
slice's central disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

IN_MEMORY = ("tomojax-auto", "tomojax-fbp", "astra", "tomopy")


def prepare(raw: Path, work: Path) -> None:
    """Write the corrected line integrals and a Data Exchange copy of the raw counts."""
    import h5py

    import tomojax as tj

    frames = tj.load_frames(raw)
    scan = frames.corrected()
    np.save(work / "lines.npy", np.asarray(scan.projections, np.float32))
    np.save(work / "angles.npy", np.asarray(scan.geometry.angles, np.float64))
    with h5py.File(raw) as source, h5py.File(work / "dx.h5", "w") as target:
        detector = source["entry/instrument/detector"]
        key = np.asarray(detector["image_key"])
        data = detector["data"]
        target["exchange/data"] = data[key == 0]
        target["exchange/data_white"] = data[key == 1]
        target["exchange/data_dark"] = data[key == 2]
        target["exchange/theta"] = np.asarray(scan.geometry.angles, np.float64)


def reconstructor(tool: str, lines: np.ndarray, angles: np.ndarray, axis: float) -> Any:
    """A no-argument call reconstructing every row of ``lines`` with ``tool``."""
    views, rows, nu = lines.shape
    if tool.startswith("tomojax"):
        import tomojax as tj

        detector = tj.Detector(nu, rows, 1, 1, center=((nu - 1) / 2 - axis, 0.0))
        scan = tj.Scan(lines, tj.ParallelGeometry(tj.Grid(nu, nu, rows, 1, 1, 1), detector, angles))
        method = "fbp" if tool == "tomojax-fbp" else "auto"
        return lambda: np.asarray(tj.reconstruct(scan, method).volume, np.float32)
    theta = np.radians(angles)
    if tool == "tomopy":
        import tomopy

        return lambda: np.asarray(
            tomopy.recon(lines, theta, center=axis, algorithm="gridrec", filter_name="ramlak"),
            np.float32,
        )
    import astra

    projection = astra.functions.geom_postalignment(
        astra.create_proj_geom("parallel", 1.0, nu, theta), (nu - 1) / 2 - axis
    )
    volume = astra.create_vol_geom(nu, nu)

    def slices() -> np.ndarray:
        out = np.empty((rows, nu, nu), np.float32)
        for row in range(rows):
            sinogram = astra.data2d.create("-sino", projection, lines[:, row, :])
            image = astra.data2d.create("-vol", volume)
            config = astra.astra_dict("FBP_CUDA")
            config["ReconstructionDataId"], config["ProjectionDataId"] = image, sinogram
            algorithm = astra.algorithm.create(config)
            astra.algorithm.run(algorithm)
            out[row] = astra.data2d.get(image)
            astra.algorithm.delete(algorithm)
            astra.data2d.delete([sinogram, image])
        return out

    return slices


def in_memory(tool: str, work: Path, axis: float, started: float) -> None:
    """Time one tool from host line integrals to a host volume, cold and warm."""
    lines, angles = np.load(work / "lines.npy"), np.load(work / "angles.npy")
    run = reconstructor(tool, lines, angles, axis)
    volume = run()
    cold = time.perf_counter() - started
    warm = []
    for _ in range(3):
        start = time.perf_counter()
        volume = run()
        warm.append(time.perf_counter() - start)
    np.save(work / f"slice-{tool}.npy", volume[volume.shape[0] // 2])
    print(json.dumps({"tool": tool, "cold_s": cold, "warm_s": sorted(warm)[1], "warm": warm}))


def end_to_end(raw: Path, out: Path, axis: float, method: str) -> None:
    """Raw counts on disk to a volume on disk, as a user runs TomoJAX."""
    import tomojax as tj

    frames = tj.load_frames(raw)
    d = frames.geometry.detector
    detector = tj.Detector(d.nu, d.nv, d.du, d.dv, center=(((d.nu - 1) / 2 - axis) * d.du, 0.0))
    frames = tj.load_frames(raw, detector=detector)
    np.save(out, np.asarray(tj.reconstruct(frames.corrected(), method).volume, np.float32))


def timed(command: list[str], log: Path) -> float:
    """Wall seconds of ``command`` in a fresh process, after a sync outside the timer."""
    subprocess.run(["sync"], check=True)
    time.sleep(3)
    start = time.perf_counter()
    with log.open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - start


def agreement(work: Path) -> dict[str, float]:
    """Correlation of each tool's blurred middle slice with TomoJAX FBP's."""
    from scipy.ndimage import gaussian_filter

    reference = gaussian_filter(np.load(work / "slice-tomojax-fbp.npy").astype(np.float64), 2)
    n = reference.shape[0]
    centers = np.arange(n) - (n - 1) / 2
    disk = np.hypot(*np.meshgrid(centers, centers)) < 0.43 * n
    out = {}
    for tool in IN_MEMORY[:1] + IN_MEMORY[2:]:
        image = gaussian_filter(np.load(work / f"slice-{tool}.npy").astype(np.float64), 2)
        out[tool] = float(np.corrcoef(reference[disk], image[disk])[0, 1])
    return out


def main() -> None:
    """Prepare once, then time every tool in memory and end to end."""
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("raw", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--axis", type=float, required=True, help="rotation axis column")
    parser.add_argument("--baselines-python", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5, help="end-to-end repeats")
    parser.add_argument("--worker", choices=IN_MEMORY, help=argparse.SUPPRESS)
    parser.add_argument("--end-to-end", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--method", default="auto", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        in_memory(args.worker, args.work, args.axis, started)
        return
    if args.end_to_end:
        end_to_end(args.raw, args.end_to_end, args.axis, args.method)
        return
    args.work.mkdir(parents=True, exist_ok=True)
    if not (args.work / "dx.h5").exists():
        prepare(args.raw, args.work)
    me = [sys.executable, __file__, str(args.raw), str(args.work), "--axis", str(args.axis),
          "--baselines-python", str(args.baselines_python)]  # fmt: skip
    for tool in IN_MEMORY:
        python = str(args.baselines_python) if tool == "tomopy" else sys.executable
        command = [python, *me[1:], "--worker", tool]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(result.stdout.strip().splitlines()[-1], flush=True)
    print(json.dumps({"agreement_with_tomojax_fbp": agreement(args.work)}), flush=True)
    tomocupy = args.baselines_python.with_name("tomocupy")
    times: dict[str, list[float]] = {"tomojax-auto": [], "tomojax-fbp": [], "tomocupy": []}
    for _ in range(args.repeats):
        for method in ("auto", "fbp"):
            out = str(args.work / f"tomojax-{method}.npy")
            command = [*me, "--end-to-end", out, "--method", method]
            times[f"tomojax-{method}"].append(timed(command, args.work / f"{method}.log"))
        times["tomocupy"].append(
            timed(
                [str(tomocupy), "recon", "--file-name", str(args.work / "dx.h5"),
                 "--rotation-axis", str(args.axis), "--rotation-axis-auto", "manual",
                 "--reconstruction-type", "full", "--reconstruction-algorithm", "fourierrec",
                 "--fbp-filter", "ramp", "--dtype", "float32", "--save-format", "h5",
                 "--out-path-name", str(args.work / "tomocupy")],
                args.work / "tomocupy.log",
            )
        )  # fmt: skip
    print(json.dumps({"end_to_end_s": {k: {"median": float(np.median(v)), "all": v}
                                        for k, v in times.items()}}))  # fmt: skip


if __name__ == "__main__":
    main()
