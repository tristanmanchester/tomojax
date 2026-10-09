r"""Reconstruct a FIPS walnut (real lab cone-beam CT) and compare with ASTRA and the reference.

The data are Der Sarkissian et al., "A cone-beam X-ray CT data collection
designed for machine learning", Scientific Data 6, 215 (2019); CC BY 4.0, on
Zenodo (records 2686726 onwards). Each walnut has three source orbits of 1201
projections (972 x 768 after the scanner's 2 x 2 binning) with per-view
geometry vectors, and a reference reconstruction: 50 iterations of
non-negative least squares on all three orbits, 501^3 voxels of 0.1 mm.

Loading follows the authors' WalnutReconstructionCodes: projections are read in
reverse order and transposed, dark- and flat-corrected and log-transformed.

    python bench/walnut.py ~/data/walnuts/Walnut1 --orbits 2 --method fbp --astra
    python bench/walnut.py ~/data/walnuts/Walnut1 --orbits 1 2 3 --method fista \\
        --iterations 50
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any

import imageio.v3 as iio
import numpy as np

import tomojax as tj

ROWS, COLS, VIEWS = 972, 768, 1200


def _read(path: Path) -> np.ndarray:
    # The scanner reads out in portrait mode: flip rows, then transpose.
    return np.transpose(np.flipud(np.asarray(iio.imread(path), np.float32)))


def load_orbit(walnut: Path, orbit: int, every: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """ASTRA ``(rows, views, columns)`` absorption data and cone_vec vectors of one orbit."""
    folder = walnut / "Projections" / f"tubeV{orbit}"
    vectors = np.loadtxt(folder / "scan_geom_corrected.geom")[0:VIEWS:every]
    dark = _read(folder / "di000000.tif")
    flat = np.mean([_read(folder / f"io00000{k}.tif") for k in (0, 1)], axis=0) - dark
    indices = range(VIEWS, 0, -every)
    data = np.empty((ROWS, len(indices), COLS), np.float32)
    for view, index in enumerate(indices):
        image = (_read(folder / f"scan_{index:06d}.tif") - dark) / flat
        data[:, view, :] = -np.log(np.clip(image, 1e-6, None))
    return data, vectors


def reference(walnut: Path, name: str = "full_AGD_50") -> np.ndarray | None:
    """A published reconstruction as an ``(x, y, z)`` array, if present.

    ``full_AGD_50`` is the three-orbit reference; ``fdk_posN`` the authors'
    ASTRA FDK of orbit N.
    """
    files = sorted((walnut / "Reconstructions").glob(f"{name}_*.tiff"))
    if not files:
        return None
    zyx = np.stack([np.asarray(iio.imread(f), np.float32) for f in files])
    return np.transpose(zyx, (2, 1, 0))


def volume_geometry(voxels_per_mm: int = 10) -> dict[str, Any]:
    """The reference's 50.1 mm cube, as an ASTRA volume geometry."""
    n = 50 * voxels_per_mm + 1
    half = n / voxels_per_mm / 2
    window = {f"WindowMin{a}": -half for a in "XYZ"} | {f"WindowMax{a}": half for a in "XYZ"}
    return {"GridColCount": n, "GridRowCount": n, "GridSliceCount": n, "option": window}


def _astra_agd(operator: Any, data: np.ndarray, iterations: int) -> np.ndarray:
    """The reference's solver: Nesterov-accelerated projected gradient, x >= 0.

    As the authors' NesterovGradient.py ASTRA plugin: step 1/L from ten power
    iterations, momentum restarted whenever the residual rises.
    """
    rng = np.random.default_rng(0)
    b = rng.random(operator.shape[1]).astype(np.float32)
    for _ in range(10):
        b = operator.T * (operator * b)
        lipschitz = float(np.linalg.norm(b))
        b /= lipschitz
    step = 1.0 / lipschitz
    sino = data.ravel()
    aty = operator.T * sino
    x = np.zeros(operator.shape[1], np.float32)
    x_old, normal, normal_old = x.copy(), np.zeros_like(x), np.zeros_like(x)
    gradient, t, losses = -aty, 1.0, []
    for _ in range(iterations):
        tau = (t - 1) / (t + 2)
        t += 1
        direction = gradient - tau / step * (x - x_old) + tau * (normal - normal_old)
        x_old[:] = x
        x = np.clip(x - step * direction, 0, None)
        forward = operator * x
        normal_old, normal = normal, operator.T * forward
        gradient = normal - aty
        losses.append(0.5 * float(np.linalg.norm(forward - sino)) ** 2)
        if losses[-1] > min(losses):
            t = 1.0
    return x


def astra_reconstruct(
    data: np.ndarray,
    proj_geom: dict[str, Any],
    vol_geom: dict[str, Any],
    method: str,
    iterations: int,
) -> np.ndarray:
    """ASTRA FDK, or the reference's accelerated NNLS, on the same data, as ``(x, y, z)``."""
    import astra

    proj_geom = astra.create_proj_geom(
        "cone_vec",
        proj_geom["DetectorRowCount"],
        proj_geom["DetectorColCount"],
        proj_geom["Vectors"],
    )
    geometry = astra.create_vol_geom(
        vol_geom["GridRowCount"], vol_geom["GridColCount"], vol_geom["GridSliceCount"]
    )
    geometry["option"].update(vol_geom["option"])
    shape = (geometry["GridSliceCount"], geometry["GridRowCount"], geometry["GridColCount"])
    if method != "fbp":
        projector = astra.create_projector("cuda3d", proj_geom, geometry)
        operator = astra.OpTomo(projector)
        volume = _astra_agd(operator, np.ascontiguousarray(data), iterations).reshape(shape)
        astra.projector.delete(projector)
        return np.transpose(volume, (2, 1, 0))
    volume = np.zeros(shape, np.float32)
    volume_id = astra.data3d.link("-vol", geometry, volume)
    sino_id = astra.data3d.link("-sino", proj_geom, np.ascontiguousarray(data))
    config = astra.astra_dict("FDK_CUDA")
    config["ProjectionDataId"], config["ReconstructionDataId"] = sino_id, volume_id
    algorithm = astra.algorithm.create(config)
    astra.algorithm.run(algorithm, 1)
    astra.algorithm.delete(algorithm)
    astra.data3d.delete([volume_id, sino_id])
    return np.transpose(volume, (2, 1, 0))


def _start_gpu_runtimes(*, astra: bool) -> None:
    """Start JAX's GPU runtime (about 2 s per process) and ASTRA's, outside the timings.

    Each library's first reconstruction is still timed, TomoJAX's compilation included.
    """
    import jax.numpy as jnp

    jnp.zeros(()).block_until_ready()
    if astra:
        import astra as astra_toolbox

        astra_toolbox.use_cuda()


def compare(volume: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Relative error and PSNR against ``truth`` over the walnut's support."""
    support = truth > 0.1 * float(np.percentile(truth, 99.9))
    difference = volume - truth
    peak = float(np.percentile(truth, 99.9))
    return {
        "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(truth)),
        "relative_l2_in_walnut": float(
            np.linalg.norm(difference[support]) / np.linalg.norm(truth[support])
        ),
        "psnr_db": float(20 * np.log10(peak / np.sqrt(np.mean(difference**2)))),
    }


def _with_astra(
    scan: tj.Scan,
    args: argparse.Namespace,
    ours: np.ndarray,
    truth: np.ndarray | None,
    *,
    gpus: int,
) -> dict[str, Any]:
    """ASTRA's reconstruction of ``scan`` on ``gpus`` GPUs: time, errors and ours against it."""
    import astra

    astra.set_gpu_index(list(range(gpus)))
    start = time.perf_counter()
    # ASTRA gets the same (binned) data and geometry back from the scan.
    data, geometry, volume = scan.to_astra()
    theirs = astra_reconstruct(data, geometry, volume, args.method, args.iterations)
    record: dict[str, Any] = {
        "method": "FDK_CUDA" if args.method == "fbp" else "accelerated NNLS (the reference's)",
        "seconds": time.perf_counter() - start,
    }
    if truth is not None:
        record |= compare(theirs, truth)
    difference = float(np.linalg.norm(ours - theirs) / np.linalg.norm(theirs))
    return {"astra": record, "tomojax_vs_astra_relative_l2": difference}


def main() -> None:
    """Run the walnut comparison and print a JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("walnut", type=Path, help="A WalnutN directory from the data collection")
    parser.add_argument("--orbits", type=int, nargs="+", default=[2])
    parser.add_argument("--every", type=int, default=1, help="Use every Nth view")
    parser.add_argument("--method", choices=["fbp", "cgls", "fista"], default="fbp")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--voxels-per-mm", type=int, default=10)
    parser.add_argument("--bin", type=int, default=1, help="Average N x N detector pixels")
    parser.add_argument("--astra", action="store_true", help="Also reconstruct with ASTRA")
    parser.add_argument("--gpus", type=int, default=1, help="GPUs each library may use")
    parser.add_argument("--save", type=Path, help="Save the TomoJAX reconstruction (.nxs)")
    args = parser.parse_args()
    # The scanner's TIFFs carry a malformed tag that tifffile reports and skips.
    logging.getLogger("tifffile").setLevel(logging.ERROR)

    start = time.perf_counter()
    orbits = [load_orbit(args.walnut, orbit, args.every) for orbit in args.orbits]
    data = np.concatenate([d for d, _ in orbits], axis=1)
    vectors = np.concatenate([v for _, v in orbits])
    vol_geom = volume_geometry(args.voxels_per_mm)
    scan = tj.Scan.from_astra(
        data,
        {
            "type": "cone_vec",
            "DetectorRowCount": ROWS,
            "DetectorColCount": COLS,
            "Vectors": vectors,
        },
        vol_geom,
    ).binned(args.bin)
    summary: dict[str, Any] = {
        "walnut": args.walnut.name,
        "orbits": args.orbits,
        "views": len(vectors),
        "bin": args.bin,
        "load_seconds": time.perf_counter() - start,
        "geometry": [
            {
                "source_to_axis_mm": segment.beam.source_to_axis,
                "source_to_detector_mm": segment.beam.source_to_detector,
                "detector_centre_mm": list(segment.detector.center),
                "pixel_mm": [segment.detector.du, segment.detector.dv],
            }
            for segment in getattr(scan.geometry, "segments", (scan.geometry,))
        ],
        "largest_pose_correction": None
        if scan.poses is None
        else {
            "rotation_deg": float(np.rad2deg(np.abs(scan.poses[:, :3]).max())),
            "shift_mm": float(np.abs(scan.poses[:, 3:]).max()),
        },
    }
    options = {} if args.method == "fbp" else {"iterations": args.iterations}
    if args.method == "fista":
        options |= {"tv_weight": 0.0, "nonnegative": True}
    _start_gpu_runtimes(astra=args.astra)
    import jax

    devices = jax.devices()[: args.gpus]
    start = time.perf_counter()
    result = tj.reconstruct(scan, args.method, devices=devices, **options)
    volume = np.asarray(result.volume)
    summary["tomojax"] = {"method": args.method, **options, "seconds": time.perf_counter() - start}
    summary["gpus"] = len(devices)
    truth = reference(args.walnut)
    if truth is not None:
        summary["tomojax"] |= compare(volume, truth)
    if args.method == "fbp" and len(args.orbits) == 1:
        published = reference(args.walnut, f"fdk_pos{args.orbits[0]}")
        if published is not None:
            summary["tomojax_vs_published_fdk"] = compare(volume, published)
    if args.astra:
        summary |= _with_astra(scan, args, volume, truth, gpus=len(devices))
    if args.save is not None:
        tj.save(args.save, result)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
