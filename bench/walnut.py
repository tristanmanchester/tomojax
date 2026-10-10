r"""Reconstruct a FIPS walnut (real lab cone-beam CT) and compare with ASTRA and the reference.

The data are Der Sarkissian et al., "A cone-beam X-ray CT data collection
designed for machine learning", Scientific Data 6, 215 (2019); CC BY 4.0, on
Zenodo (records 2686726 onwards). Each walnut has three source orbits of 1201
projections (972 x 768 after the scanner's 2 x 2 binning) with per-view
geometry vectors, and a reference reconstruction: 50 iterations of
non-negative least squares on all three orbits, 501^3 voxels of 0.1 mm.

Loading follows the authors' WalnutReconstructionCodes: projections are read in
reverse order and transposed, dark- and flat-corrected and log-transformed.

    python bench/walnut.py ~/data/walnuts/Walnut1 --orbits 2 --method fbp --libraries tomojax astra
    python bench/walnut.py ~/data/walnuts/Walnut1 --orbits 1 2 3 --method fista \\
        --iterations 50
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

import imageio.v3 as iio
import numpy as np

import tomojax as tj

if TYPE_CHECKING:
    from collections.abc import Callable

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
    """ASTRA FDK, CGLS3D_CUDA, or the reference's accelerated NNLS, as ``(x, y, z)``."""
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
    if method == "fista":  # non-negative least squares: the reference's AGD
        projector = astra.create_projector("cuda3d", proj_geom, geometry)
        operator = astra.OpTomo(projector)
        volume = _astra_agd(operator, np.ascontiguousarray(data), iterations).reshape(shape)
        astra.projector.delete(projector)
        return np.transpose(volume, (2, 1, 0))
    volume = np.zeros(shape, np.float32)
    volume_id = astra.data3d.link("-vol", geometry, volume)
    sino_id = astra.data3d.link("-sino", proj_geom, np.ascontiguousarray(data))
    config = astra.astra_dict("FDK_CUDA" if method == "fbp" else "CGLS3D_CUDA")
    config["ProjectionDataId"], config["ReconstructionDataId"] = sino_id, volume_id
    algorithm = astra.algorithm.create(config)
    astra.algorithm.run(algorithm, 1 if method == "fbp" else iterations)
    astra.algorithm.delete(algorithm)
    astra.data3d.delete([volume_id, sino_id])
    return np.transpose(volume, (2, 1, 0))


def _start_gpu_runtimes(*, tomojax: bool, astra: bool) -> None:
    """Start JAX's GPU runtime (about 2 s per process) and ASTRA's, outside the timings.

    Each library's first reconstruction is still timed, TomoJAX's compilation included.
    """
    if tomojax:
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


def _timed[T](call: Callable[[], T], repeats: int) -> tuple[T, dict[str, Any]]:
    """``call``'s last result, its first call's time (``seconds``) and ``repeats`` more's."""
    start = time.perf_counter()
    result = call()
    times: dict[str, Any] = {"seconds": time.perf_counter() - start}
    warm = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        warm.append(time.perf_counter() - start)
    if warm:
        times |= {
            "warm_seconds": warm,
            "best_warm_seconds": min(warm),
            "median_warm_seconds": float(np.median(warm)),
        }
    return result, times


_ASTRA_METHODS = {
    "fbp": "FDK_CUDA",
    "cgls": "CGLS3D_CUDA",
    "fista": "accelerated NNLS (the reference's)",
}


def _astra(
    scan: tj.Scan, args: argparse.Namespace, iterations: int, *, gpus: int
) -> tuple[np.ndarray, dict]:
    """ASTRA's reconstruction of ``scan`` on ``gpus`` GPUs, and its record."""
    import astra

    astra.set_gpu_index(list(range(gpus)))

    def theirs_now() -> np.ndarray:
        # ASTRA gets the same (binned) data and geometry back from the scan.
        data, geometry, volume = scan.to_astra()
        return astra_reconstruct(data, geometry, volume, args.method, iterations)

    theirs, times = _timed(theirs_now, args.repeats)
    return theirs, {"method": _ASTRA_METHODS[args.method], **times}


def _tomojax(
    scan: tj.Scan, args: argparse.Namespace, iterations: int, *, gpus: int
) -> tuple[np.ndarray, dict]:
    """TomoJAX's reconstruction of ``scan`` on ``gpus`` GPUs (views shared), and its record."""
    import jax

    options: dict[str, Any] = {} if args.method == "fbp" else {"iterations": iterations}
    if args.method == "fista":
        options |= {"tv_weight": 0.0, "nonnegative": True}
    # One GPU takes the ordinary one-device path, as a user would run it.
    devices = jax.devices()[:gpus] if gpus > 1 else None

    def ours() -> tuple[tj.Reconstruction, np.ndarray]:
        result = tj.reconstruct(scan, args.method, devices=devices, **options)
        return result, np.asarray(result.volume)

    (result, volume), times = _timed(ours, args.repeats)
    if args.save is not None:
        tj.save(args.save, result)
    return volume, {"method": args.method, **options, **times}


def _geometry_summary(scan: tj.Scan) -> dict[str, Any]:
    """Each orbit's source and detector distances, and the largest recorded pose correction."""
    return {
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


def main() -> None:
    """Run the walnut comparison and print a JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("walnut", type=Path, help="A WalnutN directory from the data collection")
    parser.add_argument("--orbits", type=int, nargs="+", default=[2])
    parser.add_argument("--every", type=int, default=1, help="Use every Nth view")
    parser.add_argument("--method", choices=["fbp", "cgls", "fista"], default="fbp")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--budgets", type=int, nargs="+",
        help="Iteration budgets, each a fresh solve, smallest first (default: --iterations)",
    )  # fmt: skip
    parser.add_argument("--output", type=Path, help="Write the JSON here after each measurement")
    parser.add_argument("--cache", type=Path, help="Keep the decoded projections here (.npz)")
    parser.add_argument("--voxels-per-mm", type=int, default=10)
    parser.add_argument("--bin", type=int, default=1, help="Average N x N detector pixels")
    parser.add_argument("--libraries", nargs="+", choices=["tomojax", "astra"], default=["tomojax"])
    parser.add_argument("--save-volume", type=Path, help="Save each volume here (<library>.npy)")
    parser.add_argument("--gpus", type=int, default=1, help="GPUs each library may use")
    parser.add_argument("--repeats", type=int, default=0, help="Warm runs after the first")
    parser.add_argument("--save", type=Path, help="Save the TomoJAX reconstruction (.nxs)")
    args = parser.parse_args()
    # The scanner's TIFFs carry a malformed tag that tifffile reports and skips.
    logging.getLogger("tifffile").setLevel(logging.ERROR)

    _archive(args.output)
    start = time.perf_counter()
    data, vectors = load_orbits(args.walnut, args.orbits, args.every, cache=args.cache)
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
        **_geometry_summary(scan),
    }
    _start_gpu_runtimes(tomojax="tomojax" in args.libraries, astra="astra" in args.libraries)
    summary["gpus"] = args.gpus
    truth = reference(args.walnut)
    volumes = _reconstruct_all(scan, args, summary, truth)
    if "tomojax" in volumes and args.method == "fbp" and len(args.orbits) == 1:
        published = reference(args.walnut, f"fdk_pos{args.orbits[0]}")
        if published is not None:
            summary["tomojax_vs_published_fdk"] = compare(volumes["tomojax"], published)
    if len(volumes) == 2:
        ours, theirs = volumes["tomojax"], volumes["astra"]
        summary["tomojax_vs_astra_relative_l2"] = float(
            np.linalg.norm(ours - theirs) / np.linalg.norm(theirs)
        )
    _save(args.output, summary)
    print(json.dumps(summary, indent=2))


def _reconstruct_all(
    scan: tj.Scan, args: argparse.Namespace, summary: dict[str, Any], truth: np.ndarray | None
) -> dict[str, np.ndarray]:
    """Each library's reconstruction at each budget, recorded in ``summary`` as each finishes."""
    volumes = {}
    budgets = [0] if args.method == "fbp" else sorted(set(args.budgets or [args.iterations]))
    for library in args.libraries:
        run = _tomojax if library == "tomojax" else _astra
        results = []
        for budget in budgets:  # fresh solves: the error against the time each takes
            volumes[library], entry = run(scan, args, budget, gpus=args.gpus)
            if truth is not None:
                entry |= compare(volumes[library], truth)
            if args.method != "fbp":
                entry["iterations"] = budget
            results.append(entry)
            summary[library] = results[0] if args.method == "fbp" else results
            _save(args.output, summary)
        if args.save_volume is not None:
            args.save_volume.mkdir(parents=True, exist_ok=True)
            np.save(args.save_volume / f"{library}.npy", volumes[library])
    return volumes


def _archive(path: Path | None) -> None:
    """Move an earlier attempt's ``path`` aside as ``<name>.attempt-N.json``, keeping it."""
    if path is None or not path.exists():
        return
    attempt = 1
    while path.with_suffix(f".attempt-{attempt}.json").exists():
        attempt += 1
    path.replace(path.with_suffix(f".attempt-{attempt}.json"))


def _save(path: Path | None, summary: dict[str, Any]) -> None:
    """``summary`` as JSON at ``path``, whole or not at all."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2))
    temporary.replace(path)


def load_orbits(
    walnut: Path, orbits: list[int], every: int, *, cache: Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The orbits' ASTRA ``(rows, views, columns)`` data and vectors, kept in ``cache``."""
    if cache is not None and cache.exists():
        with np.load(cache) as stored:
            return stored["data"], stored["vectors"]
    parts = [load_orbit(walnut, orbit, every) for orbit in orbits]
    data = np.concatenate([d for d, _ in parts], axis=1)
    vectors = np.concatenate([v for _, v in parts])
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        partial = cache.with_name(f"partial-{cache.name}")
        with partial.open("wb") as stream:
            np.savez(stream, data=data, vectors=vectors)
        partial.replace(cache)
    return data, vectors


if __name__ == "__main__":
    main()
