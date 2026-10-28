r"""Time and score one full-view cone-beam reconstruction.

Each invocation is one fresh-process frontier point, followed by warm repeats.
Walnut scores use the authors' full_AGD_50 volume; other cases report a common
ASTRA data fit and optional full-volume differences, not physical truth.

    python bench/cone_matrix.py walnut3 ~/data/walnuts/Walnut1 \\
        --library astra --method ossart --iterations 1 --subsets 16 \\
        --norms recompute --output bench/results/walnut3-astra.json
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
from pathlib import Path
import resource
import tempfile
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

_STARTED = time.perf_counter()


def load_case(args: argparse.Namespace) -> tuple[Any, Any, np.ndarray]:
    """Load calibrated, all-view data and the declared physical output grid."""
    import tomojax as tj
    from tomojax.geometry.api import view_frames

    if args.case.startswith("walnut"):
        from walnut import load_orbits, volume_geometry as walnut_grid

        data, vectors = load_orbits(
            args.path, [2] if args.case == "walnut1" else [1, 2, 3], 1, binning=2
        )
        pg = {
            "type": "cone_vec",
            "DetectorRowCount": data.shape[0],
            "DetectorColCount": data.shape[2],
            "Vectors": vectors,
        }
        scan = tj.Scan.from_astra(data, pg, walnut_grid(10))
        return scan, scan.grid, vectors
    if args.case == "nikon":
        grid = tj.Grid(500, 500, 500, 0.1880371266, 0.1880371266, 0.1880371266)
    elif args.case == "zeiss":
        pitch = 2 * 0.04169269943237305
        grid = tj.Grid(512, 512, 512, pitch, pitch, pitch)
    else:
        grid = None
    frames = tj.load_frames(args.path, darks=0, grid=grid)
    scan = frames.corrected(binning=4 if args.case == "cuo" else 2)
    if grid is None:
        g = scan.grid
        grid = g.resampled((g.nx // 4, g.ny // 4, g.nz // 4))
    return scan, grid, view_frames(scan.geometry)


def astra_reconstruct(
    scan: Any, grid: Any, vectors: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, dict[str, int]]:
    """Run the selected ASTRA baseline, including initialization and normalization."""
    from _astra_cone import Operators, cgls, native, ordered_subsets, volume_geometry
    import numpy as np

    data, vg = scan.projections, volume_geometry(grid)
    initial = None
    if args.start == "fdk" or args.method == "fdk":
        turns = 3 if args.case == "walnut3" else 1
        edges = np.linspace(0, len(data), turns + 1, dtype=int)
        for first, stop in itertools.pairwise(edges):
            image = native(data[first:stop], vectors[first:stop], vg, method="fdk")
            image /= turns
            if initial is None:
                initial = image
            else:
                initial += image
        if args.method == "fdk":
            return initial, {}
    if args.method in {"cgls", "sirt"} and args.storage == "native":
        image = native(
            data, vectors, vg, method=args.method, iterations=args.iterations, initial=initial
        )
        return image, {}
    if args.method == "cgls":
        groups = [slice(i, min(i + args.batch, len(data))) for i in range(0, len(data), args.batch)]
    else:
        order = (
            np.argsort(scan.geometry.angles, kind="stable")
            if args.method == "osem"
            else np.arange(len(data))
        )
        count = 1 if args.method == "sirt" else args.subsets
        groups = [order[i::count] for i in range(count)]
    with Operators(data, vectors, vg, groups) as ops:
        if args.method == "cgls":
            image = cgls(
                ops,
                initial,
                iterations=args.iterations,
                resident_residual=args.storage == "device-residual",
            )
        else:
            image = ordered_subsets(
                ops,
                initial,
                method="ossart" if args.method == "sirt" else args.method,
                iterations=args.iterations,
                norms=args.norms,
            )
        return image, ops.work


def tomojax_reconstruct(scan: Any, grid: Any, args: argparse.Namespace) -> np.ndarray:
    """Call the public workflow with the declared solver settings."""
    import numpy as np

    import tomojax as tj
    from tomojax.recon import MLEMConfig, SIRTConfig

    method = {"fdk": "fbp", "ossart": "sirt", "osem": "mlem"}.get(args.method, args.method)
    options: dict[str, Any] = {}
    if args.method != "fdk":
        options.update(iterations=args.iterations, warm_start=args.start == "fdk")
    batch = {} if args.views_per_batch is None else {"views_per_batch": args.views_per_batch}
    if args.method in {"sirt", "ossart"}:
        options["config"] = SIRTConfig(
            lower_bound=0,
            subsets=1 if args.method == "sirt" else args.subsets,
            acceleration=args.acceleration,
            **batch,
        )
    elif args.method == "osem":
        options["config"] = MLEMConfig(subsets=args.subsets, **batch)
    elif args.method == "cgls" and batch:
        from tomojax.recon import CGLSConfig

        options["config"] = CGLSConfig(**batch)
    return np.asarray(tj.reconstruct(scan, method, grid=grid, **options).volume)


def image_difference(volume: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Full-volume differences with bounded FP64 reductions and no fitted scale."""
    import numpy as np

    if volume.shape != reference.shape:
        raise ValueError("Image comparison needs identical physical grids and shapes")
    energy = norm = maximum = 0.0
    for start in range(0, volume.shape[0], 8):
        truth = np.asarray(reference[start : start + 8], np.float64)
        difference = np.asarray(volume[start : start + 8], np.float64) - truth
        energy += float(np.sum(difference * difference))
        norm += float(np.sum(truth * truth))
        maximum = max(maximum, float(np.max(np.abs(difference))))
    return {
        "relative_l2": float(np.sqrt(energy / norm)),
        "rmse": float(np.sqrt(energy / volume.size)),
        "max_abs": maximum,
    }


def common_fit(
    volume: np.ndarray, measured: np.ndarray, vectors: np.ndarray, grid: Any
) -> dict[str, float]:
    """Score every original log projection with one common calibrated ASTRA FP."""
    from _astra_cone import Operators, volume_geometry
    import numpy as np

    groups = [slice(i, min(i + 64, len(measured))) for i in range(0, len(measured), 64)]
    energy = norm = 0.0
    with Operators(measured, vectors, volume_geometry(grid), groups) as ops:
        image = ops.upload(np.ascontiguousarray(volume))
        for part in ops.parts:
            predicted = ops.download(ops.forward(image, part)).transpose(1, 0, 2)
            truth = measured[part[0]]
            for start in range(0, len(truth), 8):
                data = np.asarray(truth[start : start + 8], np.float64)
                difference = np.asarray(predicted[start : start + 8], np.float64) - data
                energy += float(np.sum(difference * difference))
                norm += float(np.sum(data * data))
    return {
        "relative_l2": float(np.sqrt(energy / norm)),
        "rmse": float(np.sqrt(energy / measured.size)),
    }


def score(
    image: np.ndarray,
    measured: np.ndarray,
    vectors: np.ndarray,
    grid: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Report image quality separately from the timed solver."""
    import numpy as np

    record: dict[str, Any] = {}
    if args.case.startswith("walnut"):
        from walnut import compare, reference

        truth = reference(args.path)
        if truth is None or truth.shape != image.shape:
            raise ValueError("The authors' full_AGD_50 reference on the identical grid is required")
        record["image_error"] = compare(image, truth)
        del truth
    else:
        record["common_fit"] = common_fit(image, measured, vectors, grid)
    if args.compare_to is not None:
        record["image_difference"] = image_difference(
            image, np.load(args.compare_to, mmap_mode="r")
        )
    if args.save_volume is not None:
        np.save(args.save_volume, image)
    return record


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Time first and repeated solves, then score the final host volume."""
    from dataclasses import replace

    from _measure import GpuSampler
    from cupy.cuda import nvtx
    import numpy as np

    sampler = GpuSampler(period=0.02)
    try:
        scan, grid, vectors = load_case(args)
        measured = scan.projections
        if args.method == "osem":
            scan = replace(scan, projections=np.maximum(measured, 0))
        loaded = time.perf_counter()
        record: dict[str, Any] = {
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "grid": grid.to_dict(),
            "data_shape": list(measured.shape),
            "load_setup_seconds": loaded - _STARTED,
            "protocol": (
                "fresh numerical imports and compilation caches; warm repeats include setup "
                "and host output; filesystem cache uncontrolled"
            ),
            "seconds": [],
            "image_reference": "authors full_AGD_50" if args.case.startswith("walnut") else None,
        }
        for repeat in range(args.repeats + 1):
            nvtx.RangePush(f"cone-solve-{repeat}")
            begin = time.perf_counter()
            try:
                if args.library == "astra":
                    image, work = astra_reconstruct(scan, grid, vectors, args)
                    record["iterative_work"] = work
                else:
                    image = tomojax_reconstruct(scan, grid, args)
                record["seconds"].append(time.perf_counter() - begin)
            finally:
                nvtx.RangePop()
            if repeat == 0:
                record["cold_with_loading"] = time.perf_counter() - _STARTED
            if not np.isfinite(image).all():
                raise RuntimeError("Nonfinite reconstruction; timing is not a valid comparison")
            if repeat < args.repeats:
                del image
                gc.collect()
        ended = time.perf_counter()
        record["gpu"] = sampler.window(_STARTED, ended)
        record["host_peak_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        record.update(score(image, measured, vectors, grid, args))
        return record
    finally:
        sampler.close()


def main() -> None:
    """Run one frontier point and write its measured record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("walnut1", "walnut3", "nikon", "zeiss", "cuo"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--library", choices=("tomojax", "astra"), required=True)
    parser.add_argument(
        "--method", choices=("fdk", "cgls", "sirt", "ossart", "osem"), required=True
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--subsets", type=int, default=4)
    parser.add_argument("--start", choices=("fdk", "default"), default="fdk")
    parser.add_argument(
        "--storage",
        choices=("native", "host", "device-residual"),
        default="native",
        help="ASTRA recurrence storage; device-residual keeps CGLS residuals on the GPU",
    )
    parser.add_argument(
        "--norms",
        choices=("resident", "host", "recompute"),
        default="recompute",
        help="ASTRA ordered-subset column normalization",
    )
    parser.add_argument("--batch", type=int, default=512, help="ASTRA host CGLS views per batch")
    parser.add_argument("--acceleration", action="store_true", help="TomoJAX SIRT acceleration")
    parser.add_argument(
        "--views-per-batch", type=int, help="TomoJAX solver views per batch (default: solver's)"
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--compare-to", type=Path, help="Same-grid reference volume for full image diagnostics"
    )
    parser.add_argument("--save-volume", type=Path, help="Temporary volume for a paired comparison")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cone-compile-") as cache:
        for variable, subdir in (
            ("CUPY_CACHE_DIR", "cupy"),
            ("JAX_COMPILATION_CACHE_DIR", "jax"),
            ("TOMOJAX_JAX_CACHE_DIR", "jax"),
        ):
            os.environ[variable] = str(Path(cache) / subdir)
        record = run(args)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
