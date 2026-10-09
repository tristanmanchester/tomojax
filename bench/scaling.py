#!/usr/bin/env python3
r"""Time TomoJAX, ASTRA and TIGRE on one GPU and on several, on one cone-beam scan.

The scan is ``compare_cone.py``'s circular scan of five ellipsoids at ``--size``.
For each library and each GPU count in ``--gpus``, a child process of its own
(TIGRE refuses a GPU on which JAX holds memory) times, from host arrays to host
arrays:

- the forward projection and the backprojection: TomoJAX's exact transpose,
  ASTRA's voxel-driven backprojector and TIGRE's matched one;
- FDK;
- ``--iterations`` of CGLS from zero, the one iterative solver all three have.

Projection and FDK errors are against the analytic line integrals and the
voxelised phantom; CGLS errors are against the phantom, so they also show the
three libraries' discretisations. Times are the best of ``--repeats`` calls
after one warm-up call. TomoJAX shares the views among the GPUs (``devices=``);
ASTRA's ``set_gpu_index`` takes the GPUs, and TIGRE's ``gpuids``. A library
that cannot use more than one GPU for an operation still reports its time.

    uv run --no-sync python bench/scaling.py --size 256 --views 360 --gpus 1 \\
        --output bench/results/scaling-256.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from compare_cone import astra_vectors, best, make_case, relative, setup, tigre_geometry
import numpy as np

LIBRARIES = ("tomojax", "astra", "tigre")


def run_tomojax(
    case: dict[str, Any], volume: np.ndarray, data: np.ndarray, gpus: int, args: argparse.Namespace
) -> list[dict]:
    """TomoJAX with its views shared among ``gpus`` GPUs."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax

    import tomojax as tj

    n = case["size"]
    grid = tj.Grid(n, n, n, 1.0, 1.0, 1.0)
    detector = tj.Detector(case["nu"], case["nv"], 1.0, 1.0)
    beam = tj.ConeBeam(case["sod"], case["sdd"])
    geometry = tj.ConeGeometry(grid, detector, case["angles"], beam)
    devices = jax.devices("gpu")[:gpus]
    scan = tj.Scan(data, geometry)
    ops = {
        "forward": lambda: np.asarray(tj.project(geometry, volume, devices=devices)),
        "backproject": lambda: np.asarray(tj.backproject(geometry, data, devices=devices)),
        "fdk": lambda: np.asarray(tj.reconstruct(scan, "fbp", devices=devices).volume),
        "cgls": lambda: np.asarray(
            tj.reconstruct(scan, "cgls", iterations=args.iterations, devices=devices).volume
        ),
    }
    return _timed("tomojax", ops, volume, data, args.repeats)


def run_astra(
    case: dict[str, Any], volume: np.ndarray, data: np.ndarray, gpus: int, args: argparse.Namespace
) -> list[dict]:
    """ASTRA on ``gpus`` GPUs (``set_gpu_index``)."""
    import astra

    astra.set_gpu_index(list(range(gpus)))
    n = case["size"]
    vol_geom = astra.create_vol_geom(n, n, n, -n / 2, n / 2, -n / 2, n / 2, -n / 2, n / 2)
    proj_geom = astra.create_proj_geom("cone_vec", case["nv"], case["nu"], astra_vectors(case))
    avol = np.ascontiguousarray(volume.transpose(2, 1, 0))  # ASTRA's (z, y, x) volumes
    sino = np.ascontiguousarray(data.transpose(1, 0, 2))  # and (v, view, u) projections
    projected, back = np.empty_like(sino), np.empty_like(avol)

    def algorithm(name: str, iterations: int = 1) -> np.ndarray:
        sid = astra.data3d.create("-sino", proj_geom, sino)
        rid = astra.data3d.create("-vol", vol_geom, 0.0)
        cfg = astra.astra_dict(name)
        cfg["ProjectionDataId"], cfg["ReconstructionDataId"] = sid, rid
        alg = astra.algorithm.create(cfg)
        try:
            astra.algorithm.run(alg, iterations)
            return astra.data3d.get(rid).transpose(2, 1, 0)
        finally:
            astra.algorithm.delete(alg)
            astra.data3d.delete([sid, rid])

    projector = astra.create_projector("cuda3d", proj_geom, vol_geom)
    try:
        ops = {
            "forward": lambda: (
                astra.projector3d.direct_FP(projector, avol, out=projected).transpose(1, 0, 2)
            ),
            "backproject": lambda: astra.projector3d.direct_BP(projector, sino, out=back),
            "fdk": lambda: algorithm("FDK_CUDA"),
            "cgls": lambda: algorithm("CGLS3D_CUDA", args.iterations),
        }
        return _timed("astra", ops, volume, data, args.repeats)
    finally:
        astra.projector3d.delete(projector)


def run_tigre(
    case: dict[str, Any], volume: np.ndarray, data: np.ndarray, gpus: int, args: argparse.Namespace
) -> list[dict]:
    """TIGRE on ``gpus`` GPUs (``gpuids``)."""
    import tigre
    from tigre import algorithms
    from tigre.utilities.gpu import GpuIds

    geo, angles = tigre_geometry(case)
    ids = GpuIds()
    ids.devices = list(ids.devices)[:gpus]
    tvol = np.ascontiguousarray(volume.transpose(2, 1, 0))
    ops = {
        "forward": lambda: tigre.Ax(tvol, geo, angles, "interpolated", gpuids=ids),
        "backproject": lambda: tigre.Atb(data, geo, angles, "matched", gpuids=ids),
        "fdk": lambda: algorithms.fdk(data, geo, angles, gpuids=ids).transpose(2, 1, 0),
        "cgls": lambda: algorithms.cgls(
            data, geo, angles, args.iterations, gpuids=ids, verbose=False
        ).transpose(2, 1, 0),
    }
    return _timed("tigre", ops, volume, data, args.repeats)


def _timed(
    library: str, ops: dict[str, Any], volume: np.ndarray, data: np.ndarray, repeats: int
) -> list[dict]:
    """Each operation's best time, with its error where there is a reference for it."""
    records = []
    for operation, call in ops.items():
        result, seconds = best(call, repeats)
        record: dict[str, Any] = {"library": library, "operation": operation, "seconds": seconds}
        if operation == "forward":
            record["error"] = relative(result, data)
        elif operation in {"fdk", "cgls"}:
            record["error"] = relative(result, volume)
        records.append(record)
    return records


RUNNERS = {"tomojax": run_tomojax, "astra": run_astra, "tigre": run_tigre}


def main() -> int:
    """Run every library on every GPU count, each in a child process, and write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--views", type=int, default=360)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpus", type=int, nargs="+", default=[1])
    parser.add_argument("--libraries", nargs="+", choices=LIBRARIES, default=list(LIBRARIES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)  # library, GPU count
    args = parser.parse_args()
    case = setup(args.size, args.views)
    cache = args.output.with_suffix(".case.npz")
    if cache.exists():
        stored = np.load(cache)
        volume, data = stored["volume"], stored["data"]
    else:
        volume, data = make_case(case)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, volume=volume, data=data)
    if args.worker:
        library, gpus = args.worker[0], int(args.worker[1])
        records = RUNNERS[library](case, volume, data, gpus, args)
        print(json.dumps([{**r, "gpus": gpus} for r in records]))
        return 0
    records = []
    for gpus in args.gpus:
        for library in args.libraries:
            # JAX may only see the CPU in the other libraries' processes.
            env = dict(os.environ)
            if library != "tomojax":
                env["JAX_PLATFORMS"] = "cpu"
            command = [sys.executable, __file__, *sys.argv[1:], "--worker", library, str(gpus)]
            child = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
            if child.returncode:
                tail = child.stderr.strip().splitlines()[-1:] or ["no output"]
                records.append({"library": library, "gpus": gpus, "failed": tail[0]})
                print(f"{library:8s} {gpus} GPU  failed: {tail[0]}", flush=True)
                continue
            for r in json.loads(child.stdout.strip().splitlines()[-1]):
                records.append(r)
                error = f"  error {r['error']:.4f}" if "error" in r else ""
                print(f"{library:8s} {gpus} GPU  {r['operation']:12s} {r['seconds']:8.3f} s{error}")
    summary = {"case": {k: v for k, v in case.items() if k != "angles"}, "records": records}
    summary["iterations"] = args.iterations
    args.output.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
