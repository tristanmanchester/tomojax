#!/usr/bin/env python3
r"""Time TomoJAX, ASTRA and TIGRE on one GPU and on several, on one cone-beam scan.

The scan is ``compare_cone.py``'s circular scan of five ellipsoids at ``--size``.
For each library and each GPU count in ``--gpus``, a child process of its own
(TIGRE refuses a GPU on which JAX holds memory) times, from host arrays to host
arrays:

- the forward projection (TIGRE's interpolated and Siddon projectors both) and
  the backprojection: TomoJAX's exact transpose, ASTRA's voxel-driven
  backprojector and TIGRE's matched one;
- FDK;
- ``--iterations`` of CGLS from zero, the one iterative solver all three have.

Each operation records its first call in the process (compiling, for TomoJAX;
earlier operations have already started each library's GPU runtime) and every
one of its ``--repeats`` warm calls, its result's error (projections against the
analytic line integrals, volumes against the voxelised phantom) and the GPUs'
sampled memory and utilisation (see ``_measure.py``). CGLS records each solver's
own account: TomoJAX's effective iterations and termination, TIGRE's residual
history (its CGLS uses the Siddon projector, projects twice per iteration and
may stop early when the residual rises). ASTRA's CGLS3D_CUDA runs on one GPU
whatever ``set_gpu_index`` names; its projectors and FDK use them all.

Equal requested iteration budgets compare implementations, not solvers at
equal reconstruction quality (TIGRE also projects more per iteration); the
errors show how the quality differs.

Each worker writes its record to ``<output>.workers/`` after every operation,
and a failure (its exit status or ``--worker-timeout``, with its log's tail)
into the same record, so nothing measured is lost; the script then exits 1. A
rerun with the same ``--output`` reuses a finished worker only when its
settings and code match, and refuses one that differs. After every worker the
summary at ``--output`` holds every worker's record there, so several calls
(one library, GPU count or ``--operations`` each) build one summary.

    uv run --no-sync python bench/scaling.py --size 256 --views 360 --gpus 1 \\
        --output bench/results/scaling-256.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING, Any

from _measure import GpuSampler, environment, file_hashes, run_bounded
from compare_cone import astra_vectors, make_case, relative, setup, tigre_geometry
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

LIBRARIES = ("tomojax", "astra", "tigre")
type Op = Callable[[], tuple[np.ndarray, dict[str, Any]]]


def run_tomojax(case: dict[str, Any], volume: np.ndarray, data: np.ndarray, gpus: int) -> dict:
    """TomoJAX's operations, with its views shared among ``gpus`` GPUs."""
    import jax

    import tomojax as tj

    n = case["size"]
    grid = tj.Grid(n, n, n, 1.0, 1.0, 1.0)
    detector = tj.Detector(case["nu"], case["nv"], 1.0, 1.0)
    beam = tj.ConeBeam(case["sod"], case["sdd"])
    geometry = tj.ConeGeometry(grid, detector, case["angles"], beam)
    # One GPU takes the ordinary one-device path, as a user would run it.
    devices = jax.devices("gpu")[:gpus] if gpus > 1 else None
    scan = tj.Scan(data, geometry)
    for device in jax.devices("gpu")[:gpus]:  # start the runtime on each GPU, untimed
        jax.device_put(np.zeros(1, np.float32), device).block_until_ready()

    def cgls() -> tuple[np.ndarray, dict[str, Any]]:
        result = tj.reconstruct(scan, "cgls", iterations=ARGS.iterations, devices=devices)
        keys = ("effective_iterations", "termination", "projector_backend", "projector_model")
        values = {k: result.info.get(k) for k in keys}
        plain = (str, int, float, bool, type(None))
        info = {k: v if isinstance(v, plain) else type(v).__name__ for k, v in values.items()}
        return np.asarray(result.volume), info

    return {
        "forward": lambda: (np.asarray(tj.project(geometry, volume, devices=devices)), {}),
        "backproject": lambda: (np.asarray(tj.backproject(geometry, data, devices=devices)), {}),
        "fdk": lambda: (np.asarray(tj.reconstruct(scan, "fbp", devices=devices).volume), {}),
        "cgls": cgls,
    }


def run_astra(case: dict[str, Any], volume: np.ndarray, data: np.ndarray, gpus: int) -> dict:
    """ASTRA's operations on ``gpus`` GPUs (``set_gpu_index``)."""
    import astra

    astra.set_gpu_index(list(range(gpus)))
    astra.use_cuda()  # start the runtime, untimed
    n = case["size"]
    vol_geom = astra.create_vol_geom(n, n, n, -n / 2, n / 2, -n / 2, n / 2, -n / 2, n / 2)
    proj_geom = astra.create_proj_geom("cone_vec", case["nv"], case["nu"], astra_vectors(case))

    def algorithm(name: str, iterations: int = 1) -> np.ndarray:
        sid = astra.data3d.create("-sino", proj_geom, np.ascontiguousarray(data.transpose(1, 0, 2)))
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
    PROJECTORS.append(lambda: astra.projector3d.delete(projector))

    # Every call converts layouts and allocates its output, as the others' do.
    def forward() -> np.ndarray:
        avol = np.ascontiguousarray(volume.transpose(2, 1, 0))  # ASTRA's (z, y, x) volumes
        out = np.empty((case["nv"], case["views"], case["nu"]), np.float32)
        return astra.projector3d.direct_FP(projector, avol, out=out).transpose(1, 0, 2)

    def backproject() -> np.ndarray:
        sino = np.ascontiguousarray(data.transpose(1, 0, 2))  # and (v, view, u) projections
        out = np.empty((n, n, n), np.float32)
        return astra.projector3d.direct_BP(projector, sino, out=out)

    cgls_info = {"iterations": ARGS.iterations, "gpus_it_can_use": 1}
    return {
        "forward": lambda: (forward(), {}),
        "backproject": lambda: (backproject(), {}),
        "fdk": lambda: (algorithm("FDK_CUDA"), {}),
        "cgls": lambda: (algorithm("CGLS3D_CUDA", ARGS.iterations), cgls_info),
    }


def run_tigre(case: dict[str, Any], volume: np.ndarray, data: np.ndarray, gpus: int) -> dict:
    """TIGRE's operations on ``gpus`` GPUs (``gpuids``)."""
    import tigre
    from tigre import algorithms
    from tigre.algorithms.krylov_subspace_algorithms import CGLS
    from tigre.utilities.gpu import GpuIds

    geo, angles = tigre_geometry(case)
    ids = GpuIds()
    ids.devices = list(ids.devices)[:gpus]
    # Start the runtime on each GPU, untimed: a tiny projection.
    small, small_angles = tigre_geometry(setup(8, 4))
    tigre.Ax(np.ones((8, 8, 8), np.float32), small, small_angles, gpuids=ids)

    def cgls() -> tuple[np.ndarray, dict[str, Any]]:
        geo.check_geo(angles)
        alg = CGLS(data, geo, angles, ARGS.iterations, gpuids=ids, verbose=False)
        alg.run_main_iter()
        residuals = np.asarray(alg.l2l, np.float64).ravel()
        info = {
            # Iterations it computed a residual for, one it then undid included.
            "attempted_iterations": int(np.count_nonzero(residuals)),
            "residual_norms": residuals.tolist(),
            "projector": "Siddon forward, matched backprojection",
        }
        return alg.getres().transpose(2, 1, 0), info

    def tvol() -> np.ndarray:
        return np.ascontiguousarray(volume.transpose(2, 1, 0))

    return {
        "forward": lambda: (tigre.Ax(tvol(), geo, angles, "interpolated", gpuids=ids), {}),
        "forward_siddon": lambda: (tigre.Ax(tvol(), geo, angles, "Siddon", gpuids=ids), {}),
        "backproject": lambda: (tigre.Atb(data, geo, angles, "matched", gpuids=ids), {}),
        "fdk": lambda: (algorithms.fdk(data, geo, angles, gpuids=ids).transpose(2, 1, 0), {}),
        "cgls": cgls,
    }


RUNNERS = {"tomojax": run_tomojax, "astra": run_astra, "tigre": run_tigre}
ARGS: argparse.Namespace  # the command line, for the runners' iteration count
PROJECTORS: list[Callable[[], None]] = []  # cleanup after a worker's operations


def _worker(
    library: str, gpus: int, case: dict[str, Any], args: argparse.Namespace, out: Path
) -> None:
    """Time ``library``'s operations on ``gpus`` GPUs, writing ``out`` after each one."""
    sampler = GpuSampler()  # before the library touches a GPU, so its memory counts
    volume, data = _load_case(args.case, case)
    record: dict[str, Any] = {
        "library": library,
        "gpus": gpus,
        "settings": _settings(args),
        "complete": False,
        "operations": [],
    }
    _write(out, record)
    try:
        ops: dict[str, Op] = RUNNERS[library](case, volume, data, gpus)
        for operation, call in ops.items():
            if args.operations and operation not in args.operations:
                continue
            if operation == "cgls" and args.cgls_repeats is not None:
                repeats = args.cgls_repeats
            else:
                repeats = args.repeats
            record["operations"].append(_time(operation, call, repeats, sampler, volume, data))
            _write(out, record)
        record["complete"] = True
        _write(out, record)
    finally:
        for cleanup in PROJECTORS:
            cleanup()
        sampler.close()


def _time(
    operation: str,
    call: Op,
    repeats: int,
    sampler: GpuSampler,
    volume: np.ndarray,
    data: np.ndarray,
) -> dict[str, Any]:
    """One operation's first call, ``repeats`` warm calls, GPU use, error and diagnostics."""
    start = time.perf_counter()
    result, info = call()
    first_stop = time.perf_counter()
    warm = []
    for _ in range(repeats):
        begin = time.perf_counter()
        result, info = call()
        warm.append(time.perf_counter() - begin)
    stop = time.perf_counter()
    record: dict[str, Any] = {
        "operation": operation,
        "first_call_seconds": first_stop - start,
        "warm_seconds": warm,
        "best_seconds": min(warm, default=first_stop - start),
        "median_seconds": float(np.median(warm)) if warm else first_stop - start,
        "gpus_first_call": sampler.window(start, first_stop),
        "gpus_warm": sampler.window(first_stop, stop) if warm else None,
        "solver": info,
    }
    if operation.startswith("forward"):
        record["error"] = relative(result, data)
    elif operation in {"fdk", "cgls"}:
        record["error"] = relative(result, volume)
    return record


def _write(path: Path, value: object) -> None:
    """``value`` as JSON at ``path``, whole or not at all."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    """What a worker's measurements depend on, its code included."""
    here = Path(__file__).parent
    code = [here / "scaling.py", here / "_measure.py", here / "compare_cone.py"]
    return {
        "size": args.size,
        "views": args.views,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "cgls_repeats": args.cgls_repeats,
        "operations": sorted(args.operations or []),
        "case": [args.case.name, args.case.stat().st_size],
        "code": file_hashes(code),
        "machine": _provenance(),
    }


def _provenance() -> dict[str, Any]:
    """The TomoJAX build, every installed package's version and the GPUs measured on."""
    import hashlib
    from importlib import metadata

    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
    return {
        "tomojax_commit": os.environ.get("TOMOJAX_COMMIT"),
        "tomojax": metadata.version("tomojax"),
        "packages_sha256": hashlib.sha256("\n".join(packages).encode()).hexdigest(),
        "gpus": environment()["gpus"],
    }


def _load_case(path: Path, case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """The cached phantom and data, checked against ``case``."""
    with np.load(path) as stored:
        volume, data = stored["volume"], stored["data"]
    n = case["size"]
    expected = {"volume": (n, n, n), "data": (case["views"], case["nv"], case["nu"])}
    for name, array in (("volume", volume), ("data", data)):
        if array.shape != expected[name] or array.dtype != np.float32:
            raise ValueError(f"{path}: {name} is {array.dtype}{array.shape}, not {expected[name]}")
        if not np.isfinite(array).all():
            raise ValueError(f"{path}: {name} is not finite")
    return volume, data


def main() -> int:
    """Run every library on every GPU count, each in a child process, and write the JSON."""
    global ARGS  # noqa: PLW0603
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--views", type=int, default=360)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3, help="Warm calls of each operation")
    parser.add_argument("--cgls-repeats", type=int, help="Warm CGLS calls (default --repeats)")
    parser.add_argument("--gpus", type=int, nargs="+", default=[1])
    parser.add_argument("--libraries", nargs="+", choices=LIBRARIES, default=list(LIBRARIES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=Path, help="Phantom and data cache (default: by --output)")
    parser.add_argument("--worker-timeout", type=float, help="Seconds before a worker is stopped")
    parser.add_argument("--operations", nargs="+", help="Time only these operations (default all)")
    parser.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)  # library, GPU count
    ARGS = args = parser.parse_args()
    case = setup(args.size, args.views)
    args.case = args.case or args.output.with_suffix(".case.npz")
    workers = args.output.with_suffix(".workers")
    if args.worker:
        library, gpus = args.worker[0], int(args.worker[1])
        _worker(library, gpus, case, args, _worker_file(workers, library, gpus, args))
        return 0
    if not args.case.exists():
        volume, data = make_case(case)
        args.case.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.case, volume=volume, data=data)
    workers.mkdir(parents=True, exist_ok=True)
    summary = {
        "case": {k: v for k, v in case.items() if k != "angles"},
        "iterations": args.iterations,
        "repeats": args.repeats,
        "cgls_repeats": args.cgls_repeats,
        "environment": environment(),
        "records": [],
    }
    failed = False
    for gpus in args.gpus:
        for library in args.libraries:
            record = _run_worker(library, gpus, args, workers)
            failed |= "failed" in record
            # Every worker's record so far, this call's or an earlier one's.
            summary["records"] = [json.loads(f.read_text()) for f in sorted(workers.glob("*.json"))]
            _write(args.output, summary)
            if record.get("failed") == "terminated":  # by our own supervisor: stop here
                return 1
    return 1 if failed else 0


def _worker_file(workers: Path, library: str, gpus: int, args: argparse.Namespace) -> Path:
    tag = "+".join(sorted(args.operations)) if args.operations else "all"
    return workers / f"{library}-{gpus}-{tag}.json"


def _run_worker(library: str, gpus: int, args: argparse.Namespace, workers: Path) -> dict:
    """``library`` on ``gpus`` GPUs in a child process, or its record from an earlier run.

    A failed worker's record, measurements so far included, says how it failed.
    """
    out = _worker_file(workers, library, gpus, args)
    if out.exists():
        earlier = json.loads(out.read_text())
        if earlier.get("settings") != _settings(args):
            raise SystemExit(f"{out} was measured with other settings or code; use a new --output")
        if earlier.get("complete"):
            return earlier
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    if library != "tomojax":
        env["JAX_PLATFORMS"] = "cpu"  # JAX may only see the CPU in the others' processes
    command = [sys.executable, __file__, *sys.argv[1:], "--worker", library, str(gpus)]
    log = out.with_suffix(".log")
    status = run_bounded(command, log=log, timeout=args.worker_timeout, env=env)
    if out.exists():
        record = json.loads(out.read_text())
    else:  # it failed before its first checkpoint
        record = {"library": library, "gpus": gpus, "settings": _settings(args), "operations": []}
    if status != "exit 0" or not record.get("complete"):
        record["failed"] = status
        record["log_tail"] = log.read_text(errors="replace").strip().splitlines()[-20:]
        _write(out, record)
    for op in record.get("operations", []):
        error = f"  error {op['error']:.4f}" if "error" in op else ""
        print(f"{library:8s} {gpus} GPU  {op['operation']:14s} {op['best_seconds']:8.3f} s{error}")
    if "failed" in record:
        print(f"{library:8s} {gpus} GPU  failed: {record['failed']}", flush=True)
    return record


if __name__ == "__main__":
    raise SystemExit(main())
