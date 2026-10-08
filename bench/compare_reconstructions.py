#!/usr/bin/env python3
"""Compare complete solves at fixed quality gates in isolated worker processes.

Each candidate starts from zero and runs its entire iteration budget in one call.
ASTRA CGLS resets its internal state on every run(), so checkpointing repeated
run() calls would silently change the algorithm. Report the full budget search
cost separately from repeated solves at the first accepted budget. This initial
Gaussian suite is not evidence of broad scientific reconstruction quality.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

SUITE = "gaussian-v1"
BUDGETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
TARGETS = {"parallel": 0.03, "anisotropic": 0.03, "lamino": 0.10}
SUITE_TARGETS = {
    SUITE: TARGETS,
    "structured-v1": {"parallel": 0.15, "anisotropic": 0.15, "lamino": 0.30},
    "structured-noisy-v1": {"parallel": 0.18, "anisotropic": 0.18, "lamino": 0.32},
}
METHODS = (
    "tomojax_fourier_cupy",
    "tomojax_fbp_pallas",
    "tomojax_fbp_host_pallas",
    "tomojax_fbp_joseph_cgls_pallas",
    "astra_fbp3d_cupy",
    "astra_fbp2d",
    "astra_fbp_cgls",
    "tigre_fbp",
    "tomojax_fista",
    "tomojax_cgls_jax",
    "tomojax_cgls_pallas",
    "tomojax_multires_cgls_pallas",
    "tomojax_joseph_cgls_jax",
    "tomojax_joseph_cgls_pallas",
    "tomojax_multires_joseph_cgls_pallas",
    "astra_cgls",
    "astra_sirt",
    "tigre_cgls",
)
DIRECT_METHODS = {
    "tomojax_fourier_cupy",
    "tomojax_fbp_pallas",
    "tomojax_fbp_host_pallas",
    "astra_fbp3d_cupy",
    "astra_fbp2d",
    "tigre_fbp",
}


def generate_fixture(path: Path, size: int, views: int, kind: str, suite: str = SUITE) -> None:
    """Generate truth outside timed workers, without retaining a GPU context."""
    from compare_projectors import make_case
    import numpy as np

    case = make_case(size, views, kind, phantom="gaussian" if suite == SUITE else "structured")
    data = case.analytic
    if suite == "structured-noisy-v1":
        rng = np.random.Generator(np.random.PCG64(7019))
        sigma = 0.01 * np.sqrt(np.mean(np.square(data, dtype=np.float64)))
        data = (data + sigma * rng.standard_normal(data.shape)).astype(np.float32)
    np.savez(
        path,
        name=case.name,
        grid=json.dumps(case.grid.to_dict()),
        detector=json.dumps(case.detector.to_dict()),
        poses=case.poses,
        angles=case.angles_deg,
        truth=case.volume,
        data=data,
        suite=suite,
    )


def load_fixture(path: Path) -> Any:
    """Read exactly the same host arrays for every competing method."""
    import numpy as np
    from physical_case import Case

    from tomojax.core.geometry.base import Detector, Grid

    with np.load(path) as data:
        return Case(
            str(data["name"]),
            Grid(**json.loads(str(data["grid"]))),
            Detector(**json.loads(str(data["detector"]))),
            data["poses"],
            data["angles"],
            data["truth"],
            data["data"],
        )


def multires_schedule(
    shape: tuple[int, int, int], budget: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Use the frozen geometry-independent coarse-to-fine comparison policy."""
    factor = math.ceil(max(shape) / 32)
    if factor == 1 or budget < 2:
        return (1,), (budget,)
    coarse = 7 * budget // 8
    return (factor, 1), (coarse, budget - coarse)


def solve_tomojax(
    case: Any, iterations: int, batch: int, method: str, *, fourier_slices: int = 16
) -> tuple[Any, dict]:
    """Run a public TomoJAX solver with unregularized, unconstrained least squares."""
    import numpy as np

    from tomojax.geometry import LaminographyGeometry, ParallelGeometry

    geometry = (
        LaminographyGeometry(case.grid, case.detector, case.angles_deg, tilt_deg=30)
        if case.name.startswith("lamino-")
        else ParallelGeometry(case.grid, case.detector, case.angles_deg)
    )
    if method == "tomojax_fourier_cupy":
        from tomojax.recon import FourierConfig, fourier_reconstruct

        result = fourier_reconstruct(
            geometry,
            case.grid,
            case.detector,
            case.analytic,
            config=FourierConfig(slices_per_batch=fourier_slices, backend="cupy"),
        )
        return result, {
            "backend": "CuPy_Fourier_host_slabs",
            "radial_interpolation": "kaiser_bessel_width_6",
            "angular_interpolation": "linear",
            "frequency_cutoff": "detector_nyquist_disk",
            "slices_per_batch": fourier_slices,
            "regulariser": "none",
            "positivity": False,
        }
    # Import JAX-based solvers only when needed, as a Fourier-only script would.
    import jax.numpy as jnp

    from tomojax.recon import CGLSConfig, FBPConfig, FistaConfig, cgls, cgls_multires, fbp, fista_tv

    if method == "tomojax_fbp_host_pallas":
        from tomojax.recon import FBPHostConfig, fbp_host

        result = fbp_host(
            geometry,
            case.grid,
            case.detector,
            case.analytic,
            config=FBPHostConfig(slices_per_batch=16, views_per_batch=32, backprojector="pallas"),
        )
        return result, {
            "backend": "Pallas_FBP_host_slabs",
            "filter": "ramp",
            "filter_support": "volume",
            "slices_per_batch": 16,
            "views_per_batch": min(32, len(case.poses)),
            "regulariser": "none",
            "positivity": False,
        }
    if method == "tomojax_fbp_pallas":
        result = fbp(
            geometry,
            case.grid,
            case.detector,
            jnp.asarray(case.analytic),
            config=FBPConfig(backprojector="pallas"),
        )
        return np.asarray(result), {
            "backend": "Pallas_FBP",
            "filter": "ramp",
            "filter_support": "volume",
            "regulariser": "none",
            "positivity": False,
        }
    if method in {"tomojax_multires_cgls_pallas", "tomojax_multires_joseph_cgls_pallas"}:
        factors, budgets = multires_schedule((case.grid.nx, case.grid.ny, case.grid.nz), iterations)
        result, info = cgls_multires(
            geometry,
            case.grid,
            case.detector,
            case.analytic,
            factors=factors,
            iterations_per_level=budgets,
            config=CGLSConfig(
                rtol=0.0,
                views_per_batch=batch,
                projector_backend="pallas",
                projector_model="joseph" if "joseph" in method else "ray",
            ),
        )
        return np.asarray(result), {**info, "regulariser": "none", "positivity": False}
    if method.startswith(("tomojax_cgls_", "tomojax_joseph_cgls_", "tomojax_fbp_joseph_cgls_")):
        initial = (
            fbp(
                geometry,
                case.grid,
                case.detector,
                jnp.asarray(case.analytic),
                config=FBPConfig(backprojector="pallas"),
            )
            if method == "tomojax_fbp_joseph_cgls_pallas"
            else None
        )
        result, info = cgls(
            geometry,
            case.grid,
            case.detector,
            jnp.asarray(case.analytic),
            init_x=initial,
            config=CGLSConfig(
                iterations=iterations,
                rtol=0.0,
                views_per_batch=batch,
                projector_backend=method.rsplit("_", 1)[-1],
                projector_model="joseph" if "joseph" in method else "ray",
            ),
        )
        return np.asarray(result), {
            **info,
            "regulariser": "none",
            "positivity": False,
            "initialization": "fbp" if initial is not None else "zero",
        }
    config = FistaConfig(
        iterations=iterations,
        tv_weight=0,
        regulariser="huber_tv",
        nonnegative=False,
        views_per_batch=batch,
        gather_dtype="fp32",
    )
    result, info = fista_tv(
        geometry, case.grid, case.detector, jnp.asarray(case.analytic), config=config
    )
    return np.asarray(result), {
        "backend": "jax",
        "effective_iters": info["effective_iterations"],
        "lipschitz": info["lipschitz"],
        "views_per_batch": batch,
        "regulariser": "none",
        "positivity": False,
    }


def solve_astra(case: Any, iterations: int, method: str) -> tuple[Any, dict]:
    """Include native layout conversion, geometry, allocation, transfers and cleanup."""
    import astra
    from direct_reconstruction import astra_geometries
    import numpy as np

    if method in {"astra_fbp3d_cupy", "astra_fbp2d"}:
        from direct_reconstruction import astra_fbp2d, astra_fbp3d

        return astra_fbp3d(case) if method == "astra_fbp3d_cupy" else astra_fbp2d(case)

    initial = 0.0
    if method == "astra_fbp_cgls":
        from direct_reconstruction import astra_fbp3d

        volume, _ = astra_fbp3d(case)
        # Native CGLS3D_CUDA requires float32 host data, unlike direct_BP.
        # Include the required download/upload handoff in complete workflow cost.
        initial = np.ascontiguousarray(volume.transpose(2, 1, 0))

    data_ids = []
    algorithm_id = None
    try:
        vg, pg = astra_geometries(case)
        data = np.ascontiguousarray(case.analytic.transpose(1, 0, 2))
        projection_id = astra.data3d.create("-sino", pg, data)
        data_ids.append(projection_id)
        volume_id = astra.data3d.create("-vol", vg, initial)
        data_ids.append(volume_id)
        algorithm = "CGLS3D_CUDA" if method in {"astra_cgls", "astra_fbp_cgls"} else "SIRT3D_CUDA"
        config = astra.astra_dict(algorithm)
        config["ProjectionDataId"] = projection_id
        config["ReconstructionDataId"] = volume_id
        algorithm_id = astra.algorithm.create(config)
        astra.algorithm.run(algorithm_id, iterations=iterations)
        volume = np.ascontiguousarray(astra.data3d.get(volume_id).transpose(2, 1, 0))
        return volume, {
            "backend": algorithm,
            "regulariser": "none",
            "positivity": False,
            "initialization": "fbp" if method == "astra_fbp_cgls" else "zero",
        }
    finally:
        if algorithm_id is not None:
            astra.algorithm.delete(algorithm_id)
        astra.data3d.delete(data_ids)


def solve_tigre(case: Any, iterations: int, method: str = "tigre_cgls") -> tuple[Any, dict]:
    """Run CERN TIGRE's public CGLS with the validated parallel adapter."""
    import numpy as np
    import tigre
    from tigre.algorithms import cgls, fbp

    padding = 0
    if method == "tigre_fbp":
        from direct_reconstruction import extend_filter_support

        case, padding = extend_filter_support(case)
    grid, detector = case.grid, case.detector
    geo = tigre.geometry(mode="parallel", nVoxel=np.asarray([grid.nz, grid.ny, grid.nx]))
    geo.dVoxel = np.asarray([grid.vz, grid.vy, grid.vx])
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    geo.nDetector = np.asarray([detector.nv, detector.nu])
    geo.dDetector = np.asarray([detector.dv, detector.du])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.accuracy = 1.0
    angles = np.deg2rad(-90.0 - case.angles_deg).astype(np.float32)
    result = (
        fbp(case.analytic.copy(), geo, angles, filter="ram_lak", verbose=False)
        if method == "tigre_fbp"
        else cgls(case.analytic.copy(), geo, angles, iterations, verbose=False)
    )
    return np.ascontiguousarray(result.transpose(2, 1, 0)), {
        "backend": "TIGRE_FBP" if method == "tigre_fbp" else "CGLS_Siddon_matched",
        "regulariser": "none",
        "positivity": False,
        **(
            {"filter_support": "volume", "detector_zero_padding_each_side": padding}
            if method == "tigre_fbp"
            else {}
        ),
    }


def quality(volume: Any, truth: Any, target: float) -> dict[str, Any]:
    """Check full-volume physical amplitude with bounded FP64 work buffers."""
    import numpy as np

    if np.shape(volume) != np.shape(truth):
        raise ValueError("quality requires matching volume and truth shapes")
    squared_error, squared_truth = 0.0, 0.0
    # Buffer conversion and reduction instead of allocating several full FP64
    # volumes. Verification is timed equally for every competing solver.
    with np.nditer(
        [volume, truth],
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=[["readonly"], ["readonly"]],
        op_dtypes=[np.float64, np.float64],
        casting="same_kind",
        order="K",
        buffersize=262144,
    ) as iterator:
        for value, reference in iterator:
            if not np.isfinite(value).all():
                return {"finite": False, "volume_relative_l2": None, "accepted": False}
            difference = value - reference
            squared_error += float(np.einsum("i,i->", difference, difference))
            squared_truth += float(np.einsum("i,i->", reference, reference))
    error = float(np.sqrt(np.divide(squared_error, squared_truth)))
    return {"finite": True, "volume_relative_l2": error, "accepted": error <= target}


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    """Find the first passing fixed budget, then repeat that entire solve."""
    import numpy as np

    case = load_fixture(args.fixture)
    suite = getattr(args, "suite", SUITE)
    target = SUITE_TARGETS[suite][args.kind]
    record: dict[str, Any] = {
        "case": case.name,
        "method": args.method,
        "suite": suite,
        "target_relative_l2": target,
        "timing_scope": "host_to_host_complete_solve_including_quality_verification",
        "startup_ms": (time.perf_counter() - args.started) * 1000,
        "search": [],
        "repeats": [],
        "status": "target_not_reached",
        "budget_kind": "single_pass" if args.method in DIRECT_METHODS else "iterations",
    }
    if args.method in {"tigre_cgls", "tigre_fbp", "astra_fbp2d"} and args.kind != "parallel":
        return {
            **record,
            "status": "unsupported_comparison",
            "reason": "This adapter is validated only for centred isotropic parallel scans",
        }
    if args.method in {"tomojax_fourier_cupy", "tomojax_fbp_host_pallas"} and args.kind == "lamino":
        return {
            **record,
            "status": "unsupported_comparison",
            "reason": "This direct TomoJAX inverse requires built-in ParallelGeometry",
        }

    def evaluate(iterations: int) -> dict:
        start = time.perf_counter()
        if args.method.startswith("tomojax_"):
            options = (
                {"fourier_slices": args.fourier_slices}
                if args.method == "tomojax_fourier_cupy"
                else {}
            )
            volume, info = solve_tomojax(case, iterations, args.batch, args.method, **options)
        elif args.method.startswith("astra_"):
            volume, info = solve_astra(case, iterations, args.method)
        else:
            volume, info = solve_tigre(case, iterations, args.method)
        solve_ms = (time.perf_counter() - start) * 1000
        result = quality(volume, case.volume, target)
        result["accepted"] &= info.get("termination") != "numerical_breakdown"
        return {
            "iterations": iterations,
            "solve_ms": solve_ms,
            "verified_ms": (time.perf_counter() - start) * 1000,
            **result,
            "info": info,
        }

    for budget in (1,) if args.method in DIRECT_METHODS else BUDGETS:
        if budget > args.max_iters:
            break
        row = evaluate(budget)
        record["search"].append(row)
        print(f"{case.name} {args.method} {row}", flush=True)
        # Retain partial results if a later candidate crashes or times out.
        write_result(args.output, record)
        if row["accepted"]:
            record["selected_iterations"] = budget
            record["cold_search_verified_ms"] = record["startup_ms"] + sum(
                run["verified_ms"] for run in record["search"]
            )
            record["repeats"] = [evaluate(budget) for _ in range(args.repeats)]
            record["selected_budget_warm_verified_median_ms"] = float(
                np.median([run["verified_ms"] for run in record["repeats"]])
            )
            record["status"] = (
                "accepted"
                if all(run["accepted"] for run in record["repeats"])
                else "unstable_quality"
            )
            break
    if record["status"] == "target_not_reached" and record["search"]:
        # Failed methods still need measured warm attempts. These are NOT
        # accepted-result timings and cannot participate in a speedup ratio.
        last_budget = record["search"][-1]["iterations"]
        record["failed_budget_iterations"] = last_budget
        record["failed_budget_repeats"] = []
        for _ in range(args.repeats):
            record["failed_budget_repeats"].append(evaluate(last_budget))
            write_result(args.output, record)
        record["failed_budget_warm_verified_median_ms"] = float(
            np.median([run["verified_ms"] for run in record["failed_budget_repeats"]])
        )
    record["loaded_solver_modules"] = [
        name for name in ("jax", "jaxlib", "astra", "tigre", "cupy") if name in sys.modules
    ]
    return record


def read_peak_memory(path: Path, pid: int) -> dict:
    """Read nvidia-smi's sampled per-process accounting, never whole-device usage."""
    samples = []
    if path.exists():
        for line in path.read_text().splitlines():
            fields = line.split(",")
            if len(fields) == 2 and fields[0].strip() == str(pid):
                with suppress(ValueError):
                    samples.append(float(fields[1]))
    return {
        "sampled_process_peak_mib": max(samples) if samples else None,
        "memory_samples": len(samples),
        "requested_memory_poll_ms": 10,
        "memory_scope": "entire_isolated_worker_including_imports_search_and_repeats",
        "memory_limitation": "sampled accounting can miss short-lived allocations",
    }


def isolated_run(command: list[str], output: Path, timeout: float) -> dict:
    """Run one method without retaining another library's CUDA allocator/context."""
    log = output.with_suffix(".log")
    memory_log = output.with_suffix(".memory.csv")
    monitor = None
    with log.open("w") as stream, memory_log.open("w") as memory:
        with suppress(FileNotFoundError):
            monitor = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                    "-lms",
                    "10",
                ],
                stdout=memory,
                stderr=subprocess.DEVNULL,
            )
        worker = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        timed_out = False
        try:
            code = worker.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            worker.kill()
            code = worker.wait()
            timed_out = True
        finally:
            if monitor is not None:
                monitor.terminate()
                monitor.wait()
    record = json.loads(output.read_text()) if output.exists() else {}
    if timed_out or code:
        record.update(status="timeout" if timed_out else "execution_failed", exit_code=code)
        record["log_tail"] = log.read_text()[-4000:]
    return {**record, **read_peak_memory(memory_log, worker.pid)}


def environment() -> dict:
    """Record software and hardware without creating a parent CUDA context."""
    versions = {}
    for package in ("tomojax", "jax", "jaxlib", "numpy", "astra-toolbox", "tigre", "cupy-cuda12x"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    gpu = None
    with suppress(FileNotFoundError, subprocess.CalledProcessError):
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
    source_hash = hashlib.sha256()
    source_root = Path(__file__).resolve().parents[1]
    for path in sorted(
        [*(source_root / "src").rglob("*.py"), *(source_root / "bench").glob("*.py")]
    ):
        source_hash.update(str(path.relative_to(source_root)).encode() + b"\0" + path.read_bytes())
    return {
        "versions": versions,
        "platform": platform.platform(),
        "gpu": gpu,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "jax_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "execution_environment": {
            name: os.environ.get(name)
            for name in (
                "JAX_PLATFORMS",
                "JAX_DEFAULT_MATMUL_PRECISION",
                "JAX_ENABLE_X64",
                "CUDA_VISIBLE_DEVICES",
                "XLA_FLAGS",
                "JAX_COMPILATION_CACHE_DIR",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "source_tree_sha256": source_hash.hexdigest(),
        "source_root": str(source_root),
    }


def write_result(path: Path, payload: dict) -> None:
    """Replace a record atomically so interruption preserves the previous result."""
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def resume_payload(path: Path, expected: dict) -> dict:
    """Resume only identical cases, settings, source and execution environment."""
    previous = json.loads(path.read_text())
    for key in ("suite", "standard_suite", "environment"):
        if previous.get(key) != expected[key]:
            raise ValueError(f"Cannot resume: {key} differs from the retained record")
    for key in (
        "sizes",
        "geometries",
        "methods",
        "views",
        "batch",
        "fourier_slices",
        "repeats",
        "max_iters",
        "timeout",
    ):
        if previous.get("arguments", {}).get(key) != expected["arguments"][key]:
            raise ValueError(f"Cannot resume: argument {key} differs from the retained record")
    arguments = expected["arguments"]
    allowed = {
        (f"{kind}-{size}-{arguments['views']}", method)
        for kind in arguments["geometries"]
        for size in arguments["sizes"]
        for method in arguments["methods"]
    }
    seen = set()
    records = previous.get("records")
    if not isinstance(records, list):
        raise ValueError("Cannot resume: records must be a list")
    for record in records:
        key = (record.get("case"), record.get("method"))
        if key not in allowed or key in seen:
            raise ValueError("Cannot resume: duplicate or unexpected case/method record")
        seen.add(key)
        status = record.get("status")
        if status not in {
            "accepted",
            "target_not_reached",
            "unsupported_comparison",
            "timeout",
            "execution_failed",
            "unstable_quality",
        }:
            raise ValueError("Cannot resume: record has no recognized terminal status")
        if status in {"accepted", "unstable_quality", "target_not_reached"}:
            failed_budget = status == "target_not_reached"
            repeats = record.get("failed_budget_repeats" if failed_budget else "repeats", [])
            if len(repeats) != arguments["repeats"]:
                scope = (
                    "failed-budget warm attempts" if failed_budget else "selected-budget repeats"
                )
                raise ValueError(f"Cannot resume: {scope} are incomplete")
            if status == "accepted" and not all(row.get("accepted") is True for row in repeats):
                raise ValueError("Cannot resume: accepted record contains failed repeats")
    return previous


def parse_arguments() -> argparse.Namespace:
    """Parse and validate the sweep or internal-worker arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=list(SUITE_TARGETS), default=SUITE)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--geometries", choices=list(TARGETS), nargs="+", default=list(TARGETS))
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    parser.add_argument("--views", type=int, default=180)
    parser.add_argument("--batch", type=int, default=16, help="Iterative view-batch size")
    parser.add_argument(
        "--fourier-slices", type=int, default=16, help="Axial slab size for the Fourier method"
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--max-iters", type=int, choices=BUDGETS, default=256)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--resume", action="store_true", help="Continue an identical saved sweep")
    parser.add_argument(
        "--output", type=Path, default=Path("bench/results/reconstruction-quality.json")
    )
    parser.add_argument("--fixture", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--generate", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--kind", choices=list(TARGETS), help=argparse.SUPPRESS)
    parser.add_argument("--started", type=float, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        min(*args.sizes, args.views, args.batch, args.fourier_slices, args.repeats) < 1
        or min(args.sizes) < 8
    ):
        parser.error("sizes must be >=8; views, batch, Fourier slices and repeats must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    if any(
        len(values) != len(set(values)) for values in (args.sizes, args.geometries, args.methods)
    ):
        parser.error("sizes, geometries and methods must not contain duplicates")
    if args.resume and (args.generate or args.method):
        parser.error("--resume applies to the complete sweep, not individual workers")
    return args


def main() -> int:
    """Write every case and failed comparison, returning failure if coverage is incomplete."""
    args = parse_arguments()
    if args.generate:
        generate_fixture(args.fixture, args.sizes[0], args.views, args.kind, args.suite)
        return 0
    if args.method:
        record = run_worker(args)
        write_result(args.output, record)
        return 0

    payload = {
        "suite": args.suite,
        "standard_suite": args.views == 180 and args.sizes == [64, 128, 256],
        "environment": environment(),
        "arguments": {**vars(args), "output": str(args.output)},
        "interpretation": (
            "Cold search includes startup and all failed smaller budgets. Repeated selected-budget "
            "times exclude offline budget selection and cannot alone satisfy the end-to-end goal. "
            "Memory covers the entire worker, not only its selected-budget solve. "
            "Fixtures load without importing JAX/Pallas; workers import their own solvers. "
            "Failed methods retain warm attempts at their last budget, never accepted-result times."
        ),
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        try:
            payload = resume_payload(args.output, payload)
        except (OSError, ValueError, TypeError, AttributeError) as error:
            raise SystemExit(str(error)) from error
    elif args.output.exists():
        raise SystemExit("output already exists; use --resume or choose a new output path")
    completed = {(row["case"], row["method"]) for row in payload["records"]}
    with tempfile.TemporaryDirectory(prefix="tomojax-quality-") as temporary:
        directory = Path(temporary)
        for size in args.sizes:
            for kind in args.geometries:
                case_name = f"{kind}-{size}-{args.views}"
                pending = [
                    method for method in args.methods if (case_name, method) not in completed
                ]
                if not pending:
                    continue
                fixture = directory / f"{kind}-{size}.npz"
                base = [sys.executable, str(Path(__file__).resolve()), "--suite", args.suite]
                subprocess.run(
                    [
                        *base,
                        "--generate",
                        "--fixture",
                        str(fixture),
                        "--sizes",
                        str(size),
                        "--views",
                        str(args.views),
                        "--kind",
                        kind,
                    ],
                    check=True,
                    timeout=args.timeout,
                )
                for method in pending:
                    path = directory / f"{kind}-{size}-{method}.json"
                    command = [
                        *base,
                        "--fixture",
                        str(fixture),
                        "--method",
                        method,
                        "--kind",
                        kind,
                        "--repeats",
                        str(args.repeats),
                        "--max-iters",
                        str(args.max_iters),
                        "--batch",
                        str(args.batch),
                        "--fourier-slices",
                        str(args.fourier_slices),
                        "--output",
                        str(path),
                        "--started",
                        str(time.perf_counter()),
                    ]
                    record = isolated_run(command, path, args.timeout)
                    record.setdefault("case", f"{kind}-{size}-{args.views}")
                    record.setdefault("method", method)
                    payload["records"].append(record)
                    print(
                        f"{record['case']} {method}: {record['status']} "
                        f"iterations={record.get('selected_iterations')} "
                        f"warm_ms={record.get('selected_budget_warm_verified_median_ms')} "
                        f"peak_mib={record.get('sampled_process_peak_mib')}",
                        flush=True,
                    )
                    write_result(args.output, payload)
    payload["passed"] = all(record["status"] == "accepted" for record in payload["records"])
    write_result(args.output, payload)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
