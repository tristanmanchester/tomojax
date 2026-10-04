#!/usr/bin/env python3
"""Compare projectors against analytic line integrals, with synchronized timings.

Run with ``uv run --extra cuda12 --group benchmark python bench/compare_projectors.py``.
ASTRA is optional; TIGRE must be installed from CERN/TIGRE (the PyPI package named
``tigre`` is unrelated). Results distinguish resident GPU work from host-to-host
calls and include every timing sample and the actual backend/version.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
import ctypes
import ctypes.util
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import TYPE_CHECKING, Any

# Leave room for external libraries on the same GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from direct_reconstruction import astra_geometries
import jax
import numpy as np
from physical_case import Case

from tomojax.core.geometry.base import grid_volume_origin
from tomojax.core.pallas import api as pallas
from tomojax.core.projector import forward_project_view_T
from tomojax.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry

if TYPE_CHECKING:
    from collections.abc import Callable


def ellipsoid_chord_lengths(
    base: np.ndarray, direction: np.ndarray, center: np.ndarray, metric: np.ndarray
) -> np.ndarray:
    """Integrate a unit-density ellipsoid defined by (x-c)' metric (x-c) <= 1."""
    delta = base - center
    a = np.einsum("...i,ij,...j->...", direction, metric, direction)
    b = np.einsum("...i,ij,...j->...", delta, metric, direction)
    c = np.einsum("...i,ij,...j->...", delta, metric, delta) - 1
    return 2 * np.sqrt(np.maximum(b * b - a * c, 0)) / a * np.linalg.norm(direction, axis=-1)


def _sample_volume(
    shape: tuple[int, int, int],
    voxel: np.ndarray,
    origin: np.ndarray,
    gaussians: list[tuple[float, np.ndarray, np.ndarray]],
    ellipsoids: list[tuple[float, np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Sample the independent phantom with bounded FP64 coordinate workspace."""
    # Bound coordinate/intermediate arrays for the 512-cubed/720-view fixture.
    # Accumulate each element in FP64 in the original component order, then cast
    # once to FP32. Chunking changes storage, not the fixture definition.
    volume = np.empty(shape, dtype=np.float32)
    for start in range(0, shape[0], 32):
        stop = min(start + 32, shape[0])
        coords = np.stack(
            np.meshgrid(
                origin[0] + np.arange(start, stop) * voxel[0],
                origin[1] + np.arange(shape[1]) * voxel[1],
                origin[2] + np.arange(shape[2]) * voxel[2],
                indexing="ij",
            ),
            axis=-1,
        )
        values = np.zeros((stop - start, shape[1], shape[2]), dtype=np.float64)
        for amp, center, inv_var in gaussians:
            values += amp * np.exp(-0.5 * np.sum((coords - center) ** 2 * inv_var, axis=-1))
        for amp, center, metric in ellipsoids:
            delta = coords - center
            values += amp * (np.einsum("...i,ij,...j->...", delta, metric, delta) <= 1)
        volume[start:stop] = values

    return volume


def make_geometry(
    size: int, n_views: int, kind: str
) -> tuple[Grid, Detector, np.ndarray, np.ndarray]:
    """Construct a benchmark scan without allocating its phantom or measurements."""
    anisotropic = kind == "anisotropic"
    grid = Grid(
        size,
        size - 3 if anisotropic else size,
        size // 2 if anisotropic else size,
        0.8 if anisotropic else 1.0,
        1.2 if anisotropic else 1.0,
        1.4 if anisotropic else 1.0,
    )
    detector = Detector(
        size + 5 if anisotropic else size,
        grid.nz + 3 if anisotropic else size,
        grid.vx,
        grid.vz,
        (0.27, -0.31) if anisotropic else (0.0, 0.0),
    )
    angles = np.linspace(0, 180, n_views, endpoint=False, dtype=np.float32)
    geometry = (
        LaminographyGeometry(grid, detector, angles, tilt_deg=30)
        if kind == "lamino"
        else ParallelGeometry(grid, detector, angles)
    )
    poses = np.asarray([geometry.pose_for_view(i) for i in range(n_views)], dtype=np.float32)
    return grid, detector, poses, angles


def make_case(size: int, n_views: int, kind: str, *, phantom: str = "gaussian") -> Case:
    """Create asymmetric phantoms and independent exact parallel-ray integrals."""
    if phantom not in {"gaussian", "structured"}:
        raise ValueError(f"Unknown phantom: {phantom}")
    grid, detector, poses, angles = make_geometry(size, n_views, kind)
    origin = np.asarray(grid_volume_origin(grid))
    shape = (grid.nx, grid.ny, grid.nz)
    voxel = np.asarray([grid.vx, grid.vy, grid.vz])
    extent = np.asarray(shape) * voxel
    u = (np.arange(detector.nu) - (detector.nu - 1) / 2) * detector.du + detector.det_center[0]
    v = (np.arange(detector.nv) - (detector.nv - 1) / 2) * detector.dv + detector.det_center[1]
    uu, vv = np.meshgrid(u, v)
    world = np.stack([uu, np.zeros_like(uu), vv], axis=-1)
    gaussian_spec = (
        [
            (1.0, (-0.10, 0.08, -0.04), (0.08, 0.10, 0.09)),
            (0.65, (0.13, -0.09, 0.10), (0.065, 0.07, 0.06)),
        ]
        if phantom == "gaussian"
        else []
    )
    gaussians = []
    for amp, center_fraction, sigma_fraction in gaussian_spec:
        center = extent * np.asarray(center_fraction)
        inv_var = 1.0 / (extent * np.asarray(sigma_fraction)) ** 2
        gaussians.append((amp, center, inv_var))
    ellipsoids = []
    if phantom == "structured":
        for amp, center_fraction, radii_fraction, angle in [
            (1.0, (0.0, 0.0, 0.0), (0.28, 0.26, 0.30), 0.0),
            (-0.55, (-0.10, 0.04, 0.03), (0.09, 0.12, 0.10), 23.0),
            (0.75, (0.12, -0.08, -0.09), (0.065, 0.055, 0.08), -17.0),
            (1.0, (0.09, 0.10, 0.13), (0.03, 0.025, 0.035), 0.0),
            (-0.4, (-0.04, -0.13, -0.12), (0.045, 0.03, 0.05), 31.0),
        ]:
            center = extent * np.asarray(center_fraction)
            radii = extent * np.asarray(radii_fraction)
            cosine, sine = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
            rotation = np.asarray([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
            metric = rotation @ np.diag(1 / radii**2) @ rotation.T
            ellipsoids.append((amp, center, metric))

    volume = _sample_volume(shape, voxel, origin, gaussians, ellipsoids)

    analytic = np.empty((n_views, detector.nv, detector.nu), dtype=np.float32)
    for start in range(0, n_views, 32):
        stop = min(start + 32, n_views)
        batch = poses[start:stop]
        base = np.einsum("vui,nij->nvuj", world, batch[:, :3, :3])
        base -= np.einsum("ni,nij->nj", batch[:, :3, 3], batch[:, :3, :3])[:, None, None, :]
        direction = batch[:, 1, :3].astype(np.float64)
        values = np.zeros((stop - start, detector.nv, detector.nu), dtype=np.float64)
        for amp, center, inv_var in gaussians:
            diff = base - center
            a = np.sum(direction**2 * inv_var, axis=-1)[:, None, None]
            b = np.sum(diff * direction[:, None, None, :] * inv_var, axis=-1)
            c = np.sum(diff**2 * inv_var, axis=-1)
            values += amp * np.sqrt(2 * np.pi / a) * np.exp(-0.5 * (c - b * b / a))
        for amp, center, metric in ellipsoids:
            values += amp * ellipsoid_chord_lengths(
                base, direction[:, None, None, :], center, metric
            )
        analytic[start:stop] = values
    return Case(
        f"{kind}-{size}-{n_views}",
        grid,
        detector,
        poses,
        angles,
        volume,
        analytic,
    )


def measure(call: Callable[[], Any], sync: Callable[[Any], None], repeats: int) -> tuple[Any, dict]:
    """Separate the cold call, warmups, and synchronized steady-state samples."""
    start = time.perf_counter()
    output = call()
    sync(output)
    cold = time.perf_counter() - start
    for _ in range(2):
        sync(call())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = call()
        sync(output)
        samples.append((time.perf_counter() - start) * 1000)
    return output, {
        "cold_ms": cold * 1000,
        "samples_ms": samples,
        "median_ms": float(np.median(samples)),
        "minimum_ms": min(samples),
    }


def errors(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    """Measure absolute and relative errors without fitting an amplitude scale."""
    diff = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        "relative_l2": float(np.linalg.norm(diff) / np.linalg.norm(expected)),
        "max_abs": float(np.max(np.abs(diff))),
    }


def jax_sync(output: Any) -> None:
    """Wait for all leaves, including tuples of loss and gradient arrays."""
    jax.block_until_ready(output)


def cuda_sync() -> Callable[[Any], None]:
    """Synchronize external CUDA work, which JAX's dependency graph cannot track."""
    runtime = ctypes.CDLL(ctypes.util.find_library("cudart") or "libcudart.so.12")
    synchronize = runtime.cudaDeviceSynchronize
    synchronize.restype = ctypes.c_int

    def wait(_output: Any) -> None:
        status = synchronize()
        if status:
            raise RuntimeError(f"cudaDeviceSynchronize failed with status {status}")

    return wait


def run_tomojax(case: Case, repeats: int) -> list[dict]:
    """Time the JAX reference and actual Pallas stack kernel on identical inputs."""
    poses, volume = jax.device_put((case.poses, case.volume))
    jax.block_until_ready((poses, volume))
    reference = jax.jit(
        lambda x: jax.vmap(lambda t: forward_project_view_T(t, case.grid, case.detector, x))(poses)
    )
    calls = {"jax": reference}
    if jax.default_backend() == "gpu":
        calls["pallas"] = jax.jit(
            lambda x: pallas.forward_project_views_T_pallas(
                poses,
                case.grid,
                case.detector,
                x,
                options=pallas.PallasProjectorOptions(tile_shape=(16, 4), num_warps=1),
            )
        )
    records = []
    ref_output = None
    for backend, call in calls.items():
        output, timing = measure(lambda call=call: call(volume), jax_sync, repeats)
        host = np.asarray(output)
        if ref_output is None:
            ref_output = host
        records.append(
            {
                "library": "tomojax",
                "backend": backend,
                "scope": "device_resident",
                **timing,
                "analytic_error": errors(host, case.analytic),
                "reference_error": errors(host, ref_output),
            }
        )
        output, timing = measure(
            lambda call=call: np.asarray(call(jax.device_put(case.volume))), lambda _: None, repeats
        )
        records.append(
            {
                "library": "tomojax",
                "backend": backend,
                "scope": "host_to_host",
                **timing,
                "analytic_error": errors(output, case.analytic),
            }
        )
    return records


def run_astra(case: Case, repeats: int) -> list[dict]:
    """Use ASTRA 2.5 direct_FP, including a zero-copy JAX/DLPack device comparison."""
    import astra

    vg, pg = astra_geometries(case)
    projector = astra.create_projector("cuda3d", pg, vg)
    volume = np.ascontiguousarray(case.volume.transpose(2, 1, 0))
    output = np.empty((case.detector.nv, len(case.poses), case.detector.nu), dtype=np.float32)
    sync = cuda_sync()
    records = []
    try:
        host, timing = measure(
            lambda: astra.projector3d.direct_FP(projector, volume, out=output), sync, repeats
        )
        records.append(
            {
                "library": "astra",
                "backend": "cuda3d",
                "scope": "host_to_host",
                **timing,
                "analytic_error": errors(host.transpose(1, 0, 2), case.analytic),
            }
        )
        if jax.default_backend() == "gpu":
            device_volume = jax.device_put(volume)
            device_output = jax.device_put(np.zeros_like(output))
            jax.block_until_ready((device_volume, device_output))
            result, timing = measure(
                lambda: astra.projector3d.direct_FP(projector, device_volume, out=device_output),
                sync,
                repeats,
            )
            records.append(
                {
                    "library": "astra",
                    "backend": "cuda3d",
                    "scope": "device_resident",
                    **timing,
                    "analytic_error": errors(np.asarray(result).transpose(1, 0, 2), case.analytic),
                }
            )
    finally:
        astra.projector3d.delete(projector)
    return records


def run_tigre(case: Case, repeats: int) -> list[dict]:
    """Compare the matching z-axis parallel scan with TIGRE's host-array API."""
    import tigre

    if not hasattr(tigre, "Ax"):
        raise ImportError("Install CERN/TIGRE from source; PyPI's tigre is unrelated")
    if not case.name.startswith("parallel-"):
        return [
            {
                "library": "tigre",
                "status": "unsupported_comparison",
                "reason": "This adapter currently validates only centered isotropic parallel scans",
            }
        ]
    g = case.grid
    volume = np.ascontiguousarray(case.volume.transpose(2, 1, 0))
    geo = tigre.geometry(mode="parallel", nVoxel=np.asarray(volume.shape))
    geo.dVoxel = np.asarray([g.vz, g.vy, g.vx])
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    geo.nDetector = np.asarray([case.detector.nv, case.detector.nu])
    geo.dDetector = np.asarray([case.detector.dv, case.detector.du])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.accuracy = 1.0
    # TIGRE's zero-angle beam follows -x, detector u follows +y, and v follows
    # -z internally, with rows reversed by its output kernel. Rotate the beam/u
    # frame into TomoJAX's object coordinates; returned rows already increase z.
    angles = np.deg2rad(-90.0 - case.angles_deg).astype(np.float32)
    records = []
    for method in ["interpolated", "Siddon"]:
        output, timing = measure(
            lambda method=method: tigre.Ax(volume, geo, angles, projection_type=method),
            lambda _: None,
            repeats,
        )
        records.append(
            {
                "library": "tigre",
                "backend": method,
                "scope": "host_to_host",
                **timing,
                "analytic_error": errors(output, case.analytic),
            }
        )
    return records


def environment() -> dict[str, Any]:
    """Capture enough runtime details to qualify performance claims."""
    versions = {}
    for package in ["tomojax", "jax", "jaxlib", "numpy", "astra-toolbox", "tigre"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    gpu_info = None
    with suppress(FileNotFoundError, subprocess.CalledProcessError):
        gpu_info = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,power.limit",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    return {
        "versions": versions,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "devices": [str(device) for device in jax.devices()],
        "jax_backend": jax.default_backend(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "jax_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "gpu_info": gpu_info,
    }


def main() -> int:
    """Write a machine-readable benchmark report and fail on execution errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="32,64,128")
    parser.add_argument("--views", type=int, default=60)
    parser.add_argument("--geometries", default="parallel,lamino,anisotropic")
    parser.add_argument("--libraries", default="tomojax,astra")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--max-relative-error",
        type=float,
        default=0.06,
        help="Maximum relative L2 error against analytic line integrals.",
    )
    parser.add_argument("--output", type=Path, default=Path("bench/results/projectors.json"))
    args = parser.parse_args()
    sizes = [int(size) for size in args.sizes.split(",")]
    if min(sizes) < 8 or args.views < 1 or args.repeats < 1:
        parser.error("sizes must be >= 8; views and repeats must be positive")
    if not np.isfinite(args.max_relative_error) or args.max_relative_error <= 0:
        parser.error("max-relative-error must be finite and positive")
    kinds = args.geometries.split(",")
    if set(kinds) - {"parallel", "lamino", "anisotropic"}:
        parser.error("geometries must be parallel, lamino, or anisotropic")
    runners = {"tomojax": run_tomojax, "astra": run_astra, "tigre": run_tigre}
    libraries = args.libraries.split(",")
    if set(libraries) - runners.keys():
        parser.error("libraries must be tomojax, astra, or tigre")
    payload = {
        "environment": environment(),
        "cases": [],
        "failed": False,
        "max_relative_error": args.max_relative_error,
        "max_reference_error": 5e-5,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for size in sizes:
        for kind in kinds:
            case = make_case(size, args.views, kind)
            records = []
            for library in libraries:
                try:
                    records.extend(runners[library](case, args.repeats))
                except ImportError as exc:
                    payload["failed"] = True
                    records.append(
                        {"library": library, "status": "unavailable", "reason": str(exc)}
                    )
                except Exception as exc:
                    payload["failed"] = True
                    records.append(
                        {
                            "library": library,
                            "status": "failed",
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
            for record in records:
                if "analytic_error" in record:
                    record["accuracy_passed"] = bool(
                        record["analytic_error"]["relative_l2"] <= args.max_relative_error
                        and record.get("reference_error", {}).get("relative_l2", 0.0) <= 5e-5
                    )
                    payload["failed"] |= not record["accuracy_passed"]
            entry = {
                "name": case.name,
                "grid": case.grid.to_dict(),
                "detector": case.detector.to_dict(),
                "poses": case.poses.tolist(),
                "records": records,
            }
            payload["cases"].append(entry)
            args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"case": case.name, "records": records}), flush=True)
            jax.clear_caches()
    return int(payload["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
