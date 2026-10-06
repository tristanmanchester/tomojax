#!/usr/bin/env python3
r"""Compare cone-beam projection and FDK with ASTRA and TIGRE on one circular scan.

The phantom is five ellipsoids voxelised at 2x2x2 sub-samples; errors compare
forward projections with exact analytic line integrals and FDK volumes with the
voxelised phantom. TomoJAX and ASTRA run in this process; TIGRE runs in a child
process because it refuses a GPU on which JAX holds memory. A second scan with
the rotation axis tilted by ``--tilt-deg`` times the general-geometry kernels.
Times are the best of ``--repeats`` warm calls, with host-array inputs for
ASTRA and TIGRE.

    uv run --no-sync python bench/compare_cone.py --size 256 --views 360 \\
        --output bench/results/cone-256.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

ELLIPSOIDS = [  # amplitude, centre and radii as fractions of the volume width, z-rotation
    (1.0, (0.0, 0.0, 0.0), (0.28, 0.26, 0.30), 0.0),
    (-0.55, (-0.10, 0.04, 0.03), (0.09, 0.12, 0.10), 23.0),
    (0.75, (0.12, -0.08, -0.09), (0.065, 0.055, 0.08), -17.0),
    (1.0, (0.09, 0.10, 0.13), (0.03, 0.025, 0.035), 0.0),
    (-0.4, (-0.04, -0.13, -0.12), (0.045, 0.03, 0.05), 31.0),
]


def shapes(extent: float) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Return (amplitude, centre, metric) ellipsoids scaled to the volume width."""
    out = []
    for amp, centre, radii, angle in ELLIPSOIDS:
        c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        metric = rotation @ np.diag(1 / (extent * np.asarray(radii)) ** 2) @ rotation.T
        out.append((amp, extent * np.asarray(centre), metric))
    return out


def setup(size: int, views: int, tilt_deg: float = 0.5) -> dict[str, Any]:
    """Return the scan description shared by every library."""
    sod = 3.0 * size
    return {
        "size": size,
        "views": views,
        "tilt_deg": tilt_deg,
        "nu": int(1.5 * size),
        "nv": int(1.5 * size),
        "sod": sod,
        "sdd": 1.5 * sod,
        "angles": np.linspace(0.0, 360.0, views, endpoint=False),
    }


def make_case(case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return the voxelised phantom and its exact cone-beam line integrals."""
    n, nu, nv = case["size"], case["nu"], case["nv"]
    ellipsoids = shapes(float(n))
    origin = -(n - 1) / 2
    volume = np.zeros((n, n, n), np.float32)
    offsets = (np.arange(2) + 0.5) / 2 - 0.5
    for i in range(n):
        acc = np.zeros((n, n))
        for ox in offsets:
            for oy in offsets:
                for oz in offsets:
                    y, z = np.meshgrid(
                        origin + np.arange(n) + oy, origin + np.arange(n) + oz, indexing="ij"
                    )
                    p = np.stack([np.full_like(y, origin + i + ox), y, z], -1)
                    for amp, c, m in ellipsoids:
                        d = p - c
                        acc += amp * (np.einsum("...i,ij,...j->...", d, m, d) <= 1)
        volume[i] = acc / 8
    u = np.arange(nu) - (nu - 1) / 2
    v = np.arange(nv) - (nv - 1) / 2
    uu, vv = np.meshgrid(u, v)
    pixels = np.stack([uu, np.full_like(uu, case["sdd"] - case["sod"]), vv], -1)
    source = np.array([0.0, -case["sod"], 0.0])
    data = np.zeros((case["views"], nv, nu), np.float32)
    for k, theta in enumerate(np.deg2rad(case["angles"])):
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        src, pix = rot.T @ source, pixels @ rot
        direction = pix - src
        total = np.zeros((nv, nu))
        for amp, centre, m in ellipsoids:
            delta = src - centre
            a = np.einsum("...i,ij,...j->...", direction, m, direction)
            b = np.einsum("...i,ij,j->...", direction, m, delta)
            cc = delta @ m @ delta - 1
            total += (
                amp
                * 2
                * np.sqrt(np.maximum(b * b - a * cc, 0))
                / a
                * np.linalg.norm(direction, axis=-1)
            )
        data[k] = total
    return volume, data


def best(fn: Any, repeats: int, sync: Any = lambda _: None) -> tuple[Any, float]:
    """Return the last result and the best wall time of ``repeats`` warm calls."""
    out = fn()
    sync(out)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        sync(out)
        times.append(time.perf_counter() - start)
    return out, min(times)


def relative(a: np.ndarray, b: np.ndarray) -> float:
    """Return the relative L2 error of ``a`` against ``b``."""
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / np.linalg.norm(b))


def run_tomojax(
    case: dict[str, Any], volume: np.ndarray, data: np.ndarray, repeats: int
) -> list[dict]:
    """Time TomoJAX's cone projector, its exact transpose and FDK."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp

    from tomojax.core.cone import cone_backproject, cone_coefficients, cone_project
    from tomojax.geometry import ConeBeam, ConeGeometry, Detector, Grid
    from tomojax.recon import fdk

    n = case["size"]
    grid = Grid(n, n, n, 1.0, 1.0, 1.0)
    det = Detector(case["nu"], case["nv"], 1.0, 1.0)
    geometry = ConeGeometry(grid, det, case["angles"], ConeBeam(case["sod"], case["sdd"]))
    coeff = cone_coefficients(jnp.asarray(geometry.poses(), jnp.float32), grid, det, geometry.beam)
    dvol, ddata = jnp.asarray(volume), jnp.asarray(data)
    forward = jax.jit(lambda x: cone_project(x, coeff, grid, det))
    adjoint = jax.jit(lambda y: cone_backproject(y, coeff, grid, det))
    sync = jax.block_until_ready
    fp, t_fp = best(lambda: forward(dvol), repeats, sync)
    _, t_bp = best(lambda: adjoint(ddata), repeats, sync)
    rec, t_fdk = best(lambda: fdk(geometry, grid, det, ddata), repeats, sync)
    # A tilted rotation axis takes the general (non-separable) kernels.
    tilted = ConeGeometry(
        grid, det, case["angles"], ConeBeam(case["sod"], case["sdd"]), tilt_deg=case["tilt_deg"]
    )
    coeff_t = cone_coefficients(jnp.asarray(tilted.poses(), jnp.float32), grid, det, tilted.beam)
    forward_t = jax.jit(lambda x: cone_project(x, coeff_t, grid, det))
    adjoint_t = jax.jit(lambda y: cone_backproject(y, coeff_t, grid, det))
    _, t_fp_t = best(lambda: forward_t(dvol), repeats, sync)
    _, t_bp_t = best(lambda: adjoint_t(ddata), repeats, sync)
    tilt = f"axis tilted {case['tilt_deg']:g} deg"
    return [
        {
            "library": "tomojax",
            "operation": "forward",
            "seconds": t_fp,
            "error": relative(np.asarray(fp), data),
        },
        {"library": "tomojax", "operation": "backproject (exact transpose)", "seconds": t_bp},
        {
            "library": "tomojax",
            "operation": "fdk",
            "seconds": t_fdk,
            "error": relative(np.asarray(rec), volume),
        },
        {"library": "tomojax", "operation": f"forward, {tilt}", "seconds": t_fp_t},
        {"library": "tomojax", "operation": f"backproject, {tilt}", "seconds": t_bp_t},
    ]


def run_astra(
    case: dict[str, Any], volume: np.ndarray, data: np.ndarray, repeats: int
) -> list[dict]:
    """Time ASTRA's cone_vec projector, backprojector and FDK_CUDA."""
    import astra

    n, views = case["size"], case["views"]
    lo, hi = -n / 2, n / 2
    vol_geom = astra.create_vol_geom(n, n, n, lo, hi, lo, hi, lo, hi)
    theta = np.deg2rad(case["angles"])
    vectors = np.zeros((views, 12))
    for k, t in enumerate(theta):  # the object turns by +t: the source and detector turn by -t
        c, s = np.cos(t), np.sin(t)
        rot_t = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        vectors[k, 0:3] = rot_t @ [0.0, -case["sod"], 0.0]
        vectors[k, 3:6] = rot_t @ [0.0, case["sdd"] - case["sod"], 0.0]
        vectors[k, 6:9] = rot_t @ [1.0, 0.0, 0.0]
        vectors[k, 9:12] = rot_t @ [0.0, 0.0, 1.0]
    proj_geom = astra.create_proj_geom("cone_vec", case["nv"], case["nu"], vectors)
    avol = np.ascontiguousarray(volume.transpose(2, 1, 0))
    sino = np.ascontiguousarray(data.transpose(1, 0, 2))
    out = np.empty((case["nv"], views, case["nu"]), np.float32)
    back = np.empty_like(avol)

    def time_pair(geom: dict) -> tuple[float, float]:
        projector = astra.create_projector("cuda3d", geom, vol_geom)
        try:
            _, t_fp = best(lambda: astra.projector3d.direct_FP(projector, avol, out=out), repeats)
            _, t_bp = best(lambda: astra.projector3d.direct_BP(projector, sino, out=back), repeats)
        finally:
            astra.projector3d.delete(projector)
        return t_fp, t_bp

    # Same scan with the rotation axis tilted about x (timings only).
    tau = np.deg2rad(case["tilt_deg"])
    tilt_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(tau), np.sin(tau)], [0.0, -np.sin(tau), np.cos(tau)]]
    )
    tilted = vectors.copy()
    for j in range(4):
        tilted[:, 3 * j : 3 * j + 3] = vectors[:, 3 * j : 3 * j + 3] @ tilt_x.T
    t_fp_t, t_bp_t = time_pair(astra.create_proj_geom("cone_vec", case["nv"], case["nu"], tilted))
    t_fp, t_bp = time_pair(proj_geom)

    def fdk_call() -> np.ndarray:
        sid = astra.data3d.create("-sino", proj_geom, sino)
        rid = astra.data3d.create("-vol", vol_geom)
        cfg = astra.astra_dict("FDK_CUDA")
        cfg["ProjectionDataId"], cfg["ReconstructionDataId"] = sid, rid
        alg = astra.algorithm.create(cfg)
        astra.algorithm.run(alg)
        result = astra.data3d.get(rid)
        astra.algorithm.delete(alg)
        astra.data3d.delete([sid, rid])
        return result

    rec, t_fdk = best(fdk_call, repeats)
    return [
        {
            "library": "astra",
            "operation": "forward",
            "seconds": t_fp,
            "error": relative(out.transpose(1, 0, 2), data),
        },
        {
            "library": "astra",
            "operation": "backproject (voxel-driven, approximate transpose)",
            "seconds": t_bp,
        },
        {
            "library": "astra",
            "operation": "fdk",
            "seconds": t_fdk,
            "error": relative(rec.transpose(2, 1, 0), volume),
        },
        {
            "library": "astra",
            "operation": f"forward, axis tilted {case['tilt_deg']:g} deg",
            "seconds": t_fp_t,
        },
        {
            "library": "astra",
            "operation": f"backproject, axis tilted {case['tilt_deg']:g} deg",
            "seconds": t_bp_t,
        },
    ]


def run_tigre(
    case: dict[str, Any], volume: np.ndarray, data: np.ndarray, repeats: int
) -> list[dict]:
    """Time TIGRE's cone projectors, matched backprojector and FDK."""
    import tigre
    from tigre import algorithms

    n = case["size"]
    geo = tigre.geometry(mode="cone", default=False)
    geo.mode = "cone"
    geo.DSD, geo.DSO = case["sdd"], case["sod"]
    geo.nDetector = np.array([case["nv"], case["nu"]])
    geo.dDetector = np.array([1.0, 1.0])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.nVoxel = np.array([n, n, n])
    geo.dVoxel = np.array([1.0, 1.0, 1.0])
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    geo.offOrigin, geo.offDetector, geo.accuracy = np.zeros(3), np.zeros(2), 0.5
    # TIGRE's angle convention, checked against the analytic data: -90 - theta.
    angles = np.deg2rad(-90.0 - case["angles"]).astype(np.float32)
    tvol = np.ascontiguousarray(volume.transpose(2, 1, 0))
    records = []
    for method in ("interpolated", "Siddon"):
        fp, t = best(lambda m=method: tigre.Ax(tvol, geo, angles, projection_type=m), repeats)
        records.append(
            {
                "library": "tigre",
                "operation": f"forward ({method})",
                "seconds": t,
                "error": relative(fp, data),
            }
        )
    _, t = best(lambda: tigre.Atb(data, geo, angles, backprojection_type="matched"), repeats)
    records.append({"library": "tigre", "operation": "backproject (matched)", "seconds": t})
    rec, t = best(lambda: algorithms.fdk(data, geo, angles), repeats)
    records.append(
        {
            "library": "tigre",
            "operation": "fdk",
            "seconds": t,
            "error": relative(rec.transpose(2, 1, 0), volume),
        }
    )
    return records


def main() -> int:
    """Run the comparison and write the JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--views", type=int, default=360)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tilt-deg", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=["tigre"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    case = setup(args.size, args.views, args.tilt_deg)
    cache = args.output.with_suffix(".case.npz")
    if cache.exists():
        stored = np.load(cache)
        volume, data = stored["volume"], stored["data"]
    else:
        volume, data = make_case(case)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, volume=volume, data=data)
    if args.worker == "tigre":
        print(json.dumps(run_tigre(case, volume, data, args.repeats)))
        return 0
    records = run_tomojax(case, volume, data, args.repeats) + run_astra(
        case, volume, data, args.repeats
    )
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    child = subprocess.run(
        [sys.executable, __file__, *sys.argv[1:], "--worker", "tigre"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    records += json.loads(child.stdout.strip().splitlines()[-1])
    summary = {"case": {k: v for k, v in case.items() if k != "angles"}, "records": records}
    args.output.write_text(json.dumps(summary, indent=2))
    for r in records:
        error = f"  error {r['error']:.4f}" if "error" in r else ""
        print(f"{r['library']:8s} {r['operation']:52s} {r['seconds']:7.3f} s{error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
