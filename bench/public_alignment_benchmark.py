"""Public free-voxel alignment pilot; never a restricted object-model fit.

Frozen six-cell development pilot: size 32, 61 irregular views, parallel /
anisotropic / 30-degree laminography, clean and 0.1-percent RMS Gaussian noise.
Every voxel is optimized from zero, alongside all five per-view pose parameters.
The independent data integrator exactly integrates the trilinear truth basis.
This pilot does not establish the larger-motion 99-percent robustness target.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial.transform import Rotation

from tomojax.core.geometry.base import Detector, Grid, grid_volume_origin

SUITE = "public-free-voxel-v1"
SIZE, VIEWS, SEED = 32, 61, 461
OUTERS, INNER_ITERS = 64, 20
IMAGE_GATES = {"parallel": 0.10, "anisotropic": 0.10, "lamino": 0.20}


def physical_poses(nominal: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Independent rotations in radians and lab translations in physical units."""
    result = np.asarray(nominal, dtype=np.float64).copy()
    rotation = Rotation.from_euler("YXZ", np.asarray(parameters)[:, [1, 0, 2]]).as_matrix()
    result[:, :3, :3] = result[:, :3, :3] @ rotation
    result[:, 0, 3] += parameters[:, 3]
    result[:, 2, 3] += parameters[:, 4]
    return result


def gauge_alignment(actual: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Fit only one shared rigid object frame, never independent per-view fixes."""
    covariance = np.einsum("vji,vjk->ik", actual[:, :3, :3], truth[:, :3, :3])
    left, _, right = np.linalg.svd(covariance)
    signs = np.diag([1.0, 1.0, np.linalg.det(left @ right)])
    rotation = left @ signs @ right
    matrix = actual[:, [0, 2], :3].reshape(-1, 3)
    rhs = (truth[:, [0, 2], 3] - actual[:, [0, 2], 3]).ravel()
    translation, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=1e-10)
    if rank != 3:
        raise ValueError("scan does not determine a shared object translation gauge")
    gauge = np.eye(4)
    gauge[:3, :3], gauge[:3, 3] = rotation, translation
    return gauge


def verify(volume: np.ndarray, parameters: np.ndarray, fixture: dict[str, Any]) -> dict[str, Any]:
    """Check the full volume and observable poses under one shared object gauge."""
    grid, detector = fixture["grid"], fixture["detector"]
    expected = fixture["truth_poses"]
    actual = physical_poses(fixture["nominal"], parameters)
    if not np.isfinite(volume).all() or not np.isfinite(actual).all():
        return {"accepted": False, "finite": False}
    gauge = gauge_alignment(actual, expected)
    aligned = actual @ gauge

    def errors(poses: np.ndarray) -> dict[str, float]:
        relative = poses[:, :3, :3] @ expected[:, :3, :3].transpose(0, 2, 1)
        angles = np.rad2deg(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec(), axis=1))
        shifts = (poses[:, [0, 2], 3] - expected[:, [0, 2], 3]) / [detector.du, detector.dv]
        return {
            "rotation_rmse_deg": float(np.sqrt(np.mean(angles**2))),
            "translation_vector_rmse_px": float(np.sqrt(np.mean(np.sum(shifts**2, axis=1)))),
            "rotation_max_deg": float(angles.max()),
            "translation_vector_max_px": float(np.linalg.norm(shifts, axis=1).max()),
        }

    shape = fixture["truth"].shape
    spacing = np.array([grid.vx, grid.vy, grid.vz])
    origin = np.array(grid_volume_origin(grid))
    points = np.indices(shape).reshape(3, -1).T * spacing + origin
    estimate_points = points @ gauge[:3, :3].T + gauge[:3, 3]
    indices = ((estimate_points - origin) / spacing).T
    aligned_volume = map_coordinates(
        volume, indices, order=1, mode="grid-constant", prefilter=False
    )
    reference = fixture["truth"].astype(np.float64).ravel()
    relative_l2 = float(np.linalg.norm(aligned_volume - reference) / np.linalg.norm(reference))
    physical = errors(aligned)
    return dict(
        finite=True,
        **physical,
        raw_pose_errors=errors(actual),
        volume_relative_l2=relative_l2,
        object_frame_transform=gauge.tolist(),
        accepted=bool(
            relative_l2 <= IMAGE_GATES[fixture["kind"]]
            and physical["rotation_rmse_deg"] <= 0.01
            and physical["translation_vector_rmse_px"] <= 0.05
        ),
    )


def generate_fixture(
    path: Path,
    kind: str,
    *,
    noisy: bool,
    size: int = SIZE,
    views: int = VIEWS,
    rotation_deg: float = 0.25,
    shift_px: float = 0.5,
    noise: float = 0.001,
) -> None:
    """Write a random-voxel scan and independent exact-basis measurements.

    Defaults reproduce the frozen pilot; other values define a different suite.
    """
    from voxel_truth import project_voxel_truth

    from tomojax.core.geometry.lamino import LaminographyGeometry
    from tomojax.core.geometry.parallel import ParallelGeometry

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
        (0.27, -0.31) if anisotropic else (0, 0),
    )
    rng = np.random.default_rng(SEED)
    angles = np.arange(views) * 180 / views + rng.uniform(-0.3, 0.3, views)
    geometry = (
        LaminographyGeometry(grid, detector, angles, tilt_deg=30)
        if kind == "lamino"
        else ParallelGeometry(grid, detector, angles)
    )
    nominal = np.asarray([geometry.pose_for_view(i) for i in range(views)])
    shape = (grid.nx, grid.ny, grid.nz)
    field = gaussian_filter(rng.normal(size=shape), sigma=1.8, mode="constant")
    field = np.exp(0.7 * field / np.std(field))
    normalized = np.stack(
        np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij"), axis=-1
    )
    support = np.maximum(1 - np.sum((normalized / 0.82) ** 2, axis=-1), 0) ** 2
    volume = field * support
    volume = (volume / volume.max()).astype(np.float32)
    parameters = np.concatenate(
        [
            np.deg2rad(rng.uniform(-rotation_deg, rotation_deg, (views, 3))),
            rng.uniform(-shift_px, shift_px, (views, 2)) * [detector.du, detector.dv],
        ],
        axis=1,
    )
    poses = physical_poses(nominal, parameters)
    data = project_voxel_truth(volume, poses, grid, detector)
    sigma = float(noise * np.sqrt(np.mean(data.astype(np.float64) ** 2))) if noisy else 0.0
    data = data + rng.normal(0, sigma, data.shape).astype(np.float32)
    np.savez(
        path,
        kind=kind,
        noisy=noisy,
        grid=json.dumps(asdict(grid)),
        detector=json.dumps(asdict(detector)),
        angles=angles,
        nominal=nominal,
        truth_poses=poses,
        truth_params=parameters,
        truth=volume,
        data=data,
        noise_sigma=sigma,
    )


def generate_analytic_fixture(
    path: Path,
    kind: str,
    *,
    noisy: bool,
    size: int = SIZE,
    views: int = VIEWS,
    rotation_deg: float = 0.25,
    shift_px: float = 0.5,
    noise: float = 0.001,
) -> None:
    """Write a scan of continuous Gaussian blobs with analytic line integrals.

    The voxel-basis pilot integrates the same trilinear basis as the solver's
    exact integrator, an inverse crime. These data match no discretisation, so
    pose errors include the model-mismatch floor real data would show.
    """
    from compare_projectors import _sample_volume

    from tomojax.core.geometry.lamino import LaminographyGeometry
    from tomojax.core.geometry.parallel import ParallelGeometry

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
        (0.27, -0.31) if anisotropic else (0, 0),
    )
    rng = np.random.default_rng(SEED)
    angles = np.arange(views) * 180 / views + rng.uniform(-0.3, 0.3, views)
    geometry = (
        LaminographyGeometry(grid, detector, angles, tilt_deg=30)
        if kind == "lamino"
        else ParallelGeometry(grid, detector, angles)
    )
    nominal = np.asarray([geometry.pose_for_view(i) for i in range(views)])
    extent = np.array([grid.nx * grid.vx, grid.ny * grid.vy, grid.nz * grid.vz])
    blobs = [
        (
            rng.uniform(0.3, 1.0),
            rng.uniform(-0.3, 0.3, 3) * extent,
            1 / (rng.uniform(0.02, 0.07, 3) * extent.min()) ** 2,
        )
        for _ in range(24)
    ]
    origin = np.asarray(grid_volume_origin(grid))
    shape = (grid.nx, grid.ny, grid.nz)
    volume = _sample_volume(shape, np.array([grid.vx, grid.vy, grid.vz]), origin, blobs, [])
    parameters = np.concatenate(
        [
            np.deg2rad(rng.uniform(-rotation_deg, rotation_deg, (views, 3))),
            rng.uniform(-shift_px, shift_px, (views, 2)) * [detector.du, detector.dv],
        ],
        axis=1,
    )
    poses = physical_poses(nominal, parameters)
    u = (np.arange(detector.nu) - (detector.nu - 1) / 2) * detector.du + detector.det_center[0]
    v = (np.arange(detector.nv) - (detector.nv - 1) / 2) * detector.dv + detector.det_center[1]
    uu, vv = np.meshgrid(u, v)
    world = np.stack([uu, np.zeros_like(uu), vv], axis=-1)
    data = np.empty((views, detector.nv, detector.nu), np.float32)
    for i, pose in enumerate(poses):
        base, direction = (world - pose[:3, 3]) @ pose[:3, :3], pose[1, :3]
        values = np.zeros(world.shape[:2])
        for amplitude, centre, inv_var in blobs:
            diff = base - centre
            a = np.sum(direction**2 * inv_var)
            b = np.sum(diff * direction * inv_var, -1)
            c = np.sum(diff**2 * inv_var, -1)
            values += amplitude * np.sqrt(2 * np.pi / a) * np.exp(-0.5 * (c - b * b / a))
        data[i] = values
    sigma = float(noise * np.sqrt(np.mean(data.astype(np.float64) ** 2))) if noisy else 0.0
    data = data + rng.normal(0, sigma, data.shape).astype(np.float32)
    np.savez(
        path,
        kind=kind,
        noisy=noisy,
        grid=json.dumps(asdict(grid)),
        detector=json.dumps(asdict(detector)),
        angles=angles,
        nominal=nominal,
        truth_poses=poses,
        truth_params=parameters,
        truth=volume.astype(np.float32),
        data=data,
        noise_sigma=sigma,
    )


def load_fixture(path: Path) -> dict[str, Any]:
    """Read host arrays and lightweight physical metadata."""
    with np.load(path) as data:
        result = {key: data[key] for key in data.files}
    result["kind"] = str(result["kind"])
    result["grid"] = Grid(**json.loads(str(result["grid"])))
    result["detector"] = Detector(**json.loads(str(result["detector"])))
    return result


def one_run(
    path: Path,
    ray_integrator: str = "sampled",
    gn_coupling: str = "fixed_volume",
    gn_jacobian: str = "central",
    gn_joint_solver: str = "stacked",
    views_per_batch: int = 1,
    factors: tuple[int, ...] = (1,),
) -> dict[str, Any]:
    """Time a public free-voxel solve including all setup and quality checks."""
    start = time.perf_counter()
    import jax
    import jax.numpy as jnp

    from tomojax.alignment import AlignConfig, align_multires
    from tomojax.alignment.api import L2LossSpec, align
    from tomojax.geometry import LaminographyGeometry, ParallelGeometry

    fixture = load_fixture(path)
    g, d = fixture["grid"], fixture["detector"]
    geometry = (
        LaminographyGeometry(g, d, fixture["angles"], tilt_deg=30)
        if fixture["kind"] == "lamino"
        else ParallelGeometry(g, d, fixture["angles"])
    )
    history = []

    def observer(x: Any, parameters: Any, stat: dict[str, Any]) -> str:
        x, parameters = jax.device_get((x, parameters))
        quality = verify(x, parameters, fixture)
        history.append(
            {
                "outer": len(history) + 1,
                "elapsed_verified_ms": (time.perf_counter() - start) * 1000,
                "quality": quality,
                "solver_stat": stat,
            }
        )
        return "stop_run" if quality["accepted"] else "continue"

    config = AlignConfig(
        ray_integrator=ray_integrator,
        gn_coupling=gn_coupling,
        gn_joint_solver=gn_joint_solver,
        pose_translation_frame="detector",
        projector_backend="pallas",
        gather_dtype="fp32",
        gn_jacobian=gn_jacobian,
        outer_iterations=OUTERS,
        iterations=INNER_ITERS,
        tv_weight=0,
        loss=L2LossSpec(),
        early_stop=False,
        views_per_batch=views_per_batch,
    )
    if factors == (1,):
        volume, parameters, info = align(
            geometry,
            g,
            d,
            jnp.asarray(fixture["data"]),
            config=config,
            init_x=jnp.zeros(fixture["truth"].shape, dtype=jnp.float32),
            observer=observer,
        )
    else:
        # Coarser levels have a different grid; acceptance is checked on the final one.
        volume, parameters, info = align_multires(
            geometry,
            g,
            d,
            jnp.asarray(fixture["data"]),
            factors=factors,
            config=config,
        )
    volume, parameters = jax.device_get((volume, parameters))
    quality = verify(volume, parameters, fixture)
    return {
        "verified_ms": (time.perf_counter() - start) * 1000,
        "quality": quality,
        "outer_iterations": len(history),
        "history": history,
        "config": asdict(config),
        "volume_degrees_of_freedom": int(volume.size),
        "pose_degrees_of_freedom": int(parameters.size),
        "execution_profile": info.get("execution_profile"),
        "solver": "public tomojax.alignment.align",
    }


def worker(args: argparse.Namespace) -> None:
    """Retain cold and repeated accepted or failed attempts in one isolated process."""
    from compare_reconstructions import write_result

    payload = {
        "suite": SUITE,
        "status": "running",
        "runs": [],
        "cold_process_start": args.process_start,
        "baseline_eligible": False,
        "repeat_count_requested": args.repeats,
        "minimum_headline_repeats": 7,
        "interpretation": "Free-voxel pilot; no claim of 99% success or large-motion coverage",
    }
    write_result(args.output, payload)
    try:
        for repeat in range(args.repeats + 1):
            row = one_run(
                args.fixture,
                args.ray_integrator,
                args.gn_coupling,
                args.gn_jacobian,
                args.gn_joint_solver,
                args.views_per_batch,
                tuple(args.factors),
            )
            if repeat == 0:
                row["fresh_process_verified_ms"] = (time.perf_counter() - args.process_start) * 1000
            payload["runs"].append(row)
            write_result(args.output, payload)
        payload["status"] = (
            "accepted"
            if all(r["quality"]["accepted"] for r in payload["runs"])
            else "target_not_reached"
        )
        payload["baseline_eligible"] = payload["status"] == "accepted" and args.repeats >= 7
        payload["warm_verified_median_ms"] = float(
            np.median([r["verified_ms"] for r in payload["runs"][1:]])
        )
    except Exception:
        payload["status"] = "execution_failed"
        payload["error"] = traceback.format_exc()
    write_result(args.output, payload)


def main() -> None:
    """Run every pilot cell serially and retain process GPU memory measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--process-start", type=float)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--ray-integrator", choices=("sampled", "exact"), default="sampled")
    parser.add_argument("--gn-coupling", choices=("fixed_volume", "joint"), default="fixed_volume")
    parser.add_argument("--gn-jacobian", choices=("central", "autodiff"), default="central")
    parser.add_argument(
        "--gn-joint-solver", choices=("stacked", "pose_eliminated"), default="stacked"
    )
    parser.add_argument("--size", type=int, default=SIZE, help="Nominal grid size")
    parser.add_argument(
        "--phantom",
        choices=("voxel", "analytic"),
        default="voxel",
        help="voxel: frozen pilot (exact voxel-basis data); analytic: continuous Gaussian blobs",
    )
    parser.add_argument(
        "--factors", type=int, nargs="+", default=[1], help="Coarse-to-fine factors, e.g. 4 2 1"
    )
    parser.add_argument("--views", type=int, default=VIEWS)
    parser.add_argument("--rotation-deg", type=float, default=0.25, help="Uniform motion bound")
    parser.add_argument("--shift-px", type=float, default=0.5, help="Uniform motion bound")
    parser.add_argument("--noise", type=float, default=0.001, help="Noise std / data RMS")
    parser.add_argument("--cells", nargs="+", help="Subset such as lamino-noisy; default all six")
    parser.add_argument(
        "--views-per-batch",
        type=int,
        default=1,
        help="Reconstruction views per batch; 0 uses the library's automatic size",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("at least one repeated complete call is required")
    if args.worker:
        if args.fixture is None or args.process_start is None:
            parser.error("worker mode requires --fixture and --process-start")
        worker(args)
        return
    from compare_reconstructions import environment, isolated_run, write_result

    args.output.mkdir(parents=True, exist_ok=False)
    payload = {
        "suite": SUITE,
        "ray_integrator": args.ray_integrator,
        "gn_coupling": args.gn_coupling,
        "gn_jacobian": args.gn_jacobian,
        "gn_joint_solver": args.gn_joint_solver,
        "views_per_batch": args.views_per_batch,
        "complete": False,
        "environment": environment(),
        "records": [],
        "scheduled_cells": [
            {"kind": k, "noisy": n}
            for k in ("parallel", "anisotropic", "lamino")
            for n in (False, True)
            if not args.cells or f"{k}-{'noisy' if n else 'clean'}" in args.cells
        ],
        "gates": {
            "rotation_rmse_deg": 0.01,
            "translation_vector_rmse_px": 0.05,
            "volume_relative_l2": IMAGE_GATES,
        },
        "fixture": {
            "size": args.size,
            "views": args.views,
            "seed": SEED,
            "noise_relative_rms": args.noise,
            "motion_rotation_bound_deg": args.rotation_deg,
            "motion_translation_bound_px": args.shift_px,
            "phantom": args.phantom,
        },
        "limits": {"outer_iters": OUTERS, "recon_iters": INNER_ITERS, "worker_timeout_s": 1800},
        "initialization": "zero free voxels and nominal poses",
        "gauge": "one shared rigid object transform, applied to both poses and volume",
        "identifiability": (
            "empirical recovery pilot; joint-noisy-distribution identifiability not established"
        ),
    }
    summary = args.output / "manifest.json"
    write_result(summary, payload)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    for cell in payload["scheduled_cells"]:
        name = f"{cell['kind']}-{'noisy' if cell['noisy'] else 'clean'}"
        fixture = args.output / f"{name}.npz"
        generate = generate_analytic_fixture if args.phantom == "analytic" else generate_fixture
        generate(
            fixture,
            cell["kind"],
            noisy=cell["noisy"],
            size=args.size,
            views=args.views,
            rotation_deg=args.rotation_deg,
            shift_px=args.shift_px,
            noise=args.noise,
        )
        result = args.output / f"{name}.json"
        command = [
            sys.executable,
            __file__,
            "--worker",
            "--fixture",
            str(fixture),
            "--output",
            str(result),
            "--repeats",
            str(args.repeats),
            "--ray-integrator",
            args.ray_integrator,
            "--gn-coupling",
            args.gn_coupling,
            "--gn-jacobian",
            args.gn_jacobian,
            "--gn-joint-solver",
            args.gn_joint_solver,
            "--views-per-batch",
            str(args.views_per_batch),
            "--factors",
            *map(str, args.factors),
            "--process-start",
            str(time.perf_counter()),
        ]
        print("START", name, flush=True)
        record = isolated_run(command, result, 1800)
        payload["records"].append(
            dict(
                case=name, fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(), **record
            )
        )
        write_result(summary, payload)
        print("DONE", name, record["status"], flush=True)
    payload["source_unchanged"] = (
        environment()["source_tree_sha256"] == payload["environment"]["source_tree_sha256"]
    )
    payload["complete"] = (
        len(payload["records"]) == len(payload["scheduled_cells"]) and payload["source_unchanged"]
    )
    write_result(summary, payload)


if __name__ == "__main__":
    main()
