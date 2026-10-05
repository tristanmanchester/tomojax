"""Recover per-view motion in a laminography scan and compare reconstructions.

The measurements are analytic line integrals of continuous Gaussian blobs at
perturbed poses, so they match no voxel discretisation. One reconstruction
uses the nominal geometry; the other is solved jointly with the poses by
``tomojax.align.align`` and ``coupled_pose_config``.

Run with ``uv run --no-sync python examples/align_misaligned_scan.py`` (a CUDA
GPU takes about a minute at the default size).
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.align import align, coupled_pose_config
from tomojax.align.api import apply_pose_updates
from tomojax.geometry import Detector, Grid, LaminographyGeometry, stack_view_poses
from tomojax.recon import CGLSConfig, cgls

ROOT = Path(__file__).resolve().parents[1]


def misaligned_scan(
    size: int = 96, views: int = 240, rotation_deg: float = 1.0, shift_px: float = 2.0
) -> dict[str, object]:
    """Return a 30-degree laminography scan with random per-view motion."""
    rng = np.random.default_rng(2026)
    grid = Grid(size, size, size, 1.0, 1.0, 1.0)
    detector = Detector(size, size, 1.0, 1.0)
    angles = np.linspace(0.0, 360.0, views, endpoint=False)
    geometry = LaminographyGeometry(grid, detector, angles, tilt_deg=30)
    nominal = stack_view_poses(geometry, views)
    truth_params = np.concatenate(
        [
            np.deg2rad(rng.uniform(-rotation_deg, rotation_deg, (views, 3))),
            rng.uniform(-shift_px, shift_px, (views, 2)),
        ],
        axis=1,
    ).astype(np.float32)
    poses = np.asarray(
        apply_pose_updates(nominal, jnp.asarray(truth_params), translation_frame="object"),
        np.float64,
    )
    blobs = [
        (rng.uniform(0.3, 1.0), rng.uniform(-0.3, 0.3, 3) * size, rng.uniform(1.0, 4.0, 3))
        for _ in range(60)
    ]
    centres = (np.arange(size) - (size - 1) / 2).astype(np.float64)
    x, y, z = np.meshgrid(centres, centres, centres, indexing="ij")
    truth = sum(
        a
        * np.exp(
            -0.5 * (((x - c[0]) / s[0]) ** 2 + ((y - c[1]) / s[1]) ** 2 + ((z - c[2]) / s[2]) ** 2)
        )
        for a, c, s in blobs
    )
    uu, vv = np.meshgrid(centres, centres)
    world = np.stack([uu, np.zeros_like(uu), vv], axis=-1)
    projections = np.empty((views, size, size), np.float32)
    for i, pose in enumerate(poses):
        base, direction = (world - pose[:3, 3]) @ pose[:3, :3], pose[1, :3]
        values = np.zeros(world.shape[:2])
        for amplitude, centre, sigma in blobs:
            inv_var = 1 / sigma**2
            diff = base - centre
            a = np.sum(direction**2 * inv_var)
            b = np.sum(diff * direction * inv_var, -1)
            c = np.sum(diff**2 * inv_var, -1)
            values += amplitude * np.sqrt(2 * np.pi / a) * np.exp(-0.5 * (c - b * b / a))
        projections[i] = values
    noise = 0.002 * np.sqrt(np.mean(projections.astype(np.float64) ** 2))
    projections += rng.normal(0, noise, projections.shape).astype(np.float32)
    return {
        "geometry": geometry,
        "grid": grid,
        "detector": detector,
        "projections": projections,
        "truth": truth.astype(np.float32),
        "truth_params": truth_params,
    }


def run_example(
    size: int = 96, views: int = 240
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Reconstruct with nominal poses and with jointly aligned poses."""
    scan = misaligned_scan(size, views)
    geometry, grid, detector = scan["geometry"], scan["grid"], scan["detector"]
    projections = jnp.asarray(scan["projections"])
    truth = scan["truth"]
    nominal, _ = cgls(geometry, grid, detector, projections, config=CGLSConfig(iters=30))
    start = time.perf_counter()
    aligned, params, info = align(
        geometry, grid, detector, projections, config=coupled_pose_config()
    )
    aligned = np.asarray(jax.device_get(aligned))
    elapsed = time.perf_counter() - start
    params = np.asarray(params)
    # Per-view errors after removing the common offset, which only moves the object.
    difference = params - scan["truth_params"]
    difference -= difference.mean(axis=0)

    def error(volume: np.ndarray) -> float:
        return float(np.linalg.norm(volume - truth) / np.linalg.norm(truth))

    metrics = {
        "size": size,
        "views": views,
        "motion": "uniform +/-1 deg rotations, +/-2 px shifts per view",
        "nominal_cgls_relative_l2": error(np.asarray(nominal)),
        "aligned_relative_l2": error(aligned),
        "rotation_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(difference[:, :3] ** 2)))),
        "translation_rmse_px": float(np.sqrt(np.mean(difference[:, 3:] ** 2))),
        "outer_iterations": len(info.get("outer_stats", [])),
        "alignment_seconds_including_compile": elapsed,
        "device_platform": jax.default_backend(),
    }
    volumes = {"truth": truth, "nominal": np.asarray(nominal), "aligned": aligned}
    return volumes, metrics


def plot(volumes: dict[str, np.ndarray], metrics: dict[str, object], path: Path) -> None:
    import matplotlib.pyplot as plt

    titles = {
        "truth": "Truth",
        "nominal": f"Nominal poses (CGLS)\nerror {metrics['nominal_cgls_relative_l2']:.3f}",
        "aligned": f"Jointly aligned\nerror {metrics['aligned_relative_l2']:.3f}",
    }
    centre = volumes["truth"].shape[2] // 2
    high = float(np.percentile(volumes["truth"], 99.9))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), constrained_layout=True)
    for axis, (key, title) in zip(axes, titles.items(), strict=True):
        axis.imshow(volumes[key][:, :, centre].T, origin="lower", cmap="gray", vmin=0, vmax=high)
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        f"30° laminography, {metrics['views']} views with random per-view motion: rotations "
        f"recovered to {metrics['rotation_rmse_deg']:.4f}° RMS",
        fontsize=10,
    )
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    volumes, metrics = run_example()
    plot(volumes, metrics, ROOT / "images" / "alignment-example.png")
    (ROOT / "images" / "alignment-example.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
