"""A moving sandstone plug before and after the plain ``tj.align`` call.

The sandstone core (``bench/phantoms/porous_rock.py``) is scanned in parallel beam over a
half turn while it drifts smoothly by about a voxel and a half and slips by two voxels a third
of the way through. The scan is reconstructed as loaded, with the nominal geometry, and after
``tj.align(scan)``; both use ``tj.reconstruct``'s default method. Two slices through the
sample, each with one enlarged region, share one grey range.

    uv run python examples/gallery/rock_alignment.py alignment-rock.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import counts, panel, phantom, save, style
import matplotlib.pyplot as plt
import numpy as np

import tomojax as tj

N = 128
TRUTH_N = 192
VIEWS = 180
PHOTONS = 1e5
CROP = 48


def motion(views: int, voxel: float, seed: int) -> np.ndarray:
    """A per-view table (alpha, beta, phi, x, z, y): smooth drift and one slip, no rotation."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, views)
    table = np.zeros((views, 6))
    for column in (3, 4):
        phase = rng.uniform(0, 2 * np.pi, 2)
        drift = 0.9 * np.sin(2 * np.pi * t + phase[0]) + 0.6 * np.sin(4 * np.pi * t + phase[1])
        table[:, column] = voxel * (
            drift + 2.0 * (np.arange(views) >= views // 3) * rng.choice((-1, 1))
        )
    return table


def simulate(seed: int = 0, size: int = N) -> tuple[tj.Scan, np.ndarray, float]:
    """The moving scan as loaded (nominal geometry), the truth on its grid, and the voxel size."""
    fine_n = size * TRUTH_N // N
    truth, meta = phantom("porous_rock", fine_n)
    fov = float(meta["fov_mm"])
    voxel = fov / size
    grid = tj.Grid(size, size, size, voxel, voxel, voxel)
    fine = tj.Grid(fine_n, fine_n, fine_n, *(3 * (fov / fine_n,)))
    detector = tj.Detector(size, size, voxel, voxel)
    geometry = tj.ParallelGeometry(grid, detector, np.linspace(0.0, 180.0, VIEWS, endpoint=False))
    moved = tj.Scan(np.zeros((VIEWS, size, size), np.float32), geometry).with_poses(
        motion(VIEWS, voxel, seed)
    )
    line_integrals = np.asarray(tj.project(moved.geometry, truth, grid=fine))
    rng = np.random.default_rng(seed + 1)
    measured = -np.log(np.maximum(counts(line_integrals, PHOTONS, rng), 1.0) / PHOTONS)
    return tj.Scan(measured.astype(np.float32), geometry), truth, voxel


def sharpest(image: np.ndarray, inside: np.ndarray) -> tuple[slice, slice]:
    """The CROP-square window inside the sample with the strongest edges."""
    gy, gx = np.gradient(image)
    edges = np.hypot(gx, gy) * inside
    best, where = -1.0, (0, 0)
    for row in range(0, image.shape[0] - CROP, 4):
        for column in range(0, image.shape[1] - CROP, 4):
            if inside[row : row + CROP, column : column + CROP].all():
                score = edges[row : row + CROP, column : column + CROP].sum()
                if score > best:
                    best, where = score, (row, column)
    return slice(where[0], where[0] + CROP), slice(where[1], where[1] + CROP)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=N)
    args = parser.parse_args()
    scan, truth, voxel = simulate(size=args.size)
    loaded = np.asarray(tj.reconstruct(scan).volume)
    aligned = np.asarray(tj.reconstruct(tj.align(scan).scan).volume)

    # Slices through the sample: its lowest well-filled slice and the middle one.
    size = loaded.shape[0]
    centers = np.arange(size) - size / 2 + 0.5
    radius = np.hypot(*np.meshgrid(centers, centers, indexing="ij"))
    inside = radius < 0.42 * size
    core = radius < 0.33 * size  # crops on the sample's grains, not its outer edge
    filled = np.flatnonzero(truth.reshape(truth.shape[0], -1).mean(axis=1) > 0.5 * truth.mean())
    bottom = int(filled[0] * size / truth.shape[0]) + 4
    heights = (("Lowest slice through the sample", bottom), ("Middle slice", size // 2))
    top = float(np.percentile(aligned[size // 2][inside], 99.5))

    style()
    fig, axes = plt.subplots(2, 4, figsize=(9.0, 4.9))
    for (label, z), row in zip(heights, axes, strict=True):
        crop = sharpest(aligned[z], core)
        for (title, volume), (whole, zoom) in zip(
            (("As loaded", loaded), ("After tj.align(scan)", aligned)),
            ((row[0], row[1]), (row[2], row[3])),
            strict=True,
        ):
            panel(whole, volume[z], title, 0.0, top)
            whole.add_patch(
                plt.Rectangle(
                    (crop[1].start, crop[0].start),
                    CROP,
                    CROP,
                    fill=False,
                    color="tab:orange",
                    lw=0.8,
                )
            )
            panel(zoom, volume[z][crop], "Enlarged", 0.0, top)
        row[0].text(-0.08, 0.5, label, transform=row[0].transAxes, rotation=90,
                    ha="right", va="center")  # fmt: skip
    fig.suptitle(
        f"Sandstone plug drifting and slipping during a parallel-beam half turn: {VIEWS} views, "
        f"{1000 * voxel:.1f} µm voxels",
        y=0.995,
    )
    fig.tight_layout()
    save(fig, args.output)


if __name__ == "__main__":
    main()
