"""A moving sandstone plug before and after the plain ``tj.align`` call.

The sandstone core (``bench/phantoms/porous_rock.py``), voxelized 1.5 times finer than the
384-voxel reconstruction grid, is scanned in parallel beam over a half turn without noise while
it drifts smoothly by up to 4.5 voxels and slips by 6 voxels a third of the way through. The
scan is reconstructed as loaded, with the nominal geometry, and after ``tj.align(scan)``; both use
``tj.reconstruct``'s default method. Two slices through the sample, clear of its calcite,
pyrite and zircon grains, each with one enlarged region, share one gray range.

    uv run python examples/gallery/rock_alignment.py alignment-rock.png

The first run builds and caches the 576-voxel phantom, about nine minutes on the CPU; later
runs take about 20 s on an 8 GB laptop GPU. ``--size 64`` makes a quick small version.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import panel, phantom, save, style
import matplotlib.pyplot as plt
import numpy as np

import tomojax as tj

N = 384
TRUTH_N = 576
VIEWS = 180
CROP = 128


def motion(views: int, unit: float, seed: int) -> np.ndarray:
    """A per-view table (alpha, beta, phi, x, z, y): smooth drift and one slip, no rotation.

    ``unit`` is 1/128 of the field of view: the drift reaches about 1.5 units, the slip 2.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, views)
    table = np.zeros((views, 6))
    for column in (3, 4):
        phase = rng.uniform(0, 2 * np.pi, 2)
        drift = 0.9 * np.sin(2 * np.pi * t + phase[0]) + 0.6 * np.sin(4 * np.pi * t + phase[1])
        table[:, column] = unit * (
            drift + 2.0 * (np.arange(views) >= views // 3) * rng.choice((-1, 1))
        )
    return table


def simulate(seed: int = 0, size: int = N) -> tuple[tj.Scan, np.ndarray, float]:
    """The moving scan as loaded (nominal geometry), the finer truth, and the voxel size."""
    fine_n = size * TRUTH_N // N
    truth, meta = phantom("porous_rock", fine_n)
    fov = float(meta["fov_mm"])
    voxel = fov / size
    grid = tj.Grid(size, size, size, voxel, voxel, voxel)
    fine = tj.Grid(fine_n, fine_n, fine_n, *(3 * (fov / fine_n,)))
    detector = tj.Detector(size, size, voxel, voxel)
    geometry = tj.ParallelGeometry(grid, detector, np.linspace(0.0, 180.0, VIEWS, endpoint=False))
    moved = tj.Scan(np.zeros((VIEWS, size, size), np.float32), geometry).with_poses(
        motion(VIEWS, fov / 128, seed)
    )
    line_integrals = np.asarray(tj.project(moved.geometry, truth, grid=fine), np.float32)
    return tj.Scan(line_integrals, geometry), truth, voxel


def sharpest(image: np.ndarray, inside: np.ndarray, crop: int) -> tuple[slice, slice]:
    """The ``crop``-square window inside the sample with the strongest edges."""
    gy, gx = np.gradient(image)
    edges = np.hypot(gx, gy) * inside
    best, where = -1.0, (0, 0)
    step = max(1, crop // 8)
    for row in range(0, image.shape[0] - crop, step):
        for column in range(0, image.shape[1] - crop, step):
            if inside[row : row + crop, column : column + crop].all():
                score = edges[row : row + crop, column : column + crop].sum()
                if score > best:
                    best, where = score, (row, column)
    return slice(where[0], where[0] + crop), slice(where[1], where[1] + crop)


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
    bottom = int(filled[0] * size / truth.shape[0]) + size // 32
    # The second slice: the one nearest the middle with no grain denser than feldspar (calcite,
    # pyrite, zircon) within two slices, whose streaks would hide the grains, and a quarter of
    # the sample above the first.
    ratio = truth.shape[0] / size
    dense = truth.reshape(truth.shape[0], -1).max(axis=1) > 0.4
    clear = [
        z
        for z in range(bottom + size // 4, int(filled[-1] / ratio) - size // 16)
        if not dense[int((z - 2) * ratio) : int((z + 3) * ratio)].any()
    ]
    second = min(clear, key=lambda z: abs(z - size // 2))
    heights = (("Lowest slice through the sample", bottom), ("A slice higher up", second))
    top = float(np.percentile(aligned[second][inside], 99.5))

    side = CROP * size // N
    style()
    fig, axes = plt.subplots(2, 4, figsize=(9.0, 4.9))
    for (label, z), row in zip(heights, axes, strict=True):
        crop = sharpest(aligned[z], core, side)
        for (title, volume), (whole, zoom) in zip(
            (("As loaded", loaded), ("After tj.align(scan)", aligned)),
            ((row[0], row[1]), (row[2], row[3])),
            strict=True,
        ):
            panel(whole, volume[z], title, 0.0, top)
            whole.add_patch(
                plt.Rectangle(
                    (crop[1].start, crop[0].start),
                    side,
                    side,
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
