"""Laminography's missing cone on a sandstone slab with a known truth.

    uv run python examples/gallery/rock_missing_cone.py rock-missing-cone.png

A 0.25 mm thick slab cut from the sandstone core (``bench/phantoms/porous_rock.py``,
voxelised 1.5 times finer than the grid) is scanned in laminography: 720 views
over a full turn, the rotation axis tilted 35° from the slab's normal, with
Poisson noise. No view measures the frequencies in a cone about the axis, so
FBP elongates every grain along the axis; FISTA-TV with nonnegativity fills
part of the cone. The figure shows an in-plane slice and a vertical section of
the truth, FBP and FISTA-TV; errors are relative L2 against the truth inside
the slab.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import counts, downsample, panel, phantom, relative_error, save, style
import matplotlib.pyplot as plt
import numpy as np

import tomojax as tj

N = 384
TRUTH_N = 576
VIEWS = 720
TILT = -35.0
PHOTONS = 1e5


def simulate(seed: int = 0) -> tuple[tj.Scan, np.ndarray, np.ndarray, float]:
    """The scan, the truth on the grid, the slab mask and the slab thickness in mm."""
    rock, meta = phantom("porous_rock", TRUTH_N)
    fov = float(meta["fov_mm"])
    thickness = fov / 4
    heights = (np.arange(TRUTH_N) - TRUTH_N / 2 + 0.5) * fov / TRUTH_N
    slab = np.where((np.abs(heights) < thickness / 2)[:, None, None], rock, 0.0)
    voxel = fov / N
    grid = tj.Grid(N, N, N, voxel, voxel, voxel)
    fine = tj.Grid(TRUTH_N, TRUTH_N, TRUTH_N, *(3 * (fov / TRUTH_N,)))
    geometry = tj.LaminographyGeometry(
        grid,
        tj.Detector(N, N, voxel, voxel),
        np.linspace(0.0, 360.0, VIEWS, endpoint=False),
        tilt=TILT,
    )
    line_integrals = np.asarray(tj.project(geometry, slab.astype(np.float32), grid=fine))
    rng = np.random.default_rng(seed)
    measured = -np.log(np.maximum(counts(line_integrals, PHOTONS, rng), 1.0) / PHOTONS)
    truth = downsample(slab, N)
    centres = (np.arange(N) - N / 2 + 0.5) * voxel
    mask = np.broadcast_to((np.abs(centres) < thickness / 2)[:, None, None], truth.shape)
    return tj.Scan(measured.astype(np.float32), geometry), truth, mask, thickness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    scan, truth, mask, thickness = simulate()
    volumes = {
        "Truth": truth,
        "FBP": np.asarray(tj.reconstruct(scan, "fbp").volume, np.float32),
        "FISTA-TV": np.asarray(tj.reconstruct(scan, "fista", nonnegative=True).volume, np.float32),
    }
    style()
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.6))
    high = float(np.percentile(truth[mask], 99.5))
    for (name, volume), (top, bottom) in zip(volumes.items(), axes.T, strict=True):
        title = (
            name if name == "Truth" else f"{name}, error {relative_error(volume, truth, mask):.3f}"
        )
        panel(top, volume[N // 2], f"{title}\nin-plane slice", 0.0, high)
        panel(bottom, volume[:, N // 2], "vertical section", 0.0, high)
    fig.suptitle(
        f"Sandstone slab {thickness:.2f} mm thick, laminography at {abs(TILT):g}° · {VIEWS} views",
        y=0.995,
    )
    fig.tight_layout()
    save(fig, args.output)


if __name__ == "__main__":
    main()
