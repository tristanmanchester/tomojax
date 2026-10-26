"""A lab cone-beam scan of a cell wider than the detector.

A scaled 18650 lithium-ion cell (``bench/phantoms/battery.py``) is scanned in
cone beam at eight times magnification over a full turn, once with a detector
covering the whole cell and once with a detector 60% as wide, shifted so that
it sees one side of the cell and a little past the rotation axis. FDK weights
the doubly measured rays near the axis (Wang, 2002), so the offset scan
reconstructs the whole cross-section: a field of view nearly twice the
detector's width.

    uv run python examples/gallery/battery_offset_detector.py images/gallery/battery-offset.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import counts, downsample, panel, phantom, relative_error, save, style
import matplotlib.pyplot as plt
import numpy as np

import tomojax as tj

N = 256  # reconstruction voxels per side
TRUTH_N = 384  # the phantom is projected from a finer grid
VIEWS = 720
PHOTONS = 5000.0
MAGNIFICATION = 8.0
SOURCE_TO_AXIS = 60.0


def scan(
    truth: np.ndarray, fov: float, columns: int, offset_mm: float, rng: np.random.Generator
) -> tj.Scan:
    """Scan ``truth`` with ``columns`` detector pixels, centred ``offset_mm`` off the axis."""
    voxel = fov / N
    grid = tj.Grid(N, N, N, voxel, voxel, voxel)
    fine = tj.Grid(TRUTH_N, TRUTH_N, TRUTH_N, *(3 * (fov / TRUTH_N,)))
    pitch = voxel * MAGNIFICATION
    detector = tj.Detector(columns, 300, pitch, pitch, (offset_mm * MAGNIFICATION, 0.0))
    beam = tj.ConeBeam(SOURCE_TO_AXIS, SOURCE_TO_AXIS * MAGNIFICATION)
    angles = np.linspace(0.0, 360.0, VIEWS, endpoint=False)
    geometry = tj.ConeGeometry(grid, detector, angles, beam)
    line_integrals = np.asarray(tj.project(geometry, truth, grid=fine))
    measured = -np.log(np.maximum(counts(line_integrals, PHOTONS, rng), 1.0) / PHOTONS)
    return tj.Scan(measured.astype(np.float32), geometry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    truth, meta = phantom("battery", TRUTH_N)
    fov = float(meta["fov_mm"])
    rng = np.random.default_rng(0)
    full = scan(truth, fov, 300, 0.0, rng)
    # 180 columns of 35 µm at the axis span -1.15 to +5.15 mm: one side of the
    # 8.5 mm cell and 1.15 mm past the axis.
    offset = scan(truth, fov, 180, 2.0, rng)
    volumes = {
        name: np.asarray(tj.reconstruct(s, "fbp").volume)
        for name, s in (("full", full), ("offset", offset))
    }
    truth = downsample(truth, N)

    style()
    centres = (np.arange(N) - N / 2 + 0.5) * fov / N
    x, y = np.meshgrid(centres, centres, indexing="ij")
    mask = np.broadcast_to((np.hypot(x, y) < 4.3)[None], truth.shape)
    z, top = N // 2 - N // 8, 0.3
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.2))
    columns = (
        ("Phantom", truth),
        (
            f"full detector, error {relative_error(volumes['full'], truth, mask):.3f}",
            volumes["full"],
        ),
        (
            f"60% detector, offset, {relative_error(volumes['offset'], truth, mask):.3f}",
            volumes["offset"],
        ),
    )
    for (title, volume), (xy_ax, xz_ax) in zip(columns, axes.T, strict=True):
        panel(xy_ax, volume[z], title, 0.0, top)
        panel(xz_ax, volume[:, N // 2, :], "", 0.0, top)
    # Rays through the disc round the axis reach the offset detector twice a turn.
    axes[0, 2].add_patch(
        plt.Circle((N / 2, N / 2), 1.15 * N / fov, fill=False, color="tab:orange", ls="--", lw=0.8)
    )
    fig.suptitle(
        f"Lab cone-beam scan of a lithium-ion cell: {VIEWS} views, "
        f"{1000 * fov / N:.0f} µm voxels; the dashed circle is seen twice a turn",
        y=0.995,
    )
    fig.tight_layout()
    save(fig, args.output)


if __name__ == "__main__":
    main()
