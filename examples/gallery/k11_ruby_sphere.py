"""Ruby sphere from Diamond K11 laminography scan 54013.

    uv run python examples/gallery/k11_ruby_sphere.py k11-54013.nxs k11-ruby-sphere.png

The default workflow aligns 1441 views with 4x4 detector binning: 2.16 micrometre
pixels become 8.64 micrometre voxels. The figure shows slices through the ruby sphere centre.
Use --binning 1 for 2.16 micrometre voxels, which need more than 8 GB of GPU memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import TYPE_CHECKING

from _common import save, style
import h5py
import matplotlib.patheffects as effects
import matplotlib.pyplot as plt
import numpy as np

import tomojax as tj
from tomojax.geometry import grid_volume_origin

if TYPE_CHECKING:
    from jax import Array

PITCH_UM = 2.16


def sampling_grid(binning: int) -> tj.Grid:
    """Initial voxel sampling; the automatic workflow chooses the physical region."""
    pitch = PITCH_UM * binning
    return tj.Grid(640 // binning, 640 // binning, 540 // binning, pitch, pitch, pitch)


def slice_figure(volume: np.ndarray | Array, grid: tj.Grid, output: Path) -> None:
    """Orthogonal slices through the ruby sphere centre, sharing one grayscale range."""
    pitch = np.array((grid.vx, grid.vy, grid.vz))
    origin = np.asarray(grid_volume_origin(grid))
    # Display bounds, not forward-model support. Keep original voxel samples.
    centre = np.array((-332.64, -60.48, 0))
    lower = centre - 345.6
    upper = centre + 345.6
    first = np.ceil((lower - origin) / pitch).astype(int)
    stop = np.ceil((upper - origin) / pitch).astype(int)
    lower = origin + (first - 0.5) * pitch
    upper = origin + (stop - 0.5) * pitch
    indices = np.rint((centre - origin) / pitch).astype(int)
    cuts = []
    for axis in (2, 1, 0):
        selection = [slice(int(a), int(b)) for a, b in zip(first, stop, strict=True)]
        selection[axis] = int(indices[axis])
        cuts.append(np.asarray(volume[tuple(selection[::-1])]))
    low, high = np.percentile(np.concatenate([cut.ravel() for cut in cuts]), (1, 99.5))
    style()
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.5), layout="constrained")
    for ax, cut, axis in zip(axes, cuts, (2, 1, 0), strict=True):
        horizontal, vertical = [k for k in range(3) if k != axis]
        extent = [lower[horizontal], upper[horizontal], lower[vertical], upper[vertical]]
        ax.imshow(cut, origin="lower", extent=extent, vmin=low, vmax=high)
        coordinate = origin[axis] + indices[axis] * pitch[axis]
        plane = "xyz"[horizontal] + "xyz"[vertical]
        ax.set_title(f"{plane} · {'xyz'[axis]} = {coordinate:.1f} µm")
        x = lower[horizontal] + 0.07 * (upper[horizontal] - lower[horizontal])
        y = lower[vertical] + 0.09 * (upper[vertical] - lower[vertical])
        stroke = [effects.withStroke(linewidth=3, foreground="black")]
        ax.plot([x, x + 200], [y, y], color="white", lw=2, path_effects=stroke)
        ax.text(x + 100, y + 28, "200 µm", color="white", ha="center", path_effects=stroke)
        ax.set_axis_off()
    fig.suptitle("Ruby sphere · K11 laminography · scan 54013")
    binning = grid.vx / PITCH_UM
    fig.supxlabel(
        f"{PITCH_UM:g} µm pixels · {binning:g}x{binning:g} binning · {grid.vx:.2f} µm voxels\n"
        "Slices through the ruby sphere centre · missing-cone artefacts remain",
        fontsize=8,
    )
    save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="the raw k11-54013.nxs acquisition")
    parser.add_argument("output", type=Path)
    parser.add_argument("--binning", type=int, choices=(1, 2, 4), default=4)
    args = parser.parse_args()
    with h5py.File(args.input) as source:
        sample = np.asarray(source["entry/instrument/EtherCAT/image_key"]) == 0
        angles = np.asarray(source["entry/imaging_sum/smaract_zrot"], np.float32)[sample]
    geometry = tj.LaminographyGeometry(
        sampling_grid(args.binning),
        tj.Detector(640, 540, PITCH_UM, PITCH_UM),
        angles,
        tilt_deg=-35.0,
    )
    start = time.perf_counter()
    scan = tj.load_frames(args.input, geometry=geometry).corrected(binning=args.binning)
    result = tj.align(scan)
    print(f"load/correct/align: {time.perf_counter() - start:.1f} s")
    automatic = result.info["automatic"]
    print("Automatic outcome:", automatic["status"], automatic["reasons"])
    print("Reconstruction grid:", result.scan.grid)
    slice_figure(result.volume, result.scan.grid, args.output)


if __name__ == "__main__":
    main()
