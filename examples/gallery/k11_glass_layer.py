"""Glass spheres between PEKK plates in Diamond K11 laminography scan 54286.

    uv run python examples/gallery/k11_glass_layer.py k11-54286.nxs k11-glass-layer.png

The default workflow aligns 1441 views with 4x4 detector binning: 2.16 micrometre
pixels become 8.64 micrometre voxels. The figure shows the sphere layer in-plane
and two cross-sections through the plates.
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
    """An in-plane overview and two enlarged cross-sections, with physical scale bars."""
    pitch = np.array((grid.vx, grid.vy, grid.vz))
    origin = np.asarray(grid_volume_origin(grid))
    shape = np.array((grid.nx, grid.ny, grid.nz))
    indices = np.rint((np.array((0, 0, 12.96)) - origin) / pitch).astype(int)
    cuts, extents = [], []
    for axis in (2, 1, 0):
        first, stop = np.zeros(3, dtype=int), shape.copy()
        if axis != 2:
            # Enlarge the central plate cross-sections without changing reconstruction.
            first = np.maximum(first, np.ceil(((-1000, -1000, -200) - origin) / pitch).astype(int))
            stop = np.minimum(stop, np.ceil(((1000, 1000, 230) - origin) / pitch).astype(int))
        selection = [slice(int(a), int(b)) for a, b in zip(first, stop, strict=True)]
        selection[axis] = int(indices[axis])
        cuts.append(np.asarray(volume[tuple(selection)]))
        horizontal, vertical = [k for k in range(3) if k != axis]
        lower = origin + (first - 0.5) * pitch
        upper = origin + (stop - 0.5) * pitch
        extents.append([lower[horizontal], upper[horizontal], lower[vertical], upper[vertical]])
    low, high = np.percentile(np.concatenate([cut.ravel() for cut in cuts]), (1, 99.5))
    style()
    fig = plt.figure(figsize=(11, 5), layout="constrained")
    panels = fig.add_gridspec(2, 2)
    axes = [
        fig.add_subplot(panels[:, 0]),
        fig.add_subplot(panels[0, 1]),
        fig.add_subplot(panels[1, 1]),
    ]
    for ax, cut, extent, axis in zip(axes, cuts, extents, (2, 1, 0), strict=True):
        horizontal, vertical = [k for k in range(3) if k != axis]
        plane = "xyz"[horizontal] + "xyz"[vertical]
        coordinate = origin[axis] + indices[axis] * pitch[axis]
        ax.imshow(cut.T, origin="lower", extent=extent, vmin=low, vmax=high)
        ax.set_title(f"{plane} · {'xyz'[axis]} = {coordinate:.1f} µm")
        x = extent[0] + 0.06 * (extent[1] - extent[0])
        y = extent[2] + 0.09 * (extent[3] - extent[2])
        stroke = [effects.withStroke(linewidth=3, foreground="black")]
        ax.plot([x, x + 200], [y, y], color="white", lw=2, path_effects=stroke)
        ax.text(
            x + 100,
            y + 0.03 * (extent[3] - extent[2]),
            "200 µm",
            color="white",
            ha="center",
            path_effects=stroke,
        )
        ax.set_axis_off()
    fig.suptitle("Glass spheres between PEKK plates · K11 laminography · scan 54286")
    binning = grid.vx / PITCH_UM
    fig.supxlabel(
        f"{PITCH_UM:g} µm pixels · {binning:g}x{binning:g} binning · {grid.vx:.2f} µm voxels\n"
        "In-plane slice and enlarged cross-sections · missing-cone artefacts remain",
        fontsize=8,
    )
    save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="the raw k11-54286.nxs acquisition")
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
