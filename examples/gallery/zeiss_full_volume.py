"""A volume larger than the GPU: the full Zeiss CuO_2 tomo-B scan at 1.13 micrometres.

    uv run python examples/gallery/zeiss_full_volume.py CuO_2_tomo-B.txrm VOLUME.npy OUT.png

The scan is 3201 views of 2028 x 2028 pixels; its native grid is 2028³ voxels
of 1.13 micrometres, 31 GiB in FP32. ``tj.reconstruct_slabs(frames, "fbp")``
corrects only the detector rows each slab of the volume needs and the blocks
are written to a memmap ``VOLUME.npy`` on disk, so neither the corrected
projections (52 GB) nor the volume is held in memory. The figure shows the
central axial slice and a vertical slice through the axis, each with one
enlarged region.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from _common import panel, save, style
import matplotlib.pyplot as plt
import numpy as np

import tomojax as tj

CROPS = ((900, 900, 400), (800, 900, 400))  # row, column, side for each slice


def reconstruct(path: Path, volume_path: Path) -> float:
    """Write the FDK volume to ``volume_path``; return the seconds it took."""
    frames = tj.load_frames(path)
    grid = frames.geometry.grid
    volume = np.lib.format.open_memmap(
        volume_path, mode="w+", dtype=np.float32, shape=(grid.nz, grid.ny, grid.nx)
    )
    start = time.perf_counter()
    with tj.reconstruct_slabs(frames, "fbp") as slabs:
        for z_slice, block in slabs:
            volume[z_slice] = block.volume
    volume.flush()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="CuO_2_tomo-B.txrm")
    parser.add_argument("volume", type=Path, help="where to write the 31 GiB volume")
    parser.add_argument("output", type=Path)
    parser.add_argument("--reuse", action="store_true", help="draw an existing VOLUME.npy")
    args = parser.parse_args()
    if not args.reuse:
        print(f"reconstructed in {reconstruct(args.input, args.volume):.0f} s")
    volume = np.load(args.volume, mmap_mode="r")
    n = volume.shape[0]
    slices = (
        ("Axial slice, centre", np.asarray(volume[n // 2])),
        ("Vertical slice through the axis", np.asarray(volume[:, n // 2])),
    )
    style()
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.4))
    low, high = np.percentile(slices[0][1], (0.5, 99.8))
    for (title, image), (row0, column0, side), row in zip(slices, CROPS, axes, strict=True):
        panel(row[0], image, title, low, high)
        row[0].add_patch(
            plt.Rectangle((column0, row0), side, side, fill=False, color="tab:orange", lw=0.8)
        )
        panel(row[1], image[row0 : row0 + side, column0 : column0 + side], "Enlarged", low, high)
    fig.suptitle(f"Zeiss CuO_2 tomo-B · 3201 views · {n}³ voxels of 1.13 µm, 31 GiB · FDK")
    fig.tight_layout()
    save(fig, args.output)


if __name__ == "__main__":
    main()
