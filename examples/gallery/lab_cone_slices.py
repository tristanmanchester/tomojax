"""Central slices of two lab cone-beam scans at their native voxel size.

    uv run python examples/gallery/lab_cone_slices.py ZEISS NIKON lab-cone-slices.png

The scans are a Zeiss Xradia acquisition (CuO_2 tomo-B, a copper oxide
specimen) and a Nikon X-Tek scan of seeds in a box, each read by
``tj.load_frames`` from the vendor's own files with the centre and shifts those
files record: the Zeiss scan's ``AutoRecon/CenterShift`` and per-view image
shifts, and for Nikon the ``_recon.xtekct`` that holds the centre Nikon's
reconstructor found. No alignment is run. Each is reconstructed by FDK in its
central plane, one voxel per detector pixel at the rotation axis, from the 64
detector rows around the centre, which cover the plane's rays in every view
whatever its recorded vertical shift. Each slice is shown whole and with one
enlarged region.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import panel, save, style
import matplotlib.pyplot as plt
import numpy as np

import tomojax as tj

# (row, column, side) of each enlarged region, in slice pixels.
CROPS = {
    "Zeiss Xradia": (800, 800, 400),
    "Nikon X-Tek": (350, 400, 300),
}


def central_slice(frames: tj.Frames) -> tuple[np.ndarray, float]:
    """FDK of the central plane at the native voxel size, and that size."""
    rows = frames.counts.shape[1]
    scan = frames.cropped(slice(rows // 2 - 32, rows // 2 + 32), slice(None)).corrected()
    native = frames.geometry.grid
    grid = tj.Grid(native.nx, native.ny, 1, native.vx, native.vy, native.vx)
    return np.asarray(tj.reconstruct(scan, "fbp", grid=grid).volume, np.float32)[0], native.vx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("zeiss", type=Path, help="CuO_2_tomo-B.txrm")
    parser.add_argument("nikon", type=Path, help="the seeds scan's _recon.xtekct file")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    images = {
        "Zeiss Xradia": central_slice(tj.load_frames(args.zeiss)),
        "Nikon X-Tek": central_slice(tj.load_frames(args.nikon, darks=0)),
    }
    style()
    fig, axes = plt.subplots(2, 2, figsize=(7, 6.8))
    for (name, (image, voxel)), (whole, zoom) in zip(images.items(), axes.T, strict=True):
        low, high = np.percentile(image, (0.5, 99.8))
        row, column, side = CROPS[name]
        panel(whole, image, f"{name} · {image.shape[1]} pixels of {voxel * 1000:.3g} µm", low, high)
        whole.add_patch(
            plt.Rectangle((column, row), side, side, fill=False, color="tab:orange", lw=0.8)
        )
        panel(zoom, image[row : row + side, column : column + side], "Enlarged", low, high)
    fig.suptitle("Lab cone beam: FDK of the central plane at the native voxel size")
    fig.tight_layout()
    save(fig, args.output)


if __name__ == "__main__":
    main()
