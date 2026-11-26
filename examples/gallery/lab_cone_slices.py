"""Central slices of three lab cone-beam scans at their native voxel size.

    uv run python examples/gallery/lab_cone_slices.py ZEISS NIKON SKYSCAN lab-cone-slices.png

The scans are a Zeiss Xradia acquisition (CuO_2 tomo-B, a copper oxide
specimen), a Nikon X-Tek scan of seeds in a box and a Bruker SkyScan 1172 scan of
grains (996 views over 199°, the axis at the detector middle), each read by
``tj.load_frames`` from the vendor's own files with the center and shifts those
files record: the Zeiss scan's ``AutoRecon/CenterShift`` and per-view image
shifts, and for Nikon the ``_recon.xtekct`` that holds the center Nikon's
reconstructor found. No alignment is run. Each is reconstructed by FDK in its
central plane, one voxel per detector pixel at the rotation axis, from the 64
detector rows around the center, which cover the plane's rays in every view
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
    "SkyScan": (1150, 760, 300),
}
# (row, column, side) of the square each whole slice is cropped to; the sample fills it.
VIEWS = {"SkyScan": (440, 460, 1100)}


def central_slice(frames: tj.Frames) -> tuple[np.ndarray, float]:
    """FDK of the central plane at the native voxel size, and that size."""
    rows = frames.counts.shape[1]
    scan = frames.cropped(slice(rows // 2 - 32, rows // 2 + 32), slice(None)).corrected()
    native = frames.geometry.grid
    grid = tj.Grid(native.nx, native.ny, 1, native.vx, native.vy, native.vx)
    return np.asarray(tj.reconstruct(scan, "fbp", grid=grid).volume, np.float32)[0], native.vx


def skyscan(log: Path) -> tj.Frames:
    """A SkyScan scan; its log gives no horizontal center, so the axis is the detector middle.

    The log's camera pixel and binning give the detector pitch and its optical-axis
    line the vertical center; the TIFFs are already flat-field corrected to 65535.
    """
    import configparser

    config = configparser.ConfigParser()
    config.read(log, encoding="latin-1")
    system, acquisition = config["System"], config["Acquisition"]
    columns, rows = int(acquisition["Number of Columns"]), int(acquisition["Number of Rows"])
    binning = float(acquisition["Camera binning"].split("x")[0])
    pitch = float(system["Camera Pixel Size (um)"]) * 1e-3 * binning
    v = (float(acquisition["Optical Axis (line)"]) - rows / 2) * pitch
    detector = tj.Detector(columns, rows, pitch, pitch, (0.0, v))
    return tj.load_frames(log, detector=detector, flats=65535, darks=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("zeiss", type=Path, help="CuO_2_tomo-B.txrm")
    parser.add_argument("nikon", type=Path, help="the seeds scan's _recon.xtekct file")
    parser.add_argument("skyscan", type=Path, help="the grains scan's .log file")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    images = {
        "Zeiss Xradia": central_slice(tj.load_frames(args.zeiss)),
        "Nikon X-Tek": central_slice(tj.load_frames(args.nikon, darks=0)),
        "SkyScan": central_slice(skyscan(args.skyscan)),
    }
    style()
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8))
    for (name, (image, voxel)), (whole, zoom) in zip(images.items(), axes.T, strict=True):
        top, left, extent = VIEWS.get(name, (0, 0, image.shape[0]))
        view = image[top : top + extent, left : left + extent]
        low, high = np.percentile(view, (0.5, 99.8))
        row, column, side = CROPS[name]
        title = f"{name} · {view.shape[1]} pixels of {voxel * 1000:.3g} µm"
        panel(whole, view, title, low, high)
        whole.add_patch(
            plt.Rectangle(
                (column - left, row - top), side, side, fill=False, color="tab:orange", lw=0.8
            )
        )
        panel(zoom, image[row : row + side, column : column + side], "Enlarged", low, high)
    fig.suptitle("Lab cone beam: FDK of the central plane at the native voxel size")
    fig.tight_layout()
    save(fig, args.output)


if __name__ == "__main__":
    main()
