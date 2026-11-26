"""Where a bubble meets the ruby sphere in K11 scan 54013, before and after ``tj.align``.

    uv run python examples/gallery/k11_sphere_alignment.py k11-54013.nxs alignment-k11-sphere.png

The scan is aligned with the plain ``tj.align(scan)`` on a 6.48 micrometer grid. The scan as
loaded and the aligned scan are then reconstructed by FBP in two slices at the detector's
2.16 micrometer pitch: through the ruby sphere's equator and 78 micrometers below it. Each slice
has one enlarged region where the large bubble meets the sphere; a row shares one gray range.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import panel, save, style
import h5py
import matplotlib.pyplot as plt
import numpy as np

import tomojax as tj

PITCH_UM = 2.16
VOXEL_UM = 6.48
SIDE = 864  # slice pixels at 2.16 um, 1.87 mm across
CENTER_UM = 22.68  # the sample's middle, in x and y
CROP = 120
# (label, slice height in um, crop's first row and column) for each row of the figure.
SLICES = (
    ("Through the sphere's equator", 3.24, (325, 321)),
    ("78 µm lower", -74.52, (342, 316)),
)


def load(path: Path) -> tj.Scan:
    """The scan as loaded: nominal geometry on a 704 x 704 x 234 grid of 6.48 um voxels."""
    with h5py.File(path) as source:
        sample = np.asarray(source["entry/instrument/EtherCAT/image_key"]) == 0
        angles = np.asarray(source["entry/imaging_sum/smaract_zrot"], np.float64)[sample]
    geometry = tj.LaminographyGeometry(
        tj.Grid(704, 704, 234, VOXEL_UM, VOXEL_UM, VOXEL_UM),
        tj.Detector(640, 540, PITCH_UM, PITCH_UM),
        angles,
        tilt=-35.0,
    )
    return tj.load_frames(path, geometry=geometry).corrected()


def planes(scan: tj.Scan) -> list[np.ndarray]:
    """FBP of each figure slice at the detector pitch."""
    images = []
    for _, height, _ in SLICES:
        grid = tj.Grid(
            SIDE, SIDE, 1, PITCH_UM, PITCH_UM, PITCH_UM, vol_center=[CENTER_UM, CENTER_UM, height]
        )
        images.append(np.asarray(tj.reconstruct(scan, "fbp", grid=grid).volume, np.float32)[0])
    return images


def figure(loaded: list[np.ndarray], aligned: list[np.ndarray], output: Path) -> None:
    """Whole slices and enlarged regions, as loaded beside after alignment."""
    style()
    fig, axes = plt.subplots(len(SLICES), 4, figsize=(9.0, 4.9))
    for (label, _, (row0, column0)), before, after, row in zip(
        SLICES, loaded, aligned, axes, strict=True
    ):
        low, high = np.percentile(before, (0.5, 99.5))
        crop = np.s_[row0 : row0 + CROP, column0 : column0 + CROP]
        for title, image, (whole, zoom) in (
            ("As loaded", before, (row[0], row[1])),
            ("After tj.align(scan)", after, (row[2], row[3])),
        ):
            panel(whole, image, title, low, high)
            whole.add_patch(
                plt.Rectangle((column0, row0), CROP, CROP, fill=False, color="tab:orange", lw=0.8)
            )
            panel(zoom, image[crop], "Enlarged", low, high)
        row[0].text(-0.08, 0.5, label, transform=row[0].transAxes, rotation=90,
                    ha="right", va="center")  # fmt: skip
    fig.suptitle(
        "Ruby sphere and bubble · K11 laminography · scan 54013 · 1441 views · 2.16 µm voxels",
        y=0.995,
    )
    fig.tight_layout()
    save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="the raw k11-54013.nxs acquisition")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    scan = load(args.input)
    loaded = planes(scan)
    figure(loaded, planes(tj.align(scan).scan), args.output)


if __name__ == "__main__":
    main()
