"""Fourier reconstruction with its native response and with ``filter="ramp"``, beside FBP.

    uv run python examples/gallery/fourier_response.py full_rows1048-1111.nxs fourier-response.png

The input is the full-resolution cut of the Diamond I12 scan of Vo et al.
(Zenodo 3339629): 1801 views over a half turn on 2560-pixel detector rows, raw
counts with flats and darks. The script corrects detector row 1080 (row 32 of
the cut) with ``Rings()``, measures the rotation center with
``tj.align(scan, fit="center")`` and reconstructs the slice three ways on the
same grid: ``tj.reconstruct(scan, "fourier")``, the same with
``filter="ramp"``, and ``tj.reconstruct(scan, "fbp")``. The top row enlarges one
region of each slice on FBP's gray range; the bottom row shows where that region
lies and each Fourier slice minus FBP, both differences on one gray range
symmetric about zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import panel, save, style
import matplotlib.pyplot as plt
import numpy as np

import tomojax as tj
from tomojax.corrections import Rings

ROW = 32  # detector row 1080 of the full detector
CROP = (900, 1180, 200)  # row, column and side of the enlarged region, in slice pixels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="the I12 cut full_rows1048-1111.nxs")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    frames = tj.load_frames(args.input).cropped(slice(ROW, ROW + 1), slice(None))
    scan = tj.align(frames.corrected(Rings()), fit="center").scan
    slices = {
        title: np.asarray(tj.reconstruct(scan, method, **options).volume, np.float32)[0]
        for title, method, options in (
            ("FBP", "fbp", {}),
            ("Fourier, default", "fourier", {}),
            ('Fourier, filter="ramp"', "fourier", {"filter": "ramp"}),
        )
    }
    fbp = slices["FBP"]
    row0, column0, side = CROP
    region = np.s_[row0 : row0 + side, column0 : column0 + side]
    low, high = np.percentile(fbp[region], (0.5, 99.5))
    differences = {title: image - fbp for title, image in slices.items() if title != "FBP"}
    limit = np.percentile(np.abs(differences["Fourier, default"][region]), 99.5)
    for title, difference in differences.items():
        rms = np.sqrt(np.mean(difference[region] ** 2)) / np.std(fbp[region])
        print(f"{title} minus FBP: RMS {rms:.3f} of FBP's standard deviation in the region")

    style()
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 6.4))
    for ax, (title, image) in zip(axes[0], slices.items(), strict=True):
        panel(ax, image[region], title, low, high)
    whole = np.percentile(fbp, (0.5, 99.5))
    panel(axes[1, 0], fbp, "FBP, whole slice", *whole)
    axes[1, 0].add_patch(
        plt.Rectangle((column0, row0), side, side, fill=False, color="tab:orange", lw=0.8)
    )
    for ax, (title, difference) in zip(axes[1, 1:], differences.items(), strict=True):
        panel(ax, difference[region], f"{title} minus FBP", -limit, limit)
    fig.suptitle(
        f"Diamond I12, Vo et al. · {scan.projections.shape[0]} views · "
        f"enlarged region {side} pixels across",
        y=0.995,
    )
    fig.tight_layout()
    save(fig, args.output)


if __name__ == "__main__":
    main()
