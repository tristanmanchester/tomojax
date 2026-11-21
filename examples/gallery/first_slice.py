r"""A first slice from a raw synchrotron scan, as the quickstart reaches it.

    uv run python examples/gallery/first_slice.py full_rows1048-1111.nxs i12-first-slice.png \
        --rows 24 40 --center -138 --crop 1500 1100 --title "Diamond I12, Vo et al."
    uv run python examples/gallery/first_slice.py 159815.nxs i13-first-slice.png \
        --view 650 700 1250 --crop 1350 1000

The script follows the quickstart on an NXtomo file of raw counts with its
flats and darks: ``tj.load_frames``, a crop to 16 detector rows,
``corrected(Stripes(9), Stripes(51, varying=True))`` for the rings, then FBP.
The rotation centre is the recorded one given by ``--center`` (in pixels),
or without it ``tj.align(fit="center", method="opposite-views")`` finds it.
The figure shows one raw frame with the cropped rows marked, the corrected
sinogram of the middle row and its slice at the detector's resolution, with
one enlarged region. Diamond I13 scan 159815 is 3001 views over a full turn
on a 2560 x 2160 detector; the I12 cut of Vo et al. (Zenodo 3339629) is 1801
views over a half turn on 64 rows of 2560 pixels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import panel, save, style
import matplotlib.pyplot as plt
import numpy as np

import tomojax as tj
from tomojax.corrections import Stripes

CROP = 400


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="an NXtomo file of raw counts")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--rows", type=int, nargs=2, default=(1072, 1088), metavar=("FIRST", "STOP")
    )
    parser.add_argument("--center", type=float, help="the recorded rotation centre, pixels")
    parser.add_argument(
        "--crop", type=int, nargs=2, default=(1400, 1000), metavar=("ROW", "COLUMN")
    )
    parser.add_argument(
        "--view",
        type=int,
        nargs=3,
        metavar=("ROW", "COLUMN", "SIDE"),
        help="the square of the slice the sample fills, in slice pixels",
    )
    parser.add_argument("--title", default="Diamond I13 scan 159815")
    args = parser.parse_args()
    rows = slice(*args.rows)
    frames = tj.load_frames(args.input)
    if args.center is not None:
        d = frames.geometry.detector
        detector = tj.Detector(d.nu, d.nv, d.du, d.dv, center=(args.center * d.du, 0.0))
        frames = tj.load_frames(args.input, detector=detector)
    raw = np.asarray(frames.selected([0]).counts, np.float32)[0]
    scan = frames.cropped(rows, slice(None)).corrected(Stripes(9), Stripes(51, varying=True))
    if args.center is None:
        scan = tj.align(scan, fit="center", method="opposite-views").scan
    print("detector centre:", scan.detector.center)
    middle = (rows.stop - rows.start) // 2
    sinogram = np.asarray(scan.projections, np.float32)[:, middle]
    image = np.asarray(tj.reconstruct(scan, "fbp").volume, np.float32)[middle]

    style()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), gridspec_kw={"width_ratios": (1.2, 1, 1, 1)})
    panel(axes[0], raw, "Raw frame, counts", *np.percentile(raw, (0.5, 99.5)))
    axes[0].axhspan(rows.start, rows.stop, color="tab:orange", alpha=0.6)
    axes[0].set_aspect("auto")
    panel(axes[1], sinogram, "Corrected sinogram, one row", *np.percentile(sinogram, (0.5, 99.5)))
    axes[1].set_aspect("auto")
    top, left, side = args.view if args.view else (0, 0, image.shape[0])
    whole = image[top : top + side, left : left + side]
    low, high = np.percentile(whole, (0.5, 99.5))
    row0, column0 = args.crop
    panel(axes[2], whole, "FBP slice, the sample", low, high)
    axes[2].add_patch(
        plt.Rectangle(
            (column0 - left, row0 - top), CROP, CROP, fill=False, color="tab:orange", lw=0.8
        )
    )
    panel(axes[3], image[row0 : row0 + CROP, column0 : column0 + CROP], "Enlarged", low, high)
    views = scan.projections.shape[0]
    fig.suptitle(f"{args.title} · {views} views · {raw.shape[1]}-pixel detector rows", y=0.995)
    fig.tight_layout()
    save(fig, args.output)


if __name__ == "__main__":
    main()
