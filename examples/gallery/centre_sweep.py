r"""A centre sweep on a raw synchrotron scan: the same slice at three detector centres.

    uv run --with matplotlib python examples/gallery/centre_sweep.py test_data.h5 \
        docs/images/centre-sweep-aps.png

The input is tomocupy's test scan from APS 2-BM (720 views over a half turn on
22 rows of 1536 pixels, with flats and darks), which records no rotation
centre. ``tj.load`` corrects it; ``tj.interactive.resolve_sweep`` and
``sweep_planes`` reconstruct the central slice at each centre given in pixels
from the detector middle, as ``tomojax inspect --centres`` does. The figure
shows the same region of the slice on one grey scale at each centre.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import panel, save, style
import matplotlib.pyplot as plt
import numpy as np

import tomojax as tj
from tomojax import interactive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="a raw scan with flats and darks")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--centres", type=float, nargs="+", default=(-24.0, -15.0, -6.0), metavar="PX"
    )
    parser.add_argument("--plane", type=int, default=11, help="the axial plane to sweep")
    parser.add_argument(
        "--region",
        type=int,
        nargs=4,
        default=(470, 930, 200, 760),
        metavar=("ROW0", "ROW1", "COLUMN0", "COLUMN1"),
        help="the part of the slice to show, in slice pixels",
    )
    args = parser.parse_args()
    scan = tj.load(args.input)
    sweep = interactive.resolve_sweep(scan, "z", args.plane, args.centres, method="fbp")
    row0, row1, column0, column1 = args.region
    crops = [
        np.asarray(plane, np.float32)[row0:row1, column0:column1]
        for plane in interactive.sweep_planes(sweep)
    ]
    low, high = np.percentile(crops[len(crops) // 2], (0.5, 99.5))

    style()
    fig, axes = plt.subplots(1, len(crops), figsize=(3.0 * len(crops), 2.8))
    for ax, centre, crop in zip(np.atleast_1d(axes), args.centres, crops, strict=True):
        panel(ax, crop, f"centre {centre:+g} px", low, high)
    fig.suptitle("APS 2-BM test scan, one slice at three detector centres", y=0.995)
    fig.tight_layout()
    save(fig, args.output)


if __name__ == "__main__":
    main()
