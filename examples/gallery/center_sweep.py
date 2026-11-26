r"""A center sweep on a raw synchrotron scan: the same slice at three detector centers.

    uv run --with matplotlib python examples/gallery/center_sweep.py test_data.h5 \
        docs/images/center-sweep-aps.png

The input is tomocupy's test scan from APS 2-BM (720 views over a half turn on
22 rows of 1536 pixels, with flats and darks), which records no rotation
center. ``tj.load`` corrects it; ``tj.interactive.center_sweep`` reconstructs
the central slice at each center given in pixels from the detector middle, as
``tomojax inspect --centers`` does. The figure
shows the same region of the slice on one gray scale at each center.
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
        "--centers", type=float, nargs="+", default=(-24.0, -15.0, -6.0), metavar="PX"
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
    planes = interactive.center_sweep(scan, "z", args.plane, args.centers, method="fbp")
    row0, row1, column0, column1 = args.region
    crops = list(np.asarray(planes, np.float32)[:, row0:row1, column0:column1])
    low, high = np.percentile(crops[len(crops) // 2], (0.5, 99.5))

    style()
    fig, axes = plt.subplots(1, len(crops), figsize=(3.0 * len(crops), 2.8))
    for ax, center, crop in zip(np.atleast_1d(axes), args.centers, crops, strict=True):
        panel(ax, crop, f"center {center:+g} px", low, high)
    fig.suptitle("APS 2-BM test scan, one slice at three detector centers", y=0.995)
    fig.tight_layout()
    save(fig, args.output)


if __name__ == "__main__":
    main()
