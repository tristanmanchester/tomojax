"""One real slice at the detector middle and at the center ``tj.align`` measures.

    uv run python examples/gallery/center_fit.py full_rows1048-1111.nxs center-fit.png

The input is the full-resolution cut of the Diamond I12 scan of Vo et al.
(Zenodo 3339629): 1801 views over a half turn on 2560-pixel detector rows, raw
counts with flats and darks. The file records no rotation center, so loading
places it at the detector middle. Detector row 1080 (row 32 of the cut) is
corrected with ``Rings()``, the center is measured with
``tj.align(scan, fit="center")``, and the slice is reconstructed by the default
``tj.reconstruct`` at both centers. The top row shows the whole slice, the
bottom row one enlarged region away from the rotation axis, all on one gray
range.
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
CROP = (1900, 700, 260)  # row, column and side of the enlarged region, in slice pixels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="the I12 cut full_rows1048-1111.nxs")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    frames = tj.load_frames(args.input).cropped(slice(ROW, ROW + 1), slice(None))
    loaded = frames.corrected(Rings())
    measured = tj.align(loaded, fit="center").scan
    offset = measured.detector.center[0] / measured.detector.du
    images = {
        title: np.asarray(tj.reconstruct(scan).volume, np.float32)[0]
        for title, scan in (
            ("Center at the detector middle", loaded),
            (f'fit="center": {offset:+.1f} pixels', measured),
        )
    }
    row0, column0, side = CROP
    region = np.s_[row0 : row0 + side, column0 : column0 + side]
    low, high = np.percentile(list(images.values())[1][region], (0.5, 99.5))

    style()
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 7.0))
    for (title, image), (top, bottom) in zip(images.items(), axes.T, strict=True):
        panel(top, image, title, low, high)
        top.add_patch(
            plt.Rectangle((column0, row0), side, side, fill=False, color="tab:orange", lw=0.8)
        )
        panel(bottom, image[region], "Enlarged", low, high)
    fig.suptitle(f"Diamond I12, Vo et al. · {image.shape[0]} pixels across", y=0.995)
    fig.tight_layout()
    save(fig, args.output)


if __name__ == "__main__":
    main()
