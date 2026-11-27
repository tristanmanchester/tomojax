"""FBP beside FISTA with its automatic TV weight, on a noisy real slice.

    uv run python examples/gallery/fista_tv_weight.py k11-52567-row1080.nxs fista-tv-weight.png

The input is detector row 1080 of Diamond K11 (DIAD) scan 52567, visit
mg40414-1: raw counts of 8000 views over a half turn on 2560 pixels of
0.54 µm, with flats and darks, written as NXtomo. The row is corrected with
``Rings()`` (automatic FISTA expects rings removed first), its rotation center
is measured with ``tj.align(scan, fit="center")``, and the slice is
reconstructed by ``tj.reconstruct(scan, "fbp")`` and by
``tj.reconstruct(scan, "fista")`` with no TV weight or iteration count given.
The top row shows one region of each slice, the bottom row an enlarged part of
it, all on one gray range.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import panel, save, style
import matplotlib.pyplot as plt
import numpy as np

import tomojax as tj
from tomojax.corrections import Rings

REGION = (1500, 1600, 400)  # row, column and side of the region, in slice pixels
ENLARGED = (200, 40, 130)  # row, column and side of the enlarged part, within REGION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="one detector row of K11 scan 52567")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    scan = tj.align(tj.load_frames(args.input).corrected(Rings()), fit="center").scan
    fbp = tj.reconstruct(scan, "fbp")
    fista = tj.reconstruct(scan, "fista")
    info = fista.info
    print(
        f"FISTA: weight {info['resolved_tv_weight']:.3g}, "
        f"{info['effective_iterations']} updates, early stop {info['early_stop']}"
    )
    row0, column0, side = REGION
    slices = {
        "FBP": np.asarray(fbp.volume, np.float32)[0][row0 : row0 + side, column0 : column0 + side],
        "FISTA, automatic TV weight": np.asarray(fista.volume, np.float32)[0][
            row0 : row0 + side, column0 : column0 + side
        ],
    }
    low, high = np.percentile(slices["FISTA, automatic TV weight"], (0.5, 99.5))
    um = 1000 * scan.detector.du

    style()
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.7))
    r, c, s = ENLARGED
    for (title, image), (top, bottom) in zip(slices.items(), axes.T, strict=True):
        panel(top, image, f"{title}\n{side * um:.0f} µm across", low, high)
        top.add_patch(plt.Rectangle((c, r), s, s, fill=False, color="tab:orange", lw=0.8))
        panel(bottom, image[r : r + s, c : c + s], f"Enlarged, {s * um:.0f} µm across", low, high)
    fig.suptitle(
        f"Diamond K11 scan 52567, one detector row · {scan.projections.shape[0]} views · "
        f"{um:.2f} µm pixels",
        y=0.995,
    )
    fig.tight_layout()
    save(fig, args.output)


if __name__ == "__main__":
    main()
