"""FBP beside FISTA-TV on Diamond K11 laminography scan 54286.

    uv run python examples/gallery/k11_fbp_fista.py k11-54286.nxs k11-fbp-fista.png

The scan is aligned with the plain ``tj.align(scan)``, then reconstructed on its
own 704 x 704 x 234 grid of 6.48 micrometre voxels (4.56 by 4.56 by 1.52 mm):
by FBP, the default, and by FISTA-TV with 100 iterations, nonnegativity and its
automatic TV weight, as the laminography guide recommends. FISTA needs the
whole sample inside its grid, and at the detector's 2.16 micrometres that grid
does not fit on an 8 GB GPU. Each row shows an in-plane slice through the
glass-sphere layer, an enlarged region of it and a vertical section through
the plates; each row has its own grey range.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from _common import panel, save, style
from _k11 import load
import matplotlib.pyplot as plt
import numpy as np

import tomojax as tj

LAYER = 119  # the grid slice through the sphere layer, z = 16 um
INNER = np.s_[244:460, 244:460]  # the sample's 1.4 mm across the in-plane slice
CROP = (80, 80, 60)  # row, column and side of the enlarged region within INNER


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="the raw k11-54286.nxs acquisition")
    parser.add_argument("output", type=Path)
    parser.add_argument("--aligned", type=Path, help="reuse a scan saved from tj.align")
    args = parser.parse_args()
    scan = tj.load(args.aligned) if args.aligned else tj.align(load(args.input, 35.0)).scan
    volumes = {}
    for name, method, options in (
        ("FBP", "fbp", {}),
        ("FISTA-TV, 100 iterations", "fista", {"iterations": 100, "nonnegative": True}),
    ):
        start = time.perf_counter()
        volumes[name] = np.asarray(tj.reconstruct(scan, method, **options).volume)
        print(f"{name}: {time.perf_counter() - start:.0f} s")
    style()
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), gridspec_kw={"width_ratios": (1, 1, 1.8)})
    row0, column0, side = CROP
    for (name, volume), row in zip(volumes.items(), axes, strict=True):
        image = volume[LAYER][INNER]
        low, high = np.percentile(image, (0.5, 99.5))
        panel(row[0], image, f"{name}\nin-plane slice", low, high)
        row[0].add_patch(
            plt.Rectangle((column0, row0), side, side, fill=False, color="tab:orange", lw=0.8)
        )
        panel(row[1], image[row0 : row0 + side, column0 : column0 + side], "Enlarged", low, high)
        panel(row[2], volume[:, 352, INNER[1]], "Vertical section", low, high)
    fig.suptitle(
        "Glass spheres between PEKK plates · K11 laminography · scan 54286 · 6.48 µm voxels"
    )
    fig.tight_layout()
    save(fig, args.output)


if __name__ == "__main__":
    main()
