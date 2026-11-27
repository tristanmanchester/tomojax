"""SIRT with and without momentum at the same number of updates, beside CGLS and FDK.

    uv run python examples/gallery/sirt_momentum.py Walnut1 sirt-momentum.png

The input is FIPS walnut 1 (Der Sarkissian et al. 2019, Zenodo 2686726), read with the
authors' recorded geometry: all 1200 views of its middle cone orbit, the
detector binned 2 x 2, on the authors' 501³ grid of 0.1 mm. Every method
reconstructs with ``region="cyl"``, the cylinder that every view sees:

- ``tj.reconstruct(scan, "fbp")``, which is FDK on a cone scan;
- ``tj.reconstruct(scan, "cgls")`` and ``tj.reconstruct(scan, "sirt")`` with no
  iteration count, so each stops at the estimated projection discrepancy or
  after 50 updates, and the default SIRT uses momentum;
- ``tj.reconstruct(scan, "sirt", iterations=N)``, plain SIRT with the number of
  updates N the default SIRT took.

The top row shows the central slice, the bottom row one enlarged region, all
on one gray range.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from _common import panel, save, style
import matplotlib.pyplot as plt
import numpy as np

import tomojax as tj

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _walnut import load_scan

CROP = (170, 170, 160)  # row, column and side of the enlarged region, in slice pixels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("walnut", type=Path, help="the unzipped Walnut1 folder")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    scan = load_scan(args.walnut, orbits=(2,), every=1, binning=2, size=501)

    def central(recon: tj.Reconstruction) -> np.ndarray:
        volume = np.asarray(recon.volume, np.float32)
        return volume[volume.shape[0] // 2]

    sirt = tj.reconstruct(scan, "sirt", region="cyl")
    updates = sirt.info["effective_iterations"]
    cgls = tj.reconstruct(scan, "cgls", region="cyl")
    for name, recon in (("SIRT", sirt), ("CGLS", cgls)):
        info = recon.info
        print(f"{name}: {info['effective_iterations']} updates, {info['termination']}")
    slices = {
        "FDK": central(tj.reconstruct(scan, "fbp", region="cyl")),
        f"CGLS\n{cgls.info['effective_iterations']} updates": central(cgls),
        f"SIRT without momentum\n{updates} updates": central(
            tj.reconstruct(scan, "sirt", iterations=updates, region="cyl")
        ),
        f"SIRT, default momentum\n{updates} updates": central(sirt),
    }
    row0, column0, side = CROP
    low, high = np.percentile(list(slices.values())[1], (0.5, 99.8))

    style()
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.4))
    for (title, image), (top, bottom) in zip(slices.items(), axes.T, strict=True):
        panel(top, image, title, low, high, pixels=300)
        top.add_patch(
            plt.Rectangle((column0, row0), side, side, fill=False, color="tab:orange", lw=0.8)
        )
        panel(bottom, image[row0 : row0 + side, column0 : column0 + side], "", low, high)
    fig.suptitle(
        "FIPS walnut 1, central slice · middle orbit, 1200 cone-beam views · 0.1 mm voxels",
        y=0.995,
    )
    fig.tight_layout()
    save(fig, args.output)


if __name__ == "__main__":
    main()
