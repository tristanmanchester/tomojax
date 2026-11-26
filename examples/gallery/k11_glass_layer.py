"""Glass spheres between PEKK plates in Diamond K11 laminography scan 54286.

    uv run python examples/gallery/k11_glass_layer.py k11-54286.nxs k11-glass-layer.png

The 1441 views are aligned with the plain ``tj.align(scan)`` on a 6.48 micrometer
grid, then reconstructed by FBP at the detector's 2.16 micrometer pitch: one
in-plane slice through the sphere layer and two vertical sections through the
plates. The rotation axis leans towards the beam by +35 degrees in TomoJAX's
convention. The three planes share one gray range.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import save, style
from _k11 import PITCH_UM, load, plane
import matplotlib.patheffects as effects
import matplotlib.pyplot as plt
import numpy as np

import tomojax as tj

LAYER_UM = 12.96  # height of the in-plane slice through the sphere layer
ACROSS = 704  # in-plane slice pixels, 1.52 mm
WIDE, TALL = 926, 200  # vertical sections, 2.0 mm by 0.43 mm


def planes(scan: tj.Scan) -> list[tuple[str, np.ndarray]]:
    """The in-plane slice and the vertical sections at y = 0 and x = 0."""
    return [
        (f"xy · z = {LAYER_UM:.0f} µm", plane(scan, (0, 0, LAYER_UM), (ACROSS, ACROSS, 1))),
        ("xz · y = 0", plane(scan, (0, 0, 15.0), (WIDE, 1, TALL))),
        ("yz · x = 0", plane(scan, (0, 0, 15.0), (1, WIDE, TALL))),
    ]


def figure(cuts: list[tuple[str, np.ndarray]], output: Path) -> None:
    """The in-plane slice beside the two sections, with 200 um scale bars."""
    low, high = np.percentile(np.concatenate([cut.ravel() for _, cut in cuts]), (1, 99.5))
    style()
    fig = plt.figure(figsize=(11, 5), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1, 1.3))
    axes = [fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])]
    stroke = [effects.withStroke(linewidth=3, foreground="black")]
    bar = 200 / PITCH_UM
    for ax, (title, cut) in zip(axes, cuts, strict=True):
        ax.imshow(cut, origin="lower", vmin=low, vmax=high)
        x0, y0 = 0.05 * cut.shape[1], 0.1 * cut.shape[0]
        ax.plot([x0, x0 + bar], [y0, y0], color="white", lw=2, path_effects=stroke)
        ax.text(x0 + bar / 2, y0 + 0.04 * cut.shape[0], "200 µm", color="white", ha="center",
                path_effects=stroke)  # fmt: skip
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle(
        "Glass spheres between PEKK plates · K11 laminography · scan 54286 · 2.16 µm voxels"
    )
    save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="the raw k11-54286.nxs acquisition")
    parser.add_argument("output", type=Path)
    parser.add_argument("--aligned", type=Path, help="reuse a scan saved from tj.align")
    args = parser.parse_args()
    scan = tj.load(args.aligned) if args.aligned else tj.align(load(args.input, 35.0)).scan
    figure(planes(scan), args.output)


if __name__ == "__main__":
    main()
