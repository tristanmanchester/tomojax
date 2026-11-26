"""Ruby sphere from Diamond K11 laminography scan 54013.

    uv run python examples/gallery/k11_ruby_sphere.py k11-54013.nxs k11-ruby-sphere.png

The 1441 views are aligned with the plain ``tj.align(scan)`` on a 6.48 micrometer
grid, then reconstructed by FBP at the detector's 2.16 micrometer pitch in three
orthogonal planes through the ruby sphere's center. The planes share one gray range.
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

CENTER_UM = (-345.0, -60.0, 0.0)  # the sphere's center after alignment, x, y, z
SIDE = 320  # plane pixels, 691 um across


def planes(scan: tj.Scan) -> list[tuple[str, np.ndarray]]:
    """The xy, xz and yz planes through the sphere's center, each SIDE pixels square."""
    x, y, z = CENTER_UM
    return [
        (f"xy · z = {z:.0f} µm", plane(scan, CENTER_UM, (SIDE, SIDE, 1))),
        (f"xz · y = {y:.0f} µm", plane(scan, CENTER_UM, (SIDE, 1, SIDE))),
        (f"yz · x = {x:.0f} µm", plane(scan, CENTER_UM, (1, SIDE, SIDE))),
    ]


def figure(cuts: list[tuple[str, np.ndarray]], output: Path) -> None:
    """The three planes with 200 um scale bars, sharing one gray range."""
    low, high = np.percentile(np.concatenate([cut.ravel() for _, cut in cuts]), (1, 99.5))
    style()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.9), layout="constrained")
    stroke = [effects.withStroke(linewidth=3, foreground="black")]
    for ax, (title, cut) in zip(axes, cuts, strict=True):
        ax.imshow(cut, origin="lower", vmin=low, vmax=high)
        bar = 200 / PITCH_UM
        ax.plot([20, 20 + bar], [24, 24], color="white", lw=2, path_effects=stroke)
        ax.text(20 + bar / 2, 34, "200 µm", color="white", ha="center", path_effects=stroke)
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle("Ruby sphere · K11 laminography · scan 54013 · 1441 views · 2.16 µm voxels")
    save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="the raw k11-54013.nxs acquisition")
    parser.add_argument("output", type=Path)
    parser.add_argument("--aligned", type=Path, help="reuse a scan saved from tj.align")
    args = parser.parse_args()
    scan = tj.load(args.aligned) if args.aligned else tj.align(load(args.input, -35.0)).scan
    figure(planes(scan), args.output)


if __name__ == "__main__":
    main()
