"""The same parallel-beam slice on a wide centered panel and a half-width offset panel."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import panel, relative_error, save, style
from battery_offset_detector import IN_VIEW
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import matplotlib.pyplot as plt
import numpy as np

import tomojax as tj


def draw_acquisition(
    ax: plt.Axes, share: plt.Axes, width: int, center: float, radius: float
) -> None:
    """Parallel illumination and the fraction of the orbit that sees each radius."""
    lo, hi = (center - width / 2) / radius, (center + width / 2) / radius
    ax.add_patch(Rectangle((-1.65, lo), 3.03, hi - lo, color="#e5f3ef", zorder=0))
    for y in np.linspace(lo, hi, 7):
        ax.annotate(
            "",
            (-0.90, y),
            (-1.60, y),
            arrowprops={"arrowstyle": "->", "color": "#519b94", "lw": 0.8},
        )
    ax.add_patch(Circle((0, 0), 1, fc="#d1d3ce", ec="#364e4c", lw=1))
    ax.add_patch(Circle((-0.23, 0.18), 0.27, fc="#a3ada7", ec="none"))
    ax.add_patch(Circle((0.34, -0.29), 0.17, fc="#a3ada7", ec="none"))
    ax.plot([1.38, 1.38], [lo, hi], color="#253e3a", lw=4, solid_capstyle="butt")
    ax.plot(0, 0, "+", color="#b33345", ms=7, mew=1.5)
    ax.plot([0, 1.41], [0, 0], ":", color="#b33345", lw=0.8)
    ax.add_patch(
        FancyArrowPatch(
            (-0.65, 0.94),
            (0.65, 0.94),
            connectionstyle="arc3,rad=-.5",
            arrowstyle="-|>",
            mutation_scale=9,
            color="#364e4c",
            lw=1,
        )
    )
    ax.text(-1.62, -1.28, "beam", fontsize=8, color="#366d67")
    ax.text(0.96, -1.28, "panel", fontsize=8)
    if center:
        ax.annotate(
            f"{width / 2 - abs(center):g} px",
            xy=(1.38, lo),
            xytext=(0.61, -0.55),
            ha="left",
            color="#a43544",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#a43544", "lw": 0.8},
        )
    ax.set(xlim=(-1.7, 1.72), ylim=(-1.35, 1.42), aspect="equal")
    ax.set_axis_off()
    coords = np.linspace(-1.01, 1.01, 301)
    r = np.hypot(coords[:, None], coords)
    fraction = (
        np.arccos(np.clip(lo / np.maximum(r, 1e-12), -1, 1))
        - np.arccos(np.clip(hi / np.maximum(r, 1e-12), -1, 1))
    ) / np.pi
    colors = IN_VIEW(Normalize(0.5, 1)(fraction))
    colors[..., 3] = r <= 1
    share.imshow(colors, extent=(-1, 1, -1, 1), origin="lower")
    share.plot(0, 0, "+", color="#ad3044", ms=7, mew=1.5)
    share.text(
        0,
        -0.48,
        f"{100 * (np.arccos(np.clip(lo, -1, 1)) - np.arccos(np.clip(hi, -1, 1))) / np.pi:.0f}% rim",
        ha="center",
        fontsize=8,
        color="#172f2c",
        bbox={
            "boxstyle": "round,pad=.14",
            "edgecolor": "none",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )
    share.set_axis_off()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    n, width, overlap = 480, 256, 32
    coords = np.arange(n) - (n - 1) / 2
    truth = np.zeros((n, n))
    rng = np.random.default_rng(27)
    disks = [(0.0, 0.0, 0.46 * n, 1.0)]
    for _ in range(85):
        x, y = rng.uniform(-0.42, 0.42, 2) * n
        r = rng.uniform(0.009, 0.038) * n
        if np.hypot(x, y) + r < 0.46 * n:
            disks.append((x, y, r, -0.37))
    for x, y, r, value in disks:
        truth += value * ((coords - x) ** 2 + (coords[:, None] - y) ** 2 < r * r)
    style()
    fig = plt.figure(figsize=(7.5, 4.2), layout="constrained")
    grid = fig.add_gridspec(2, 6, height_ratios=(0.76, 1.35))
    panel(fig.add_subplot(grid[1, :2]), truth, "Phantom", 0, 1.1)
    for i, (columns, center) in enumerate(((2 * width, 0.0), (width, (width - overlap) / 2))):
        count = 480 * (1 + i)
        angles = np.arange(count) * 180 / 480
        u = np.arange(columns) - (columns - 1) / 2 + center
        angle = np.deg2rad(angles)[:, None]
        data = np.zeros((count, columns))
        for x, y, r, v in disks:
            d = u - x * np.cos(angle) + y * np.sin(angle)
            data += 2 * v * np.sqrt(np.maximum(0, r * r - d * d))
        g = tj.ParallelGeometry(
            tj.Grid(n, n, 1, 1, 1, 1), tj.Detector(columns, 1, 1, 1, center=(center, 0)), angles
        )
        volume = np.asarray(
            tj.reconstruct(tj.Scan(data[:, None].astype("f4"), g), "fourier").volume
        )[0]
        diagram = fig.add_subplot(grid[0, 3 * i : 3 * i + 2])
        diagram.set_title(("Centered half turn", "Offset full turn")[i], loc="left")
        draw_acquisition(diagram, fig.add_subplot(grid[0, 3 * i + 2]), columns, center, n / 2)
        error = relative_error(volume, truth, np.hypot(coords, coords[:, None]) < 0.49 * n)
        panel(
            fig.add_subplot(grid[1, 2 + 2 * i : 4 + 2 * i]),
            volume,
            f"{columns} columns · error {100 * error:.1f}%",
            0,
            1.1,
        )
    save(fig, args.output, palette=False)


if __name__ == "__main__":
    main()
