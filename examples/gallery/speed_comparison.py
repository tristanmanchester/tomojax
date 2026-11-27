"""Full-volume reconstruction times for Diamond K11 scan 52544, measured 2026-11-26.

    uv run python examples/gallery/speed_comparison.py docs/images/speed-comparison.png

Each cell summarizes three worker launches in the mode with the lowest median
for that method and configuration. The clock includes the final output flush
and excludes input warm-up and one-time conversion. TomoPy used 40 CPU cores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import save, style
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

PANELS = (
    "1 \u00d7 Tesla V100S",
    "4 \u00d7 Tesla V100S",
    "1 \u00d7 RTX PRO 6000\nBlackwell",
    "8 \u00d7 RTX PRO 6000\nBlackwell",
)
BLUE, GRAY = "#2a78d6", "#52514e"

# (Minimum, median, maximum) seconds in panel order. Both TomoJAX methods
# used streaming in all four panels. The CPU has one result per node type.
FOURIER = {
    "TomoJAX Fourier": (
        (67.52, 74.51, 77.81), (39.86, 41.28, 47.04),
        (26.23, 27.87, 39.40), (35.69, 35.91, 39.02),
    ),
    "tomocupy fourierrec": (
        (108.06, 109.79, 118.28), (65.27, 66.35, 67.35),
        (69.61, 69.71, 69.75), (44.99, 45.82, 46.39),
    ),
    "httomo LPRec": (
        (721.59, 722.48, 724.16), (436.13, 461.66, 504.96),
        (197.59, 198.54, 199.80), (153.78, 153.87, 156.66),
    ),
    "TomoPy gridrec (CPU)": (
        (372.83, 430.41, 464.66), None,
        (143.47, 143.87, 144.93), None,
    ),
}  # fmt: skip
FBP = {
    "TomoJAX FBP": (
        (365.13, 367.44, 368.74), (101.17, 101.21, 101.66),
        (59.27, 59.61, 59.70), (27.48, 28.99, 32.63),
    ),
    "tomocupy linerec": (
        (713.82, 714.11, 715.82), (187.66, 188.66, 188.79),
        (186.84, 187.14, 187.30), (45.14, 45.77, 47.13),
    ),
    "TIGRE FBP": (
        (538.57, 558.34, 635.40), (167.63, 172.34, 175.79),
        (178.22, 180.50, 180.85), (47.67, 48.91, 51.97),
    ),
    "Nabu FBP": (
        (559.72, 591.79, 602.50), (175.91, 184.85, 190.04),
        (179.26, 186.84, 191.20), (69.17, 71.49, 72.16),
    ),
    "ASTRA FBP": (
        (747.91, 768.64, 788.81), (199.37, 199.59, 200.22),
        (179.38, 182.04, 186.80), (44.90, 46.21, 46.22),
    ),
    "LEAP FBP": (
        (719.15, 753.28, 791.64), (197.11, 198.18, 209.99),
        (220.85, 227.70, 262.19), (49.77, 51.28, 51.38),
    ),
    "httomo FBP": (
        (1168.36, 1170.10, 1170.70), (578.96, 586.80, 648.50),
        (322.24, 322.52, 324.90), (132.42, 132.49, 133.76),
    ),
    "CIL FBP": (
        (1235.79, 1240.20, 1253.64), (312.83, 320.18, 341.68),
        (324.12, 360.43, 360.72), (79.57, 80.02, 80.06),
    ),
}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    style()
    fig, axes = plt.subplots(2, 2, figsize=(5.3, 8.35), sharey=True)
    fig.subplots_adjust(left=0.32, right=0.997, bottom=0.055, top=0.83, wspace=0.14, hspace=0.29)
    fig.text(0.006, 0.98, "Diamond K11 scan 52544", va="top", size=14, weight="bold")
    fig.text(0.006, 0.943, "Full volume · 2160 \u00d7 2560 \u00d7 2560 voxels", va="top", size=11)
    fig.text(0.006, 0.913, "Seconds from launch to last output written", va="top", size=11)
    fig.text(
        0.006,
        0.882,
        "Fastest mode · dots are medians, whiskers span three runs",
        va="top",
        color=GRAY,
    )

    for index, (title, ax) in enumerate(zip(PANELS, axes.flat, strict=True)):
        ax.set_title(title, weight="bold", fontsize=10)
        ax.set_xscale("log")
        ax.set_xlim(20, 1750)
        ax.set_xticks([30, 100, 300, 1000], ["30", "100", "300", "1000"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.tick_params(axis="both", length=0, pad=3)
        ax.grid(axis="x", color="0.9", lw=0.7)
        ax.set_axisbelow(True)
        ax.set_ylim(len(FOURIER) + len(FBP) + 0.7, -1.5)
        ax.set_yticks([])
        y = -1
        for family, methods in (("Fourier-type", FOURIER), ("Filtered backprojection", FBP)):
            if index % 2 == 0:
                ax.text(
                    -0.03,
                    y,
                    family,
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=9,
                    weight="bold",
                )
            ax.axhline(y, color="0.77", lw=0.75)
            for method, measurements in methods.items():
                y += 1
                color = BLUE if method.startswith("TomoJAX") else GRAY
                if index % 2 == 0:
                    ax.text(
                        -0.03,
                        y,
                        method,
                        transform=ax.get_yaxis_transform(),
                        ha="right",
                        va="center",
                        fontsize=10,
                        color=color,
                    )
                cell = measurements[index]
                if cell is None:
                    continue
                low, median, high = cell
                ax.plot([low, high], [y, y], color=color, lw=1.4, marker="|", ms=5)
                ax.plot(median, y, marker="o", ms=3.4, color=color)
                # Keep labels inside narrow panels at both ends of the time axis.
                left = median > 300
                ax.annotate(
                    f"{median:#.4g}".rstrip("."),  # four significant figures
                    (low if left else high, y),
                    xytext=(-4 if left else 4, 0),
                    textcoords="offset points",
                    ha="right" if left else "left",
                    va="center",
                    color=color,
                    fontsize=10,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.1},
                )
            y += 1.3
    fig.supxlabel("Elapsed time (s, log scale)", x=0.65, y=0.002, fontsize=10)
    save(fig, args.output, palette=False)


if __name__ == "__main__":
    main()
