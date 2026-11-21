"""Full-view SIRT times of TomoJAX, ASTRA and TIGRE from docs/measurements.md.

    uv run python examples/gallery/sirt_times.py sirt-times.png

Draws the warm solve times and common-projector residuals recorded in the
measurement guide's "Full-view walnut SIRT" and "Full-view Nikon and Zeiss
SIRT" tables (measured 2026-10-19). TomoJAX runs 12 accelerated iterations,
ASTRA and TIGRE 24, each from its own FDK. Nothing is measured here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import save, style
import matplotlib.pyplot as plt

LIBRARIES = ("TomoJAX", "ASTRA", "TIGRE")
COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
# Scan: ((warm seconds, common-projector relative residual) per library).
TABLE = {
    "Walnut, one orbit\n1200 views, 501³":
        ((53.47, 0.024742), (67.27, 0.025998), (363.06, 0.044170)),
    "Walnut, three orbits\n3600 views, 501³":
        ((174.22, 0.026222), (234.35, 0.027558), (1120.54, 0.045232)),
    "Nikon seeds\n1571 views, 500³": ((89.63, 0.028788), (125.78, 0.030057), (493.43, 0.056112)),
    "Zeiss CuO_2\n801 views, 512³": ((36.15, 0.187915), (62.76, 0.189497), (290.02, 0.207728)),
}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    style()
    fig, axes = plt.subplots(1, len(TABLE), figsize=(11, 2.6), sharey=True)
    for ax, (scan, rows) in zip(axes, TABLE.items(), strict=True):
        positions = range(len(LIBRARIES))
        seconds = [row[0] for row in rows]
        ax.barh(positions, seconds, color=COLORS, height=0.6)
        for y, (time, residual) in zip(positions, rows, strict=True):
            ax.text(time, y, f"  {time:.0f} s · {residual:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, max(seconds) * 1.75)
        ax.set_title(scan)
        ax.set_xlabel("warm time (s)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", color="0.9")
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(LIBRARIES)), LIBRARIES)
    axes[0].invert_yaxis()  # shared: TomoJAX on top in every panel
    fig.suptitle(
        "Full-view SIRT on an RTX 4070 Laptop GPU: warm time · relative residual "
        "(TomoJAX 12 accelerated iterations, ASTRA and TIGRE 24)",
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, args.output)


if __name__ == "__main__":
    main()
