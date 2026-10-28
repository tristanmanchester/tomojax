"""A dark card for the README figures: pixel-exact image panels, labels and scale bars.

Positions and sizes are in output pixels from the top left. Images are
resampled to their panel size here (area averaging to shrink, nearest
neighbours to enlarge by whole factors) and drawn one pixel to one pixel, so
matplotlib never interpolates them; one grey scale is shared by every panel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

BACKGROUND = "#101318"
TEXT = "#e8eaed"
MUTED = "#9aa3ad"
ACCENT = "#f2b84b"
FONTS = ["Inter", "Adwaita Sans", "Helvetica Neue", "Arial", "DejaVu Sans"]


def resample(image: np.ndarray, size: int, *, smooth: bool = False) -> np.ndarray:
    """A square ``image`` at ``size`` pixels.

    Area averages shrink it; whole-pixel copies enlarge it, or linear
    interpolation if ``smooth``.
    """
    n = image.shape[0]
    if smooth and size > n:
        at = np.clip((np.arange(size) + 0.5) * n / size - 0.5, 0, n - 1)
        low = np.minimum(np.floor(at).astype(int), n - 2)
        weights = np.zeros((size, n))
        weights[np.arange(size), low] = 1 - (at - low)
        weights[np.arange(size), low + 1] = at - low
        return weights @ image @ weights.T
    if size % n == 0:
        k = size // n
        return np.repeat(np.repeat(image, k, axis=0), k, axis=1)
    edges = np.linspace(0, n, size + 1)
    weights = np.zeros((size, n))
    for i in range(size):
        a, b = edges[i], edges[i + 1]
        cells = np.arange(int(np.floor(a)), int(np.ceil(b)))
        weights[i, cells] = np.minimum(cells + 1, b) - np.maximum(cells, a)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ image @ weights.T


class Card:
    """A ``width`` x ``height`` pixel figure on the dark background."""

    def __init__(self, width: int, height: int) -> None:
        mpl.rcParams.update(
            {"font.family": "sans-serif", "font.sans-serif": FONTS, "text.antialiased": True}
        )
        self.width, self.height, self.dpi = width, height, 100
        self.fig = plt.figure(
            figsize=(width / self.dpi, height / self.dpi), dpi=self.dpi, facecolor=BACKGROUND
        )

    def _box(self, x: float, y: float, w: float, h: float) -> list[float]:
        return [x / self.width, 1 - (y + h) / self.height, w / self.width, h / self.height]

    def image(
        self,
        x: int,
        y: int,
        size: int,
        data: np.ndarray,
        vmin: float,
        vmax: float,
        *,
        smooth: bool = False,
    ) -> Any:
        """A slice (first index x, second y; y drawn upwards) as a ``size`` pixel square."""
        ax = self.fig.add_axes(self._box(x, y, size, size))
        picture = resample(np.asarray(data, np.float64).T[::-1], size, smooth=smooth)
        ax.imshow(
            picture,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="none",
            extent=(0, size, size, 0),
        )
        ax.set_axis_off()
        return ax

    def picture(self, x: int, y: int, rgb: np.ndarray) -> None:
        """An RGB image drawn one pixel to one pixel with its top left at ``(x, y)``."""
        h, w = rgb.shape[:2]
        ax = self.fig.add_axes(self._box(x, y, w, h))
        ax.imshow(rgb, interpolation="none", extent=(0, w, h, 0))
        ax.set_axis_off()

    def text(
        self,
        x: float,
        y: float,
        s: str,
        size: float,
        color: str = TEXT,
        weight: str = "normal",
        ha: str = "left",
        va: str = "baseline",
    ) -> None:
        self.fig.text(
            x / self.width,
            1 - y / self.height,
            s,
            fontsize=size * 72 / self.dpi,
            color=color,
            weight=weight,
            ha=ha,
            va=va,
        )

    def rectangle(
        self, x: float, y: float, w: float, h: float, color: str = ACCENT, width: float = 1.5
    ) -> None:
        self.fig.add_artist(
            plt.Rectangle(
                (x / self.width, 1 - (y + h) / self.height),
                w / self.width,
                h / self.height,
                fill=False,
                edgecolor=color,
                linewidth=width * 72 / self.dpi,
                transform=self.fig.transFigure,
            )
        )

    def line(self, xs: list[float], ys: list[float], color: str = TEXT, width: float = 2.0) -> None:
        self.fig.add_artist(
            plt.Line2D(
                [x / self.width for x in xs],
                [1 - y / self.height for y in ys],
                color=color,
                linewidth=width * 72 / self.dpi,
                transform=self.fig.transFigure,
                solid_capstyle="butt",
            )
        )

    def scale_bar(self, x: float, y: float, pixels: float, label: str) -> None:
        """A bar ``pixels`` long, its left end at ``(x, y)``, labelled above."""
        self.line([x, x + pixels], [y, y], TEXT, 3.0)
        self.text(x + pixels / 2, y - 9, label, 19, TEXT, ha="center")

    def axes(self, x: int, y: int, w: int, h: int) -> Any:
        """Plain axes for a plot, styled for the card."""
        ax = self.fig.add_axes(self._box(x, y, w, h), facecolor=BACKGROUND)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#4a525c")
        ax.tick_params(colors=MUTED, labelsize=15 * 72 / self.dpi, length=4, width=0.8)
        return ax

    def save(self, path: Path, colours: int | None = 256) -> None:
        """Write a PNG of ``colours`` palette entries, or full colour for ``None``."""
        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=self.dpi, facecolor=BACKGROUND)
        plt.close(self.fig)
        with Image.open(path) as image:
            picture = image.convert("RGB")
        if colours is not None:
            picture = picture.quantize(colours, dither=Image.Dither.NONE)
        picture.save(path, optimize=True)
