"""One figure per scored case: every image on the reference's window, and radial profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


def save(
    path: Path,
    reference: np.ndarray,
    images: dict[str, np.ndarray],
    centre: tuple[float, float],
    pitch: float,
    title: str,
    window: tuple[float, float] | None = None,
) -> None:
    """Write ``reference`` and ``images`` (``[y, x]``, same units) and their radial profiles.

    ``centre`` is in physical x, y pixel indices, ``pitch`` the pixel size in mm and ``window``
    the grey range (by default the reference's 0.5th to 99.5th percentile).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = {"reference": reference, **images}
    low, high = np.percentile(reference, [0.5, 99.5]) if window is None else window
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(3.0 * (len(panels) + 1), 3.3))
    for ax, (name, image) in zip(axes, panels.items(), strict=False):
        ax.imshow(image, origin="lower", cmap="gray", vmin=low, vmax=high)
        ax.set_title(name, fontsize=9)
        ax.set_axis_off()
    y, x = np.meshgrid(*(np.arange(n) for n in reference.shape), indexing="ij")
    ring = np.floor(np.hypot(x - centre[0], y - centre[1])).astype(int)
    count = np.maximum(np.bincount(ring.ravel()), 1)
    radius = np.arange(count.size) * pitch
    for name, image in panels.items():
        profile = np.bincount(ring.ravel(), image.ravel(), count.size) / count
        axes[-1].plot(radius, profile, lw=1.6 if name == "reference" else 1.0, label=name)
    axes[-1].set_xlabel("radius (mm)", fontsize=9)
    axes[-1].set_ylim(low, high + 0.2 * (high - low))
    axes[-1].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def curves(
    path: Path, lines: dict[str, tuple[np.ndarray, np.ndarray]], limit: float, title: str
) -> None:
    """Write binned deviations from a straight line against measured line integral.

    ``lines`` maps a label to ``(measured, deviation)``; the band is ``±limit``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    ax.axhspan(-limit, limit, color="0.9", label="linear")
    for name, (measured, deviation) in lines.items():
        ax.plot(measured, deviation, lw=1.0, label=name)
    ax.set_xlabel("measured line integral", fontsize=9)
    ax.set_ylabel("deviation from a line", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
