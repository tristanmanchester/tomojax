"""Shared pieces of the gallery figures: phantoms, measurement noise and plots.

The phantoms are the procedural generators in ``bench/phantoms``. Each figure
projects a phantom voxelised finer than the reconstruction grid, so the data
are not made by the model that inverts them.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".artifacts" / "figures" / "phantoms"


def phantom(name: str, n: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the ``name`` phantom at ``n`` voxels per side, building it once."""
    path = CACHE / f"{name}-{n}.npz"
    if path.exists():
        with np.load(path) as data:
            return data["volume"], json.loads(str(data["meta"]))
    sys.path.insert(0, str(ROOT / "bench" / "phantoms"))
    volume, meta = getattr(importlib.import_module(name), name)(n)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, volume=volume, meta=json.dumps(meta))
    return volume, meta


def counts(
    line_integrals: np.ndarray, photons: float, rng: np.random.Generator, gain: Any = 1.0
) -> np.ndarray:
    """Poisson counts of a beam of ``photons`` per pixel through ``line_integrals``."""
    mean = photons * gain * np.exp(-np.asarray(line_integrals, np.float64))
    return rng.poisson(mean).astype(np.float32)


def relative_error(volume: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """Relative L2 error of ``volume`` against ``truth`` inside ``mask``."""
    difference = np.asarray(volume, np.float64)[mask] - truth[mask]
    return float(np.linalg.norm(difference) / np.linalg.norm(truth[mask]))


def downsample(volume: np.ndarray, n: int) -> np.ndarray:
    """Average ``volume`` onto ``n`` voxels per side (the reconstruction grid)."""
    from scipy.ndimage import zoom

    return zoom(np.asarray(volume, np.float32), n / volume.shape[0], order=1, grid_mode=True)


def style() -> None:
    """Matplotlib settings shared by the figures."""
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.titlepad": 4,
            "image.cmap": "gray",
            "image.interpolation": "nearest",
        }
    )


def panel(ax: Any, image: np.ndarray, title: str, vmin: float, vmax: float) -> None:
    """Show a slice ``image`` (first index along x) without axes."""
    ax.imshow(np.asarray(image).T, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_axis_off()


def save(fig: Any, path: Path) -> None:
    """Write ``fig`` to ``path`` as a 256-colour PNG and close it."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    with Image.open(path) as image:
        quantised = image.convert("RGB").quantize(256)
    quantised.save(path, optimize=True)
    print(f"wrote {path}")
