"""Score a reconstruction on an agreed feature, without ground truth.

Every score compares a volume with itself inside a feature mask and a surrounding
mask, both boolean arrays of the volume's shape (for example the .mask of TomoJAX
feature handles made with the seeded and band recipes). Higher contrast-to-noise,
lower noise and a narrower edge are better; compare candidates on the same masks.

Run it on saved arrays::

    python feature_scores.py volume.npy feature.npy surround.npy

or call ``scores(volume, feature, surround)`` from Python, for example in the TomoJAX
MCP python tool with ``runpy.run_path(path)["scores"]``.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy import ndimage


def contrast_to_noise(volume: np.ndarray, feature: np.ndarray, surround: np.ndarray) -> float:
    """Difference of the means inside and around the feature over their pooled std."""
    inside, around = volume[feature], volume[surround]
    pooled = np.sqrt((inside.var() + around.var()) / 2)
    return float(abs(inside.mean() - around.mean()) / pooled)


def noise(volume: np.ndarray, feature: np.ndarray) -> float:
    """Standard deviation inside the feature, which should be uniform."""
    return float(volume[feature].std())


def edge_width(volume: np.ndarray, feature: np.ndarray, reach: float = 6.0) -> float:
    """The 10-90% rise distance across the feature boundary, in voxels.

    Values within ``reach`` voxels of the boundary are averaged in half-voxel bins
    of signed distance (negative inside), giving the edge profile. A step edge
    measures under one voxel and an edge blurred by a Gaussian of sigma about 2.4
    sigma (the ideal 10-90% rise is 2.56 sigma; binning a curved boundary reads low).
    The feature must be thicker than the blur: across a thin shell both of its edges
    fall within ``reach`` and the width measures the shell's own profile.
    """
    inside = np.asarray(ndimage.distance_transform_edt(feature))
    outside = np.asarray(ndimage.distance_transform_edt(~feature))
    distance = np.where(feature, 0.5 - inside, outside - 0.5)
    near = np.abs(distance) <= reach
    edges = np.arange(-reach, reach + 0.5, 0.5)
    which = np.digitize(distance[near], edges)
    values, distances = volume[near], distance[near]
    centers, profile = [], []
    for b in range(1, len(edges)):
        selected = which == b
        if selected.any():
            centers.append(distances[selected].mean())
            profile.append(values[selected].mean())
    x, y = np.asarray(centers), np.asarray(profile)
    hi, lo = y[0], y[-1]
    level = (y - lo) / (hi - lo)  # 1 inside, 0 outside

    def crossing(fraction: float) -> float:
        # The first position, from inside outwards, where the profile falls below it.
        k = int(np.argmax(level < fraction))
        return float(np.interp(fraction, level[[k, k - 1]], x[[k, k - 1]]))

    return abs(crossing(0.1) - crossing(0.9))


def scores(
    volume: np.ndarray, feature: np.ndarray, surround: np.ndarray, margin: int = 2
) -> dict[str, float]:
    """All three scores for one candidate.

    Contrast and noise use the feature's interior, ``margin`` voxels in from its
    boundary, so blurred edges do not count as noise. Use ``margin=0`` for a feature
    too thin to have an interior.
    """
    volume = np.asarray(volume, dtype=np.float64)
    feature, surround = np.asarray(feature, dtype=bool), np.asarray(surround, dtype=bool)
    interior = ndimage.binary_erosion(feature, iterations=margin) if margin else feature
    if not interior.any():
        raise ValueError(f"the feature has no voxels {margin} in from its edge; lower margin")
    return {
        "contrast_to_noise": contrast_to_noise(volume, interior, surround),
        "noise": noise(volume, interior),
        "edge_width": edge_width(volume, feature),
    }


if __name__ == "__main__":
    volume, feature, surround = (np.load(path) for path in sys.argv[1:4])
    sys.stdout.write(json.dumps(scores(volume, feature, surround)) + "\n")
