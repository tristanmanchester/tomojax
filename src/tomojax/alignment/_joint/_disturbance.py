"""Cross-detector agreement of unexpected column signals before a pose fit."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np

from ._columns import transient_profile
from ._noise import Windows

if TYPE_CHECKING:
    from ._pose import ViewOrder


class DisturbedViews(TypedDict):
    """Scores in acquisition order, the flagged half-open intervals and their measured angles."""

    score: list[float | None]
    ranges: list[list[int]]
    angles_deg: list[list[float]]
    message: str


def disturbance_score(
    data: np.ndarray, weight: np.ndarray | None, sigma: float, order: ViewOrder
) -> np.ndarray:
    """Fraction of power shared by opposite detector bands, accounting for noise and disagreement."""
    n, h, w = data.shape
    b = max(1, round(max(h, w) / 80))
    rows, cols = h // b, w // b
    score = np.zeros(n)
    if min(rows, cols) < 2:
        return score
    if b > 1:
        shape = (n, rows, b, cols, b)
        cut = data[:, : rows * b, : cols * b].reshape(shape)
        if weight is None:
            data = np.asarray(np.mean(cut, axis=(2, 4), dtype=np.float64), dtype=np.float32)
        else:
            weights = weight[:, : rows * b, : cols * b].reshape(shape)
            weight = np.asarray(np.mean(weights, axis=(2, 4), dtype=np.float32))
            total = np.mean(cut * weights, axis=(2, 4), dtype=np.float64)
            data = np.divide(total, weight, where=weight > 0, out=np.zeros_like(weight))
    band = max(1, rows // 4)
    for start, stop in order.segments:
        if stop - start < 2:
            continue
        block = data[start:stop]
        precision = None if weight is None else weight[start:stop]
        windows = Windows(block.shape)
        windows.gather(block)
        measured = windows.sd(precision)
        sd = sigma / b if measured is None else measured
        ends = []
        masks = []
        for edge in (slice(0, band), slice(-band, None)):
            observed = None if precision is None else precision[:, edge]
            residual, valid, _, _ = transient_profile(block[:, edge], observed, sd)
            ends.append(residual)
            masks.append(valid)
        paired = masks[0] & masks[1]
        shared = np.sum(ends[0] * ends[1], axis=1, where=paired)
        # If both bands measure the same field with independent errors, their
        # cross-product estimates its power; half the squared difference estimates
        # one band's error power. Compare against both that and the pixel noise.
        error = np.sum((ends[0] - ends[1]) ** 2, axis=1, where=paired) / 2
        variance = np.maximum(error, paired.sum(axis=1) * sd**2)
        # Perfect agreement with noiseless measurements has unit common-signal
        # fraction. Express the power as a bounded fraction, including that limit.
        power = np.maximum(shared, 0)
        score[start:stop] = np.divide(
            power, power + variance, where=power + variance > 0, out=np.zeros_like(power)
        )
    return score


def disturbed_report(
    scores: np.ndarray, angles: np.ndarray, segments: np.ndarray
) -> DisturbedViews:
    """Half-open input ranges with common power exceeding independent error power."""
    selected = scores > 0.5
    starts = np.flatnonzero(selected & np.r_[True, ~selected[:-1] | (np.diff(segments) != 0)])
    ends = np.flatnonzero(selected & np.r_[~selected[1:] | (np.diff(segments) != 0), True])
    ranges = np.column_stack((starts, ends + 1)).tolist()
    degrees = np.column_stack((angles[starts], angles[ends])).tolist()
    if ranges:
        spans = ", ".join(f"{a:.2f}° to {b:.2f}°" for a, b in degrees)
        message = (
            f"Column disturbances in {int(selected.sum())} views ({spans}). "
            "Inspect those projections and pass confirmed ranges to AlignConfig(exclude_views=...)."
        )
    else:
        message = "No transient column disturbance detected."
    return {
        "score": [float(s) if np.isfinite(s) else None for s in scores],
        "ranges": ranges,
        "angles_deg": degrees,
        "message": message,
    }
