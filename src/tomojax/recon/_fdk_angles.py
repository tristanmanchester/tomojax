"""Circular view ordering shared by FDK weights and virtual detector columns."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Geometry

_TURN = 2 * np.pi


def grouped_angles(geometry: Geometry, n_views: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distinct angles modulo a turn, each view's group, and group sizes.

    Angles within a millionth of a turn share their angular measure, including
    views on opposite sides of the 0/360-degree boundary.
    """
    thetas = getattr(geometry, "angles", None)
    if thetas is None:
        raise ValueError("FDK needs a geometry with view angles")
    degrees = np.asarray(thetas, dtype=np.float64)
    if degrees.ndim != 1 or len(degrees) < n_views:
        raise ValueError("FDK needs one finite rotation angle per view")
    degrees = degrees[:n_views]
    if not np.isfinite(degrees).all():
        raise ValueError("FDK needs one finite rotation angle per view")
    canonical = np.remainder(degrees, 360.0)
    key = np.round(canonical / 360.0 * 1e6).astype(np.int64) % 1_000_000
    _, first, group, repeats = np.unique(
        key, return_index=True, return_inverse=True, return_counts=True
    )
    return np.deg2rad(canonical[first]), group, repeats


def ordered_arc(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Order distinct angles from the largest circular gap and unwrap the arc.

    A closing gap no larger than 1.5 typical view steps counts as a full turn,
    matching FDK's full-turn tolerance without depending on angle labels.
    """
    if len(angles) < 2:
        raise ValueError("FDK needs at least two distinct rotation angles")
    order = np.argsort(angles)
    sorted_angles = angles[order]
    circular_gaps = np.diff(np.append(sorted_angles, sorted_angles[0] + _TURN))
    largest = int(np.argmax(circular_gaps))
    order = np.roll(order, -(largest + 1))
    gaps = np.remainder(np.diff(angles[order]), _TURN)
    sorted_angles = angles[order[0]] + np.append(0.0, np.cumsum(gaps))
    full_turn = circular_gaps[largest] <= 1.5 * float(np.median(gaps))
    return order, sorted_angles, bool(full_turn)
