"""Circular view ordering shared by FBP, FDK and virtual detector columns."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tomojax.core.geometry.cone import cone_parts
from tomojax.core.geometry.views import host_view_poses

from ._directions import group_repeats

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
    # The angle of the source around object z defines the quadrature. Inverting
    # world-from-object gives R.T @ (source - translation) in the object.
    parts = cone_parts(geometry)
    if parts is None:
        raise ValueError("FDK needs a cone-beam trajectory")
    poses = host_view_poses(geometry, n_views, dtype=np.float64)
    if poses is None:
        poses = np.asarray([geometry.pose_matrix(i) for i in range(n_views)], np.float64)
    azimuth = np.empty(n_views)
    start = 0
    for count, beam, _ in parts:
        stop = n_views if count is None else min(start + count, n_views)
        pose = poses[start:stop]
        source = np.einsum("nij,ni->nj", pose[:, :3, :3], beam.source() - pose[:, :3, 3])
        azimuth[start:stop] = np.arctan2(-source[:, 0], -source[:, 1])
        start = stop
    return group_repeats(azimuth)


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
