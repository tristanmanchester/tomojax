"""Repeated view directions, grouped for analytic reconstruction weights."""

from __future__ import annotations

import numpy as np

_TURN = 2 * np.pi
# Views within a millionth of a turn are one direction measured again.
_REPEAT = _TURN * 1e-6


def group_repeats(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distinct directions in ``[0, 2 pi)``, each view's group, and group sizes.

    Angles in radians within a millionth of a turn of a neighbour share a
    group, whichever whole turn they are labelled with and whatever their
    rounding, so a direction measured twice keeps one angular cell.
    """
    canonical = np.remainder(np.asarray(angles, np.float64), _TURN)
    order = np.argsort(canonical)
    gaps = np.diff(np.append(canonical[order], canonical[order[0]] + _TURN))
    # Start after the largest gap so that no group straddles the cut.
    order = np.roll(order, -(int(np.argmax(gaps)) + 1))
    unwrapped = canonical[order[0]] + np.append(
        0.0, np.cumsum(np.remainder(np.diff(canonical[order]), _TURN))
    )
    sorted_groups = np.cumsum(np.append(0, np.diff(unwrapped) > _REPEAT))
    repeats = np.bincount(sorted_groups)
    representatives = np.bincount(sorted_groups, weights=unwrapped) / repeats
    groups = np.empty_like(sorted_groups)
    groups[order] = sorted_groups
    return np.remainder(representatives, _TURN), groups, repeats


def group_projection_planes(rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each unit ray's projection-plane group and its exposure count.

    Used when too few directions are present to fit an axis. Opposing rays share
    a plane; doubled-angle half-turn grouping uses half the repeat tolerance.
    """
    rays = np.asarray(rays, np.float64)
    remaining = np.ones(len(rays), bool)
    inverse = np.empty(len(rays), np.intp)
    repeats = []
    while remaining.any():
        first = np.flatnonzero(remaining)[0]
        ray = rays[first]
        distance = np.minimum(
            np.linalg.norm(rays - ray, axis=1), np.linalg.norm(rays + ray, axis=1)
        )
        distance[first] = 0.0
        group = np.flatnonzero(remaining & (distance <= _REPEAT / 2))
        inverse[group] = len(repeats)
        repeats.append(len(group))
        remaining[group] = False
    return inverse, np.asarray(repeats, np.intp)
