"""Full-turn fan redundancy from the acquisition's conjugate detector coverage.

The circular, equatorial conjugate of (beta, gamma) is
(beta + pi + 2*gamma, -gamma). A changing panel must share a ray's measure
with that view's panel, not with a mirror of its own. Repeated directions
share both sides' measure. Noncircular FDK remains an approximate inverse.
"""

from __future__ import annotations

import numpy as np


def _reliability(ray: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Smooth edge reliability odds, recovering Wang's sin-squared partition."""
    lo, hi, pixel = parameters[..., 0], parameters[..., 1], parameters[..., 2]
    width = np.maximum(np.minimum(-lo, hi), pixel)
    distance = np.minimum(ray - lo, hi - ray)
    return np.tan(np.pi / 4 * np.clip(distance / width, 0, 1)) ** 2


def _members(groups: np.ndarray, repeats: np.ndarray) -> np.ndarray:
    """View indices at each distinct direction; -1 pads unequal repeat counts."""
    order = np.argsort(groups, kind="stable")
    starts = np.r_[0, np.cumsum(repeats)[:-1]]
    members = np.full((len(repeats), int(repeats.max())), -1, dtype=int)
    for repeat in range(members.shape[1]):
        valid = repeats > repeat
        members[valid, repeat] = order[starts[valid] + repeat]
    return members


def _common_panels(parameters: np.ndarray, members: np.ndarray) -> np.ndarray | None:
    """A stationary reliability mixture, if every direction has the same panels."""
    panels, labels = np.unique(parameters, axis=0, return_inverse=True)
    if len(panels) == 1:
        return panels
    if np.any(members < 0):
        return None
    grouped = np.sort(labels[members], axis=1)
    return panels[grouped[0]] if np.all(grouped == grouped[0]) else None


def _stationary_weights(
    fans: np.ndarray, parameters: np.ndarray, common: np.ndarray, counts: tuple[int, ...]
) -> np.ndarray:
    """The same conjugate denominator, evaluated once per fixed arrangement."""
    weights = np.empty_like(fans)
    first = 0
    for count in counts:
        ray = fans[first]
        denominator = np.mean(
            _reliability(ray[None, :], common[:, None, :])
            + _reliability(-ray[None, :], common[:, None, :]),
            axis=0,
        )
        weights[first : first + count] = _reliability(ray, parameters[first]) / denominator
        first += count
    return weights


def _mean_reliability(
    groups: np.ndarray,
    rays: np.ndarray,
    parameters: np.ndarray,
    members: np.ndarray,
    repeats: np.ndarray,
) -> np.ndarray:
    """Mean reliability over the actual panels at each queried direction."""
    total = np.zeros(np.broadcast_shapes(groups.shape, rays.shape))
    for repeat in range(members.shape[1]):
        index = members[groups, repeat]
        total += np.where(index >= 0, _reliability(rays, parameters[index]), 0)
    return total / repeats[groups]


def full_turn_weights(
    angles: np.ndarray,
    groups: np.ndarray,
    repeats: np.ndarray,
    fans: np.ndarray,
    pixels: np.ndarray,
    counts: tuple[int, ...],
) -> np.ndarray:
    """Angular weights for per-view fans, including changing or repeated panels.

    ``counts`` partitions the views into runs sharing one fan. Pixel centres
    lie half a pixel inside their panel edges. Reliability tends continuously
    to equal weighting as the panel centres; absent conjugates receive none.
    """
    lo, hi = fans.min(axis=1), fans.max(axis=1)
    if np.any((lo > pixels) | (hi < -pixels)):
        raise ValueError("FDK needs the rotation axis to project onto the detector")
    parameters = np.column_stack([lo - pixels / 2, hi + pixels / 2, pixels])
    members = _members(groups, repeats)
    order = np.argsort(angles)
    nodes = angles[order]
    wrapped = np.r_[nodes[-1] - 2 * np.pi, nodes, nodes[0] + 2 * np.pi]
    measure = np.empty(len(angles))
    measure[order] = (wrapped[2:] - wrapped[:-2]) / 2
    common = _common_panels(parameters, members)
    if common is not None:
        weights = _stationary_weights(fans, parameters, common, counts)
    else:
        own = _reliability(fans, parameters[:, None, :])
        current = _mean_reliability(groups[:, None], fans, parameters, members, repeats)
        conjugate = np.remainder(angles[groups, None] + np.pi + 2 * fans, 2 * np.pi)
        right = np.searchsorted(nodes, conjugate, side="right") % len(nodes)
        left = (right - 1) % len(nodes)
        fraction = np.remainder(conjugate - nodes[left], 2 * np.pi) / np.remainder(
            nodes[right] - nodes[left], 2 * np.pi
        )
        opposite = (1 - fraction) * _mean_reliability(
            order[left], -fans, parameters, members, repeats
        ) + fraction * _mean_reliability(order[right], -fans, parameters, members, repeats)
        weights = own / (current + opposite)
    return weights * measure[groups, None] / repeats[groups, None]
