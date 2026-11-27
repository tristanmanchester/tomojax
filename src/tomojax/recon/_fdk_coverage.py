"""Full-turn fan redundancy from the acquisition's conjugate detector coverage.

The circular, equatorial conjugate of (beta, gamma) is
(beta + pi + 2*gamma, -gamma). A changing panel must share a ray's measure
with that view's panel, not with a mirror of its own. Repeated directions
share both sides' measure. Noncircular FDK remains an approximate inverse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tomojax._typed_arrays import Float64Array


def reliability(ray: Float64Array, parameters: Float64Array) -> Float64Array:
    """Smooth edge reliability odds, recovering Wang's sin-squared partition."""
    lo, hi, pixel = parameters[..., 0], parameters[..., 1], parameters[..., 2]
    width = np.maximum(np.minimum(-lo, hi), pixel)
    distance = np.minimum(ray - lo, hi - ray)
    return np.tan(np.pi / 4 * np.clip(distance / width, 0, 1)) ** 2


def _members(groups: NDArray[np.int64], repeats: NDArray[np.int64]) -> NDArray[np.int64]:
    """View indices at each distinct direction; -1 pads unequal repeat counts."""
    order = np.argsort(groups, kind="stable")
    starts = np.concatenate(([0], np.cumsum(repeats)[:-1]))
    members = np.full((len(repeats), repeats.max().item()), -1, dtype=int)
    for repeat in range(members.shape[1]):
        valid = repeats > repeat
        members[valid, repeat] = order[starts[valid] + repeat]
    return members


def _common_panels(parameters: Float64Array, members: NDArray[np.int64]) -> Float64Array | None:
    """A stationary reliability mixture, if every direction has the same panels."""
    panels, labels = np.unique(parameters, axis=0, return_inverse=True)
    if len(panels) == 1:
        return panels
    if np.any(members < 0):
        return None
    grouped = np.sort(labels[members], axis=1)
    return panels[grouped[0]] if np.all(grouped == grouped[0]) else None


def _stationary_weights(
    fans: Float64Array, parameters: Float64Array, common: Float64Array, counts: tuple[int, ...]
) -> np.ndarray:
    """The same conjugate denominator, evaluated once per fixed arrangement."""
    weights = np.empty_like(fans)
    first = 0
    for count in counts:
        ray: Float64Array = fans[first]
        denominator: Float64Array = np.mean(
            reliability(ray[None, :], common[:, None, :])
            + reliability(-ray[None, :], common[:, None, :]),
            axis=0,
        )
        own = reliability(ray, parameters[first])
        weights[first : first + count] = np.divide(
            own, denominator, out=np.zeros_like(own), where=denominator > 0
        )
        first += count
    return weights


def _mean_reliability(
    groups: NDArray[np.int64],
    rays: Float64Array,
    parameters: Float64Array,
    members: NDArray[np.int64],
    repeats: NDArray[np.int64],
) -> np.ndarray:
    """Mean reliability over the actual panels at each queried direction."""
    total = np.zeros(np.broadcast_shapes(groups.shape, rays.shape))
    for repeat in range(members.shape[1]):
        index = members[groups, repeat]
        total += np.where(index >= 0, reliability(rays, parameters[index]), 0)
    return total / repeats[groups]


def pair_reliability(
    angles: Float64Array,
    groups: NDArray[np.int64],
    repeats: NDArray[np.int64],
    fans: Float64Array,
    parameters: Float64Array,
    selected: NDArray[np.int64],
) -> tuple[Float64Array, Float64Array]:
    """Reliability on both sides of selected rays, interpolating the conjugate panel in angle."""
    members = _members(groups, repeats)
    order = np.argsort(angles)
    nodes = angles[order]
    current = _mean_reliability(selected[:, None], fans, parameters, members, repeats)
    conjugate = np.remainder(angles[selected, None] + np.pi + 2 * fans, 2 * np.pi)
    right = np.searchsorted(nodes, conjugate, side="right") % len(nodes)
    left = (right - 1) % len(nodes)
    fraction = np.remainder(conjugate - nodes[left], 2 * np.pi) / np.remainder(
        nodes[right] - nodes[left], 2 * np.pi
    )
    opposite = (1 - fraction) * _mean_reliability(
        order[left], -fans, parameters, members, repeats
    ) + fraction * _mean_reliability(order[right], -fans, parameters, members, repeats)
    return current, opposite


def full_turn_weights(
    angles: Float64Array,
    groups: NDArray[np.int64],
    repeats: NDArray[np.int64],
    fans: Float64Array,
    pixels: Float64Array,
    counts: tuple[int, ...],
) -> np.ndarray:
    """Angular weights for per-view fans, including changing or repeated panels.

    ``counts`` partitions the views into runs sharing one fan. Pixel centers
    lie half a pixel inside their panel edges. Reliability tends continuously
    to equal weighting as the panel centers; absent conjugates receive none.
    """
    lo, hi = fans.min(axis=1), fans.max(axis=1)
    parameters = np.column_stack([lo - pixels / 2, hi + pixels / 2, pixels])
    members = _members(groups, repeats)
    order = np.argsort(angles)
    nodes = angles[order]
    wrapped = np.concatenate((nodes[-1:] - 2 * np.pi, nodes, nodes[:1] + 2 * np.pi))
    measure = np.empty(len(angles))
    measure[order] = (wrapped[2:] - wrapped[:-2]) / 2
    common = _common_panels(parameters, members)
    if common is not None:
        weights = _stationary_weights(fans, parameters, common, counts)
    else:
        own = reliability(fans, parameters[:, None, :])
        current, opposite = pair_reliability(angles, groups, repeats, fans, parameters, groups)
        total = current + opposite
        weights = np.divide(own, total, out=np.zeros_like(own), where=total > 0)
    return weights * measure[groups, None] / repeats[groups, None]
