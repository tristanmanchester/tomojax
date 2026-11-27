"""Interpolate the two directed samples of a Fourier line independently."""

from __future__ import annotations

import numpy as np

from tomojax.core.cuda_types import CuPyValue


class OpposedDirections:
    """Bracketing measured views for both signs of every measured Fourier direction.

    Weighted partial profiles at angles a and a + pi have complementary
    support. Interpolating between them as adjacent half-turn directions would
    mix two different signals. Interpolate each on the directed circle, then
    sum their transforms with the measured detector's two physical phases.
    """

    def __init__(self, xp: CuPyValue, directed: np.ndarray, folded: tuple[float, ...]) -> None:
        nodes, first, counts = np.unique(directed, return_index=True, return_counts=True)
        self.groups = None
        if np.any(counts > 1):
            self.groups = (
                xp.asarray(np.argsort(directed, kind="stable")),
                xp.asarray(np.r_[0, np.cumsum(counts)[:-1]]),
                xp.asarray(counts[:, None]),
            )
            first = np.arange(len(nodes))
        knots = np.unique(folded)
        self.angles = tuple(np.repeat(knots, 2))
        wanted = np.ravel(np.column_stack((knots, knots + np.pi)))
        right = np.searchsorted(nodes, wanted) % len(nodes)
        left = (right - 1) % len(nodes)
        span = np.remainder(nodes[right] - nodes[left], 2 * np.pi)
        fraction = np.remainder(nodes[right] - wanted, 2 * np.pi) / span
        left = np.where(fraction == 0, right, left)
        self.exact = np.all(fraction == 0)
        self.identity = (
            self.exact
            and np.array_equal(first[right], np.arange(len(directed)))
            and self.groups is None
        )
        self.left, self.right = xp.asarray(first[left]), xp.asarray(first[right])
        self.fraction = xp.asarray(fraction[:, None].astype(np.float32))
        self.xp = xp

    def sample(self, raw: CuPyValue) -> CuPyValue:
        """Return the positive and negative profiles interleaved in half-turn order."""
        if self.identity:
            return raw
        if self.groups is not None:
            order, starts, counts = self.groups
            raw = self.xp.add.reduceat(raw[:, order], starts, axis=1) / counts.astype(raw.dtype)
        right = raw[:, self.right]
        if self.exact:
            return right
        return right + (raw[:, self.left] - right) * self.fraction
