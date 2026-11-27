"""Interpolate the two directed samples of a Fourier line independently."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomojax._typed_arrays import Float64Array
    from tomojax.core.cuda_types import CuPyValue


class OpposedDirections:
    """Bracketing measured views for both signs of every measured Fourier direction.

    Weighted partial profiles at angles a and a + pi have complementary
    support. Interpolating between them as adjacent half-turn directions would
    mix two different signals. Interpolate each on the directed circle, then
    sum their transforms with the measured detector's two physical phases.
    """

    def __init__(
        self,
        folded: tuple[float, ...],
        flipped: np.ndarray | tuple[int, ...],
        place: Callable[[np.ndarray], CuPyValue],
    ) -> None:
        directed: Float64Array = np.asarray(folded) + np.asarray(flipped, dtype=np.float64) * np.pi
        nodes, first, counts = np.unique(directed, return_index=True, return_counts=True)
        # Only average repeated exposures. Unique profiles stay in their input
        # slots, including the thousands of ordinary views of an inclusive scan.
        order = np.argsort(directed, kind="stable")
        starts = np.cumsum(counts) - counts
        self.repeats = [
            (first[counts == n], order[starts[counts == n, None] + np.arange(n)])
            for n in np.unique(counts[counts > 1])
        ]
        knots = np.unique(folded)
        self.angles = tuple(np.repeat(knots, 2))
        self.flipped = np.tile(np.array([0, 1], dtype=np.int8), len(knots))
        wanted = np.ravel(np.column_stack((knots, knots + np.pi)))
        self.order = np.argsort(np.argsort(wanted))
        right = np.searchsorted(nodes, wanted) % len(nodes)
        left = (right - 1) % len(nodes)
        span = np.remainder(nodes[right] - nodes[left], 2 * np.pi)
        fraction = np.remainder(nodes[right] - wanted, 2 * np.pi) / span
        left = np.where(fraction == 0, right, left)
        self.exact = np.all(fraction == 0)
        self.identity = (
            self.exact
            and np.array_equal(first[right], np.arange(len(directed)))
            and not self.repeats
        )
        self.left, self.right = first[left], first[right]
        self.fraction = place(fraction[:, None].astype(np.float32))

    def sample(self, raw: np.ndarray | CuPyValue) -> np.ndarray | CuPyValue:
        """Interleave both directions on an owned NumPy or CuPy slab."""
        if self.identity:
            return raw
        for target, members in self.repeats:
            raw[:, target] = raw.take(members, axis=1).mean(axis=2)
        right = raw.take(self.right, axis=1)
        if self.exact:
            return right
        return right + (raw.take(self.left, axis=1) - right) * self.fraction
