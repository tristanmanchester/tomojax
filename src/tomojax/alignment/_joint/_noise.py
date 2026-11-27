"""Measurement-only noise calibration for the pose innovation prior.

Short Haar cells mix smooth object curvature with detector noise. A normalized
fourth difference has four vanishing moments, so a separable detector contrast
removes cubic trends while retaining unit variance for independent unit noise.
The estimate is frozen before fitting and never uses a reconstruction or poses.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from tomojax.core.robust import MAD_TO_SD

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Protocol

    import jax
    from numpy.typing import DTypeLike

    class ProjectionSource(Protocol):
        @property
        def shape(self) -> tuple[int, ...]: ...

        def __getitem__(
            self, key: slice | tuple[np.ndarray, np.ndarray, np.ndarray], /
        ) -> np.ndarray | jax.Array: ...


# Most residuals any robust statistic here samples.
MAX_SAMPLES = 65_536
# Robust loss width in noise standard deviations, for the fit and its refinement.
LOSS_WIDTH_SIGMAS = 2.5
_NO_CONTRAST = "noise calibration needs an observed detector contrast"


def estimate_noise(
    projections: ProjectionSource,
    *,
    precision: np.ndarray | None = None,
    dtype: DTypeLike | None = None,
    window: tuple[int, int] | None = None,
) -> float:
    """Estimate a measurement SD from bounded fourth-difference contrasts.

    ``projections`` is a host (view, v, u) array; ``precision`` may broadcast
    to it and marks observed pixels by positive entries. Contrast coefficients
    are normalized to unit noise variance. At most 65,536 windows are sampled
    with a fixed, data-independent seed, before accessing their pixels.
    ``dtype`` optionally rounds those samples before FP64 accumulation, matching
    the fit's FP32 observations without converting a whole host stack.

    Both detector axes are used when possible. A short axis reduces the order;
    masks can require a one-dimensional or lower-order contrast. This choice
    depends only on observable stencil support, never on the estimated noise
    or motion. No observable contrast raises ValueError. Correlated noise and
    unresolved object texture can still affect this measurement-only scale.
    """
    mask = None if precision is None else np.broadcast_to(precision, projections.shape)
    shape = projections.shape if window is None else (projections.shape[0], *window)
    for orders in _orders(shape):
        windows = Windows(shape, orders)
        windows.gather(projections, dtype)
        sd = windows.sd(mask)
        if sd is not None:
            return sd
    raise ValueError(_NO_CONTRAST)


class Windows:
    """The contrast windows sampled at one pair of (v, u) orders, chosen before any pixel is read.

    ``views``, ``rows`` and ``columns`` place each window's first pixel, ``taps`` are its
    pixels' offsets, v major, and ``values`` each tap's gathered values in FP64.
    """

    def __init__(self, shape: tuple[int, ...], orders: tuple[int, int] | None = None) -> None:
        """``orders`` None takes the first orders ``estimate_noise`` tries."""
        n, nv, nu = shape
        orders = orders or next(_orders(shape), None)
        if orders is None:
            raise ValueError(_NO_CONTRAST)
        self.orders = vorder, uorder = orders
        rows, columns = nv - vorder, nu - uorder
        population = n * rows * columns
        selected = (
            np.arange(population)
            if population <= MAX_SAMPLES
            else np.random.default_rng(0).integers(population, size=MAX_SAMPLES)
        )
        self.views, cells = np.divmod(selected, rows * columns)
        self.rows, self.columns = np.divmod(cells, columns)
        self.taps = [(v, u) for v in range(vorder + 1) for u in range(uorder + 1)]
        self.values = [np.empty(len(selected)) for _ in self.taps]

    def gather(
        self, projections: ProjectionSource, dtype: DTypeLike | None = None, start: int = 0
    ) -> None:
        """Keep the pixels of the windows in ``projections``, views ``start`` onward.

        A stack read front to back in slabs passes each slab in turn, so nothing reads it at
        random. ``dtype`` rounds the values as ``estimate_noise``'s does.
        """
        inside = np.flatnonzero((self.views >= start) & (self.views < start + projections.shape[0]))
        views, rows, columns = self.views[inside] - start, self.rows[inside], self.columns[inside]
        for (v, u), values in zip(self.taps, self.values, strict=True):
            values[inside] = np.asarray(projections[views, rows + v, columns + u], dtype)

    def sd(self, mask: np.ndarray | None = None) -> float | None:
        """The robust SD of the normalized contrasts, None when no window is wholly observed.

        A pixel is observed where it is finite and ``mask``, if given, is positive.
        """
        vorder, uorder = self.orders
        coefficients = np.zeros(len(self.views), np.float64)
        valid = np.ones(len(self.views), bool)
        normalizer = math.sqrt(math.comb(2 * vorder, vorder) * math.comb(2 * uorder, uorder))
        for (v, u), values in zip(self.taps, self.values, strict=True):
            observed = np.isfinite(values)
            if mask is not None:
                observed &= mask[self.views, self.rows + v, self.columns + u] > 0
            valid &= observed
            weight = (-1) ** (u + v) * math.comb(vorder, v) * math.comb(uorder, u)
            coefficients += weight * np.where(observed, values, 0)
        coefficients = coefficients[valid] / normalizer
        if not coefficients.size:
            return None
        centered = coefficients - np.median(coefficients)
        return float(np.median(np.abs(centered)) * MAD_TO_SD)


def _orders(shape: tuple[int, ...]) -> Iterator[tuple[int, int]]:
    """The (v, u) contrast orders to try, highest first; a short axis lowers its order."""
    _, nv, nu = shape
    tried = set()
    for requested in range(4, 0, -1):
        ov, ou = min(requested, nv - 1), min(requested, nu - 1)
        for orders in ((ov, ou), (0, ou), (ov, 0)):
            if orders != (0, 0) and orders not in tried:
                tried.add(orders)
                yield orders
