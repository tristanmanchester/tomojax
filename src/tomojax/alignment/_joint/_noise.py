"""Measurement-only noise calibration for the pose innovation prior.

Short Haar cells mix smooth object curvature with detector noise. A normalized
fourth difference has four vanishing moments, so a separable detector contrast
removes cubic trends while retaining unit variance for independent unit noise.
The estimate is frozen before fitting and never uses a reconstruction or poses.
"""

import math

import numpy as np

_MAX_SAMPLES = 65_536
_NORMAL_MAD = 0.6744897501960817


def estimate_noise(projections, *, precision=None, dtype=None):
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
    n, nv, nu = projections.shape
    mask = None if precision is None else np.broadcast_to(precision, projections.shape)
    tried = set()
    for requested in range(4, 0, -1):
        ov, ou = min(requested, nv - 1), min(requested, nu - 1)
        for orders in ((ov, ou), (0, ou), (ov, 0)):
            if orders == (0, 0) or orders in tried:
                continue
            tried.add(orders)
            vorder, uorder = orders
            rows, columns = nv - vorder, nu - uorder
            population = n * rows * columns
            selected = (
                np.arange(population)
                if population <= _MAX_SAMPLES
                else np.random.default_rng(0).integers(population, size=_MAX_SAMPLES)
            )
            views, cells = np.divmod(selected, rows * columns)
            row, column = np.divmod(cells, columns)
            coefficients = np.zeros(len(selected), np.float64)
            valid = np.ones(len(selected), bool)
            normalizer = math.sqrt(math.comb(2 * vorder, vorder) * math.comb(2 * uorder, uorder))
            for v in range(vorder + 1):
                for u in range(uorder + 1):
                    indices = (views, row + v, column + u)
                    values = np.asarray(projections[indices], dtype=dtype).astype(
                        np.float64, copy=False
                    )
                    observed = np.isfinite(values)
                    if mask is not None:
                        observed &= mask[indices] > 0
                    valid &= observed
                    weight = (-1) ** (u + v) * math.comb(vorder, v) * math.comb(uorder, u)
                    coefficients += weight * np.where(observed, values, 0)
            coefficients = coefficients[valid] / normalizer
            if coefficients.size:
                centered = coefficients - np.median(coefficients)
                return float(np.median(np.abs(centered)) / _NORMAL_MAD)
    raise ValueError("noise calibration needs an observed detector contrast")
