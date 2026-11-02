"""Bounded measurement-only noise statistics for automatic regularization."""

from __future__ import annotations

import math

import jax
from jax.core import Tracer
import jax.numpy as jnp
import numpy as np


def _gather_samples(array, coordinates):
    """Gather only requested pixels, including broadcast observation weights."""
    if not hasattr(array, "shape"):
        array = np.asarray(array)
    if not array.shape:
        return array
    indices = tuple(
        np.zeros_like(index) if extent == 1 else index
        for index, extent in zip(coordinates[-len(array.shape) :], array.shape, strict=True)
    )
    return array[indices]


def _require_samples(present):
    if not bool(present):
        raise ValueError("Noise level cannot be estimated: pass tv_weight explicitly.")


def estimate_projection_noise(projections, *, weights=None):
    """Estimate one measurement SD from at most 65536 sampled Haar cells.

    A fixed seed samples cells uniformly with replacement, independently of the
    measured values. Smaller stacks use every cell. Two detector axes use the
    orthonormal HH coefficient; a singleton axis uses the one-dimensional H
    coefficient. Unpaired odd edge pixels are omitted. The statistic is the
    global centered MAD divided by the Gaussian consistency factor, not a
    median of per-view estimates. Object texture may contribute to this scale;
    it is not an independently identified photon-noise model.

    NumPy/memmap and JAX arrays are indexed before conversion, so neither the
    projection stack nor a stack-sized coefficient buffer is copied. Weights
    may broadcast to the stack: any nonpositive corner excludes its whole
    sampled cell, while positive magnitudes do not imply inverse variance.
    No valid sampled cells raises ValueError (a wrapped runtime error inside
    JIT); this does not assert that no valid cells exist elsewhere in the scan.
    A 1x1 detector has no spatial noise statistic. Automatic hyperparameters
    are frozen: the JAX result has zero derivative through this estimator.
    """
    n, nv, nu = projections.shape
    dv, du = (2 if nv > 1 else 1), (2 if nu > 1 else 1)
    if dv * du == 1:
        _require_samples(False)
    rows, columns = nv // dv, nu // du
    population = n * rows * columns
    selected = (
        np.arange(population)
        if population <= 65536
        else np.random.default_rng(0).integers(population, size=65536)
    )
    views, cells = np.divmod(selected, rows * columns)
    row, column = np.divmod(cells, columns)
    on_device = any(isinstance(x, jax.Array | Tracer) for x in (projections, weights))
    xp = jnp if on_device else np
    dtype = jnp.result_type(projections.dtype, jnp.float32) if on_device else np.float64
    coefficients = xp.zeros(selected.size, dtype=dtype)
    valid = xp.ones(selected.size, dtype=bool)
    for u in range(du):
        for v in range(dv):
            coordinates = (views, dv * row + v, du * column + u)
            values = xp.asarray(_gather_samples(projections, coordinates), dtype=dtype)
            coefficients = coefficients + (-1) ** (u + v) * values
            if weights is not None:
                valid = valid & (xp.asarray(_gather_samples(weights, coordinates)) > 0)
    coefficients = coefficients / math.sqrt(dv * du)
    if weights is not None:
        present = xp.any(valid)
        if isinstance(present, Tracer):
            jax.debug.callback(_require_samples, present)
        else:
            _require_samples(present)
        coefficients = xp.where(valid, coefficients, xp.nan)
    median = xp.nanmedian(coefficients)
    sigma = xp.nanmedian(xp.abs(coefficients - median)) / 0.6744897501960817
    return jax.lax.stop_gradient(sigma) if on_device else sigma
