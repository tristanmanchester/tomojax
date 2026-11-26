"""Bounded measurement-only noise statistics for automatic regularization."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol, TypeGuard

import jax
from jax.core import Tracer
import jax.numpy as jnp
import numpy as np

from tomojax.core.robust import MAD_TO_SD

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class _SampledStack(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype: ...

    def __getitem__(self, indices: tuple[np.ndarray, ...], /) -> np.ndarray: ...


def _sampled(value: object) -> TypeGuard[_SampledStack]:
    # hasattr, unlike a runtime Protocol check, sees attributes a proxy delegates.
    return hasattr(value, "shape")


def _gather_samples(
    array: _SampledStack | ArrayLike, coordinates: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> _SampledStack | np.ndarray:
    """Gather only requested pixels, including broadcast observation weights."""
    stack = array if _sampled(array) else np.asarray(array)
    shape = stack.shape
    if not shape:
        return stack
    indices = tuple(
        np.zeros_like(index) if extent == 1 else index
        for index, extent in zip(coordinates[-len(shape) :], shape, strict=True)
    )
    return stack[indices]


def _require_samples(present: jax.Array | bool | np.bool_) -> None:
    if not bool(present):
        raise ValueError("Noise level cannot be estimated: pass tv_weight explicitly.")


def _sample_cells(population: int) -> np.ndarray:
    return (
        np.arange(population)
        if population <= 65536
        else np.random.default_rng(0).integers(population, size=65536)
    )


def estimate_projection_noise(
    projections: _SampledStack | jax.Array | np.ndarray,
    *,
    weights: jax.Array | np.ndarray | np.generic | float | None = None,
) -> jax.Array | np.float64:
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
    selected = _sample_cells(population)
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
    sigma = xp.nanmedian(xp.abs(coefficients - median)) * MAD_TO_SD
    return jax.lax.stop_gradient(sigma) if on_device else sigma


def detector_error_scale(projections: jax.Array | np.ndarray) -> jax.Array | np.float64:
    """Combine measurement noise and unresolved subpixel curvature on the detector.

    For a uniform subpixel phase the quadratic interpolation remainder has
    mean D²/12 and variance (D²)**2/720, where D² is a second difference.
    The means along both axes add before squaring. Each squared stencil has
    noise bias 6 sigma**2. Orthogonal stencils share their center pixel, so
    their covariance contributes another 8 sigma**2 to the squared sum.
    """
    sigma = estimate_projection_noise(projections)
    n, nv, nu = projections.shape
    v, u = int(nv >= 3), int(nu >= 3)
    population = n * (nv - 2 * v) * (nu - 2 * u)
    selected = _sample_cells(population)
    views, cells = np.divmod(selected, (nv - 2 * v) * (nu - 2 * u))
    rows, columns = np.divmod(cells, nu - 2 * u)
    rows, columns = rows + v, columns + u
    xp = jnp if isinstance(projections, jax.Array | Tracer) else np
    center = projections[views, rows, columns].astype(np.float32)

    def difference(a: int, b: int) -> jax.Array | np.ndarray:
        left = projections[views, rows - a, columns - b].astype(np.float32)
        right = projections[views, rows + a, columns + b].astype(np.float32)
        return (left - center) + (right - center)

    dv = difference(v, 0)
    du = difference(0, u)
    curvature = xp.mean(dv * dv + du * du) - 6 * (v + u) * sigma**2
    laplacian = xp.mean((du + dv) ** 2) - (6 * (v + u) + 8 * v * u) * sigma**2
    mse = sigma**2 + xp.maximum(curvature, 0) / 720 + xp.maximum(laplacian, 0) / 144
    return jax.lax.stop_gradient(jnp.sqrt(mse)) if xp is jnp else np.float64(np.sqrt(mse))
