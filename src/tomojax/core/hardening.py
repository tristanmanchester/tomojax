"""The single-material beam hardening curve: a positive mixture of exponentials.

With one material, an energy-integrating detector and no scatter, the measured
line integral of a polychromatic beam is ``q = g(p) = -ln sum_j w_j exp(-k_j p)``,
where ``p`` is the material's line integral in the units of a reference
attenuation, ``w_j >= 0`` (summing to one) is the detected spectrum and ``k_j > 0``
each component's attenuation relative to the reference. Transmission
``exp(-g)`` is then completely monotone in ``p``, and by Bernstein's theorem
every such curve has this form: the mixture is the whole physical class, so
any curve built from it is increasing, concave and asymptotically linear.

Arrays here are float64 on the host, except :func:`invert`, which maps float32
line integrals on the device through a tabulated inverse.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import nnls
from scipy.special import logsumexp

if TYPE_CHECKING:
    from collections.abc import Callable

# Rates of a fitted mixture: a log-spaced grid from 1/64 to 64 times the mean
# measured attenuation, wide enough for unfiltered low-voltage beams and hard
# filtered ones.
RATE_SPAN = np.geomspace(1 / 64, 64, 61)
_TABLE = 8192
# Rays a fit reads at a time.
_CHUNK = 1 << 22
# The fastest rate a fit may use, in inverse shortest fitted paths.
_FASTEST = 3.0


def normalise(
    rates: tuple[float, ...], weights: tuple[float, ...], *, unit_slope: bool
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Weights summing to one, components with no weight dropped, sorted by rate.

    With ``unit_slope`` the rates are scaled so the curve's slope at zero,
    ``sum_j w_j k_j``, is one: thin paths keep their attenuation.
    """
    k = np.asarray(rates, np.float64)
    w = np.asarray(weights, np.float64)
    if k.ndim != 1 or k.shape != w.shape or k.size == 0:
        raise ValueError("rates and weights must be equal-length, nonempty sequences")
    if not (np.isfinite(k).all() and np.isfinite(w).all()):
        raise ValueError("rates and weights must be finite")
    if (k <= 0).any() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("rates must be positive, and weights nonnegative with a positive sum")
    keep = w > 0
    k, w = k[keep], w[keep] / w[keep].sum()
    if unit_slope:
        k = k / float(w @ k)
    order = np.argsort(k, kind="stable")
    return tuple(float(x) for x in k[order]), tuple(float(x) for x in w[order])


def _pointwise(function: Callable[[np.ndarray], np.ndarray], p: np.ndarray) -> np.ndarray:
    """``function`` of float64 ``p``, a few hundred thousand values at a time."""
    p = np.asarray(p, np.float64)
    flat = p.reshape(-1)
    out = np.empty_like(flat)
    step = max(1, _CHUNK // 16)
    for first in range(0, flat.size, step):
        out[first : first + step] = function(flat[first : first + step])
    return out.reshape(p.shape)


def curve(rates: tuple[float, ...], weights: tuple[float, ...], p: np.ndarray) -> np.ndarray:
    """The measured line integral ``g(p)`` of material line integral ``p``."""
    log_w, k = np.log(np.asarray(weights)), np.asarray(rates)
    return _pointwise(lambda x: -np.asarray(logsumexp(log_w - np.multiply.outer(x, k), axis=-1)), p)


def slope(rates: tuple[float, ...], weights: tuple[float, ...], p: np.ndarray) -> np.ndarray:
    """``g'(p)``, the mean rate of the spectrum hardened by ``p``."""
    log_w, k = np.log(np.asarray(weights)), np.asarray(rates)

    def mean_rate(x: np.ndarray) -> np.ndarray:
        logits = log_w - np.multiply.outer(x, k)
        return np.exp(logits - logsumexp(logits, axis=-1, keepdims=True)) @ k

    return _pointwise(mean_rate, p)


def inverse(rates: tuple[float, ...], weights: tuple[float, ...], q: np.ndarray) -> np.ndarray:
    """``p`` with ``g(p) = q``, by Newton's method in float64.

    ``g`` is concave with ``g(p) <= g'(0) p``, so from ``p = q / g'(0)`` the
    iterates rise monotonically to the root and never overshoot it.
    """
    q = np.asarray(q, np.float64)
    p = q / float(np.dot(rates, weights))
    for _ in range(200):
        step = (q - curve(rates, weights, p)) / slope(rates, weights, p)
        p = p + step
        if np.all(np.abs(step) <= 1e-13 * (1 + np.abs(p))):
            break
    return p


@lru_cache(maxsize=64)
def _table(rates: tuple[float, ...], weights: tuple[float, ...]) -> tuple[float, float, np.ndarray]:
    """Values and slopes of the inverse on a uniform grid of ``q``, for Hermite interpolation.

    The grid ends where the curve is linear to float32 precision, or at
    ``q = 60`` (a transmission of 1e-26) if that comes first; beyond either end
    the inverse continues along its end tangent.
    """
    k, w = np.asarray(rates), np.asarray(weights)
    p_end = 1.0
    if k.size > 1:
        # The components above the slowest decay as exp(-(k - k0) p) relative to it.
        gap = k[1:] - k[0]
        p_end = float(np.max(np.log(np.maximum(w[1:] / w[0], 1e-300) * 1e8) / gap))
    q_hi = float(min(60.0, max(1.0, curve(rates, weights, np.asarray(max(p_end, 1.0))))))
    q_lo = -0.5
    q = np.linspace(q_lo, q_hi, _TABLE)
    p = inverse(rates, weights, q)
    derivative = 1.0 / slope(rates, weights, p)
    nodes = np.stack([p, derivative], axis=1).astype(np.float32)
    return q_lo, (q_hi - q_lo) / (_TABLE - 1), nodes


def invert(rates: tuple[float, ...], weights: tuple[float, ...], q: jax.Array) -> jax.Array:
    """Material line integrals ``g^-1(q)`` of float32 line integrals ``q``, on the device.

    Cubic Hermite interpolation between tabulated values and slopes of the
    inverse, and its end tangents outside the table. Non-finite values pass through.
    """
    q_lo, h, nodes = _table(rates, weights)
    table = jnp.asarray(nodes)
    last = nodes.shape[0] - 1
    t = (q - q_lo) / h
    i = jnp.clip(jnp.floor(t), 0, last - 1).astype(jnp.int32)
    s = jnp.clip(t - i, 0.0, 1.0)
    p0, d0 = table[i, 0], table[i, 1] * h
    p1, d1 = table[i + 1, 0], table[i + 1, 1] * h
    s2, s3 = s * s, s * s * s
    inside = (
        (2 * s3 - 3 * s2 + 1) * p0
        + (s3 - 2 * s2 + s) * d0
        + (3 * s2 - 2 * s3) * p1
        + (s3 - s2) * d1
    )
    below = table[0, 0] + (q - q_lo) * table[0, 1]
    above = table[last, 0] + (q - (q_lo + last * h)) * table[last, 1]
    return jnp.where(t < 0, below, jnp.where(t > last, above, inside))


def _nnls(
    length: np.ndarray, transmission: np.ndarray, weight: np.ndarray, rates: np.ndarray
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Nonnegative weights of ``rates`` summing to one that best predict binned transmission."""
    root = np.sqrt(weight)
    design = root[:, None] * np.exp(-np.multiply.outer(length, rates))
    target = root * transmission
    # The weights sum to one (no absorption without material) through a heavy row.
    heavy = 1e3 * float(np.sqrt(weight.sum()))
    design = np.vstack([design, np.full((1, rates.size), heavy)])
    target = np.append(target, heavy)
    w, _ = nnls(design, target, maxiter=50 * rates.size)
    if w.sum() <= 0:
        raise ValueError("the beam hardening fit found no positive mixture")
    return normalise(tuple(rates), tuple(w), unit_slope=False)


def fit(
    length: np.ndarray,
    measured: np.ndarray,
    *,
    bins: int = 512,
    rates: np.ndarray | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...], dict[str, float]]:
    """The mixture that best predicts measured line integrals from path lengths.

    ``length`` and ``measured`` are paired samples, a path length through the
    material (any length unit, zero outside it) and the line integral
    measured along that ray. Transmission ``exp(-q)`` is linear in the
    weights, so the fit is a nonnegative least-squares problem: each ray is
    weighted by the inverse of its Poisson variance, ``1 / exp(-q)`` up to the
    unknown dose, and rays are pooled into ``bins`` intervals of length first,
    reading a few million at a time.
    ``rates`` (per unit length) default to :data:`RATE_SPAN` times the mean
    measured attenuation per unit length; those above ``_FASTEST`` e-folds
    over the shortest path fitted are left out, since such a component would
    be absorbed unseen yet set the curve's slope at zero.

    Returns rates (per unit length), weights and the root-mean-square misfit
    of the bins' mean line integrals.
    """
    x_all = np.asarray(length).reshape(-1)
    q_all = np.asarray(measured).reshape(-1)
    chunks = [slice(i, i + _CHUNK) for i in range(0, x_all.size, _CHUNK)]

    def valid(part: slice) -> tuple[np.ndarray, np.ndarray]:
        x, q = x_all[part].astype(np.float64), q_all[part].astype(np.float64)
        keep = np.isfinite(x) & np.isfinite(q) & (x >= 0)
        return x[keep], q[keep]

    longest = max((float(valid(c)[0].max(initial=0.0)) for c in chunks), default=0.0)
    if not longest > 0:
        raise ValueError("no ray passes through the material")
    sums = np.zeros((5, bins))  # weight, weight x, weight T, T, T q, by bin of length
    moment = np.zeros(2)
    for part in chunks:
        x, q = valid(part)
        transmission = np.exp(-q)
        weight = 1.0 / np.maximum(transmission, 1e-3)
        which = np.minimum((x / longest * bins).astype(np.int64), bins - 1)
        for row, values in enumerate(
            (weight, weight * x, weight * transmission, transmission, transmission * q)
        ):
            sums[row] += np.bincount(which, values, bins)
        through = x > 0
        moment += (np.sum(q[through] * x[through]), np.sum(x[through] ** 2))
    used = sums[0] > 0
    total = sums[0][used]
    xb, tb = sums[1][used] / total, sums[2][used] / total
    if rates is None:
        rates = RATE_SPAN * max(moment[0] / moment[1], 1e-12)
    k = np.asarray(rates, np.float64)
    # A component absorbed before the shortest path fitted is invisible, yet
    # would set the curve's slope at zero: rates stay below _FASTEST over it.
    shortest = float(xb[xb > 0].min(initial=np.inf))
    if np.isfinite(shortest) and (k <= _FASTEST / shortest).any():
        k = k[k <= _FASTEST / shortest]
    fitted_rates, fitted_weights = _nnls(xb, tb, total, k)
    # Rates slower than the curve's slope at the longest path fitted would only
    # bend it beyond the data, so the curve continues no flatter than half that
    # slope: a thicker path elsewhere in the scan is corrected conservatively.
    end = float(slope(fitted_rates, fitted_weights, np.asarray(xb.max())))
    if (k < 0.5 * end).any() and (k >= 0.5 * end).any():
        fitted_rates, fitted_weights = _nnls(xb, tb, total, k[k >= 0.5 * end])
    # The misfit of each bin's mean line integral, weighted as the rays are.
    precision = sums[3][used]
    mean_q = sums[4][used] / np.maximum(precision, 1e-300)
    residual = curve(fitted_rates, fitted_weights, xb) - mean_q
    rms = float(np.sqrt(np.sum(precision * residual**2) / np.sum(precision)))
    return fitted_rates, fitted_weights, {"rms": rms}
