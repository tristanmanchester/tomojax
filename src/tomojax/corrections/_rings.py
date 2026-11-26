"""Detector-response correction with geometric or statistical separation from the object.

Each column reports ``q + a + b (1 - exp(-q))``. Independent-view regressions
fit neighboring-column references. Parallel full turns use opposed-ray parity;
other scans use angular covariance and a spatial scene-power prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from itertools import pairwise
from typing import TYPE_CHECKING, ClassVar, cast

import jax
import jax.numpy as jnp
import numpy as np

from tomojax._data import views_of
from tomojax.core.robust import MAD_TO_SD
from tomojax.errors import InvalidParameter
from tomojax.geometry import (
    ConeGeometry,
    ConeSegments,
    LaminographyGeometry,
    ParallelGeometry,
    RotationAxisGeometry,
)

from ._records import Correction
from ._ring_offsets import correlate, offset_prior, quadratic_taps, shifted_columns, window_sum
from ._ring_parity import regularize
from ._ring_reference_cuda import reference as cuda_reference
from ._ring_turn import turn_response
from ._steps import Domain, Step, bitonic_median

if TYPE_CHECKING:
    from tomojax.geometry import ScanGeometry

WINDOWS = (30, 20, 14, 9, 6)


def _weighted_quadratics(
    values: jax.Array, weights: jax.Array, windows: tuple[int, ...], guard: int
) -> list[jax.Array]:
    """Each column's weighted least-squares quadratic through its neighbors, per window.

    Neighbors are taken relative to the column's own value and their offsets scaled to
    [-1, 1] at each window, then the 3x3 normal equations are solved by Cholesky, so float32
    keeps the local variation's digits. The windows share their sums over common neighbors.
    """
    widest = max(windows)
    values_shifted = shifted_columns(values, widest)
    weights_shifted = shifted_columns(weights, widest)
    s = [jnp.zeros_like(values) for _ in range(5)]
    t = [jnp.zeros_like(values) for _ in range(3)]
    fits = {}
    # Pair the even/odd moments too: tiny asymmetric residuals otherwise steer
    # Tukey weights very differently in nearly flat, reflected neighborhoods.
    for j in range(guard + 1, widest + 1):
        left, right = weights_shifted[widest - j], weights_shifted[widest + j]
        dl = left * (values_shifted[widest - j] - values)
        dr = right * (values_shifted[widest + j] - values)
        s = [s[k] + float(j**k) * (right + left if k % 2 == 0 else right - left) for k in range(5)]
        t = [t[k] + float(j**k) * (dr + dl if k % 2 == 0 else dr - dl) for k in range(3)]
        if j not in windows:
            continue
        sk = [m / float(j**k) for k, m in enumerate(s)]
        tk = [m / float(j**k) for k, m in enumerate(t)]
        l00 = jnp.sqrt(sk[0])
        l10, l20 = sk[1] / l00, sk[2] / l00
        l11 = jnp.sqrt(sk[2] - l10 * l10)
        l21 = (sk[3] - l20 * l10) / l11
        l22 = jnp.sqrt(sk[4] - l20 * l20 - l21 * l21)
        y0 = tk[0] / l00
        y1 = (tk[1] - l10 * y0) / l11
        c2 = (tk[2] - l20 * y0 - l21 * y1) / (l22 * l22)
        c1 = (y1 - l21 * c2) / l11
        fits[j] = values + (y0 - l10 * c1 - l20 * c2) / l00
    # On CUDA, materialize the five fits before adaptive weighting. Otherwise
    # fusion expands them into dozens of full-sized intermediates and repeats
    # their algebra in the following regression reductions. CPU keeps its fusion.
    return jax.lax.platform_dependent(
        [fits[w] for w in windows],
        cuda=jax.lax.optimization_barrier,
        default=lambda values: values,
    )


def _reference(ranked: jax.Array, gamma: float = 2.0, local: int = 64) -> jax.Array:
    """Adaptive spatial reference, with a tiled float32 CUDA implementation."""
    if ranked.dtype != jnp.float32:
        return _reference_jax(ranked, gamma, local)
    taps = tuple(tuple(quadratic_taps(w, 1)) for w in WINDOWS)
    return jax.lax.platform_dependent(
        ranked,
        cuda=lambda values: cuda_reference(values, taps, gamma, local),
        default=lambda values: _reference_jax(values, gamma, local),
    )


def _reference_jax(ranked: jax.Array, gamma: float = 2.0, local: int = 64) -> jax.Array:
    """The adaptive spatial reference for rows of line integrals.

    It starts from the local quadratic through +-30 columns and moves toward the +-20, 14, 9 and 6
    ones only as far as each differs from it by more than ``gamma`` times its noise at that view
    and place, so roughness of the object's values (an edge turning at some radius) is
    followed rather than turned into a column correction. Neighbors are Tukey-weighted by their
    misfit to the +-6 quadratic so that a defective column does not steer them.
    """
    taps = [quadratic_taps(w, 1) for w in WINDOWS]
    r = ranked - correlate(ranked, taps[-1])
    sigma = MAD_TO_SD * jnp.median(jnp.abs(r), axis=1, keepdims=True)
    u = r / (4.685 * jnp.maximum(sigma, 1e-3 * jnp.median(sigma) + 1e-12))
    weights = 1e-3 + jnp.where(jnp.abs(u) < 1, (1 - u * u) ** 2, 0.0)
    refs = _weighted_quadratics(ranked, weights, WINDOWS, 1)
    # the noise: local scatter about the narrowest fit (median of every 8th column within
    # +-local), carried through each pair of references with the columns' correlation
    r = ranked - refs[-1]
    shifts = shifted_columns(jnp.abs(r), local)
    scatter = bitonic_median([shifts[local + k] for k in range(-local, local + 1, 8)])
    scatter = MAD_TO_SD * scatter / np.sqrt(1 + (taps[-1] ** 2).sum())
    r = r - r.mean(1, keepdims=True)
    var = jnp.maximum((r * r).mean(1, keepdims=True), jnp.finfo(r.dtype).tiny)
    rho = [
        jnp.clip((r[:, k:] * r[:, :-k]).mean(1, keepdims=True) / var, 0, 1)
        if k < r.shape[1]
        else jnp.zeros_like(var)
        for k in (1, 2)
    ]
    out = refs[0]
    for w, t, ref in zip(WINDOWS[1:], taps[1:], refs[1:], strict=True):
        d = np.pad(t, WINDOWS[0] - w) - taps[0]
        lags = [float((d[k:] * d[:-k]).sum()) for k in (1, 2)]
        noise = scatter * jnp.sqrt(float((d * d).sum()) + 2 * (rho[0] * lags[0] + rho[1] * lags[1]))
        diff = ref - out
        out = out + jnp.sign(diff) * jnp.maximum(jnp.abs(diff) - gamma * noise, 0.0)
    return out


def _feasible_response(
    q: jax.Array, ref: jax.Array, alpha: jax.Array, beta: jax.Array, lowest: jax.Array | None = None
) -> tuple[jax.Array, jax.Array]:
    """Refit regularized coefficients to cover every observation on a stable branch.

    For negative beta, derivative >= 1/2 requires p >= log(-2 beta).
    Its image begins at alpha + h(beta), h = log(-2 beta) + beta + 1/2.
    Concavity makes the tangent to h an upper bound. Projecting onto
    alpha + tangent <= min(q) therefore guarantees an admissible inverse.
    This is the least change in the reference regression's ridge Gram metric,
    not a clipped output or an unconstrained fit followed by a rejected column.
    ``lowest`` is each column's least observation when ``q`` is only a sample of them.
    """
    x = -jnp.expm1(-ref)
    w = jnp.exp(-jnp.minimum(ref, 50.0))
    sw = w.sum(0)
    xm = (w * x).sum(0) / sw
    variance = (w * (x - xm) ** 2).sum(0) / sw + 1e-4
    negative = jnp.where(beta < 0, beta, -1.0)
    h = jnp.log(-2 * negative) + negative + 0.5
    normal = 1 + 1 / negative - xm
    lowest = q.min(0) if lowest is None else lowest
    excess = jnp.where(beta < 0, jnp.maximum(alpha + h - lowest, 0), 0)
    amount = excess / (variance + normal * normal)
    return alpha + amount * (normal * xm - variance), beta - amount * normal


def _invert(q: jax.Array, alpha: jax.Array, beta: jax.Array) -> jax.Array:
    """Invert an admissible response by Halley updates from monotone-branch bounds."""
    v = q - alpha
    b = jnp.maximum(beta, jnp.finfo(q.dtype).tiny)
    # For b > 0, U = max(v-b, 0) + log1p(b) is an upper bound when v >= 0.
    # Substituting U into p = log(b) - log(p+b-v) supplies a useful lower bound.
    logarithmic = jnp.log(b) - jnp.log(jnp.maximum(b - v, 0) + jnp.log1p(b))
    positive = jnp.maximum(jnp.maximum(v / (1 + b), v - b), logarithmic)
    negative = jnp.maximum(v, -jnp.log1p(jnp.maximum(-v, 0) / b))
    # Positive beta is concave: start below the root. For convex negative beta,
    # v-beta starts above the root on the fitted derivative >= 1/2 branch.
    p = jnp.where(beta < 0, v - beta, jnp.where(v >= 0, positive, negative))
    for _ in range(3):
        e = jnp.exp(-p)
        residual = p - beta * (e - 1) - v
        slope, curvature = 1 + beta * e, -beta * e
        p = p - 2 * residual * slope / (2 * slope * slope - residual * curvature)
    return p


def _fit_unsorted(q: jax.Array, ref: jax.Array, instrument: jax.Array) -> tuple[jax.Array, ...]:
    """Response contrasts and their influence, using independent neighboring-view noise.

    Regressing on the noisy spatial reference itself correlates predictor and error.
    The angular interpolation of the reference, ``instrument``, is the instrument,
    including distinct endpoint views.
    """
    x, z = -jnp.expm1(-ref), -jnp.expm1(-instrument)
    w = jnp.exp(-jnp.minimum(instrument, 50.0))
    d, sw = q - ref, w.sum(0)
    xm, zm, level = (w * x).sum(0) / sw, (w * z).sum(0) / sw, (w * d).sum(0) / sw
    xc, zc = x - xm, z - zm
    denominator = jnp.maximum((w * xc * zc).sum(0), 0) + 1e-4 * sw
    # a dead column sees no spread of transmission, so its slope only amplifies its neighbors
    beta = jnp.clip((w * zc * (d - level)).sum(0) / denominator, -0.5, 0.5)
    residual = d - level - beta * xc
    scene_level = (w * q).sum(0) / sw
    scene_slope = (w * zc * (q - scene_level)).sum(0) / denominator
    return (
        level,
        beta,
        xm,
        w * residual / sw,
        w * zc * residual / denominator,
        scene_level,
        scene_slope,
    )


def _angular_variance(influence: jax.Array, groups: tuple[int, ...]) -> jax.Array:
    """Bartlett sandwich spectral variance, without correlating separate trajectories.

    All overlapping windows avoid a privileged block origin. The finite-sample
    factor restores the iid variance removed by centering each trajectory.
    """
    variance = jnp.zeros(influence.shape[1] // 2 + 1, influence.dtype)
    offset = 0
    for size in groups:
        part = influence[offset : offset + size]
        offset += size
        length = max(1, int(np.sqrt(size)))
        centered = part - part.mean(0, keepdims=True)
        padded = jnp.pad(centered, ((length - 1, length - 1), (0, 0)))
        windows = window_sum(padded, length, "VALID")
        power = jnp.abs(jnp.fft.rfft(windows, axis=1)) ** 2
        effective = size - length + (length * length - 1) / (3 * size)
        variance = variance + power.sum(0) / length * size / max(effective, 1)
    return variance


def _response_posterior(
    profile: jax.Array,
    influence: jax.Array,
    scene: jax.Array,
    taps: np.ndarray,
    groups: tuple[int, ...],
    offset: jax.Array | None = None,
) -> jax.Array:
    """Regularize a local response contrast by angular noise and spatial scene power.

    The local quadratic measures a high-pass contrast, not the response itself.
    Its transfer function is a local approximation for the weighted fit. A white
    detector-field prior supplies its Wiener inverse, while the unfiltered scene
    spectrum protects stationary object structure that angular variance cannot see.
    The separation is statistical: sufficiently fine object structure remains
    indistinguishable from a detector defect without geometric redundancy.
    """
    variance = _angular_variance(influence, groups)
    spectrum = jnp.fft.rfft(profile)
    power = jnp.abs(spectrum) ** 2
    scene_power = jnp.abs(jnp.fft.rfft(scene - scene.mean())) ** 2
    half = len(taps) // 2
    frequency = np.fft.rfftfreq(profile.size)
    response = jnp.asarray(
        1 - np.cos(2 * np.pi * frequency[:, None] * np.arange(-half, half + 1)) @ taps,
        profile.dtype,
    )
    h2 = response * response
    prior = jnp.maximum(jnp.sum(h2 * (power - variance)) / jnp.sum(h2 * h2), 0)
    edges = np.unique(np.r_[0, np.geomspace(1, len(power), 25).astype(int)])
    gains = []
    for lo, hi in pairwise(edges):
        h = response[lo:hi]
        object_power = jnp.maximum(scene_power[lo:hi].mean() - prior, 0)
        gains.append(prior * h / ((prior + object_power) * h * h + variance[lo:hi].mean() + 1e-20))
    gain = jnp.concatenate(gains).at[0].set(0)
    estimate = spectrum * gain
    if offset is not None:
        # Update the prior mean with the observed local contrast. Where this
        # stencil resolves the field, the observation supersedes the prior.
        estimate += (1 - response * gain) * jnp.fft.rfft(offset).at[0].set(0)
    return jnp.fft.irfft(estimate, n=profile.size).astype(profile.dtype)


def _turn_offset(
    q: jax.Array,
    narrow: jax.Array,
    ref: jax.Array,
    beta: jax.Array,
    adjacent: jax.Array,
) -> jax.Array:
    """The column intercept from views where the spatial references agree.

    A moving boundary can bias a broad reference in either direction. The
    narrow-to-broad discrepancy estimates interpolation error. Angular second
    differences measure the noise floor without the constant column offset.
    Interpolate the weighted median between sample mass centers so that the
    fitted intercept changes continuously as a view's weight changes.
    """
    noise = MAD_TO_SD * jnp.median(jnp.abs(q - adjacent), axis=0)
    scale = jnp.maximum(noise, np.spacing(np.ones((), dtype=q.dtype)))
    weights = 1 / (1 + ((ref - narrow) / scale) ** 2)
    residual = q - ref + beta * jnp.expm1(-ref)
    order = jnp.argsort(residual, axis=0)
    mass = jnp.take_along_axis(weights, order, axis=0)
    centered = jnp.cumsum(mass, axis=0) - 0.5 * mass - 0.5 * weights.sum(0)
    hi = jnp.clip(jnp.sum(centered < 0, axis=0), 1, len(q) - 1)
    pair = jnp.stack([hi - 1, hi])
    bounds = jnp.take_along_axis(centered, pair, axis=0)
    left, right = jnp.unstack(bounds)
    values = jnp.take_along_axis(residual, jnp.take_along_axis(order, pair, axis=0), axis=0)
    fraction = jnp.clip(-left / jnp.where(right > left, right - left, 1), 0, 1)
    low, high = jnp.unstack(values)
    return low + fraction * (high - low)


def _row_response(
    q: jax.Array,
    neighbors: jax.Array,
    weights: jax.Array,
    groups: tuple[int, ...],
    lowest: jax.Array,
    passes: int,
    axis: float | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """A row's fitted responses, each after inverting the ones before, and ``q`` corrected.

    ``lowest`` is each column's least value over every view, which keeps the inverse
    admissible everywhere when ``q`` only summarizes the views; the inverse is
    increasing, so it carries the least value through each pass.
    """
    taps = quadratic_taps(6, 0)
    responses: list[tuple[jax.Array, jax.Array]] = []
    for k in range(passes):
        ref = correlate(q, taps)
        instrument = (ref[neighbors] * weights[:, :, None]).sum(1)
        level, beta, xm, il, ib, ml, mb = _fit_unsorted(q, ref, instrument)
        beta = _response_posterior(beta, ib, mb, taps, groups)
        if axis is None:
            offset = offset_prior(q, groups) if k == 0 else None
            level = _response_posterior(level, il, ml, taps, groups, offset)
            beta = jnp.maximum(beta, -0.5)
            alpha = level - beta * xm
        else:
            grid_axis = round(2 * axis) / 2
            # The rank fit supplies the initial broad field. On the second
            # pass the local regression measures the remaining level directly.
            if k == 0:
                level = offset_prior(q, groups)
            transmission = -jnp.expm1(-ref)
            mean_transmission = transmission.mean(0)
            level = regularize(level + beta * (mean_transmission - xm), grid_axis)
            beta = regularize(beta, grid_axis)
            adjacent = (q[neighbors] * weights[:, :, None]).sum(1)
            intercept = regularize(_turn_offset(q, ref, _reference(q), beta, adjacent), grid_axis)
            alpha, beta = turn_response(q, ref, transmission, level, beta, intercept, axis)
        alpha, beta = _feasible_response(q, ref, alpha, beta, lowest)
        responses.append((alpha, beta))
        q, lowest = _invert(q, alpha, beta), _invert(lowest, alpha, beta)
    alphas, betas = zip(*responses, strict=True)
    return jnp.stack(alphas), jnp.stack(betas), q


@dataclass(frozen=True)
class _ViewPlan:
    order: tuple[int, ...]
    neighbors: tuple[tuple[int, int], ...]
    weights: tuple[tuple[float, float], ...]
    groups: tuple[int, ...]
    closing: bool = False


def _angular_order(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order a single turn across the phase wrap, retaining explicit multi-turn labels."""
    if np.ptp(angles) >= 360:
        indices = np.argsort(angles, kind="stable")
        return indices, angles[indices]
    phase = np.mod(angles, 360)
    indices = np.argsort(phase, kind="stable")
    ordered = phase[indices]
    # Cut the circle at the largest unsampled arc, not at the arbitrary zero of
    # the angle labels. A tie favors the smallest phase for deterministic order.
    gaps = np.diff(np.r_[ordered[-1] - 360, ordered])
    cut = int(np.argmax(gaps))
    indices = np.roll(indices, -cut)
    return indices, np.mod(phase[indices] - ordered[cut], 360) + ordered[cut]


def _periodic_order(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool] | None:
    """A uniform full turn, with a repeated closing phase paired across its cut.

    Eligibility and fine-fit ordering share this convention, including modulo
    angle labels. Explicit multi-turn acquisitions retain their noncyclic route.
    """
    size = len(angles)
    if size < 8 or np.ptp(angles) > 360 + 180 / (size - 1):
        return None
    phase = np.mod(angles, 360)
    order = np.argsort(phase, kind="stable")
    gaps = np.diff(np.r_[phase[order], phase[order[0]] + 360])
    short = np.flatnonzero(gaps < 180 / (size - 1))
    closing = len(short) == 1
    if len(short) > 1:
        return None
    spacing = 360 / (size - int(closing))
    ordinary = np.delete(gaps, short)
    if not np.all(np.abs(ordinary - spacing) <= 0.1 * spacing):
        return None
    if closing:
        order = np.roll(order, -int(short[0]))
        order = np.r_[order[:1], order[2:], order[1:2]]
    unwrapped = np.mod(phase[order] - phase[order[0]], 360) + phase[order[0]]
    if closing:
        unwrapped[-1] += 360
    return order, unwrapped, closing


def _view_plan(geometry: object, *, cyclic: bool = False) -> _ViewPlan | None:
    """Angular neighbors within declared acquisition trajectories, not calibration blocks."""
    layout = geometry
    while (base := getattr(layout, "base", None)) is not None:
        layout = base
    if isinstance(layout, ConeSegments):
        stops = np.cumsum(layout.trajectory_lengths)[:-1]
        trajectories = np.split(np.asarray(layout.angles, np.float64), stops)
    elif isinstance(
        layout, ParallelGeometry | ConeGeometry | LaminographyGeometry | RotationAxisGeometry
    ):
        trajectories = (np.asarray(layout.angles, np.float64),)
    else:
        return None
    order, neighbors, weights, groups = [], [], [], []
    offset = 0
    closing = False
    for trajectory in trajectories:
        angles = trajectory
        size = len(angles)
        if size < 2:
            raise ValueError("Rings needs at least two views per trajectory")
        if cyclic:
            periodic = _periodic_order(angles)
            assert periodic is not None
            indices, angles, closing = periodic
        else:
            indices, angles = _angular_order(angles)
        order.extend(int(j + offset) for j in indices)
        if cyclic and closing:
            # A repeated closing phase is one observation for the spatial fit,
            # not an extra angular weight or its own independent instrument.
            angles = np.r_[0.5 * (angles[0] + angles[-1] - 360), angles[1:-1]]
            size -= 1
        i = np.arange(size)
        if cyclic:
            lo, hi = (i - 1) % size, (i + 1) % size
            left, right = (angles - angles[lo]) % 360, (angles[hi] - angles) % 360
            weight = right / (left + right)
        else:
            lo, hi = _open_neighbors(size)
            gap = angles[hi] - angles[lo]
            weight = np.divide(angles[hi] - angles, gap, out=np.full(size, 0.5), where=gap != 0)
        neighbors.extend((int(a + offset), int(b + offset)) for a, b in zip(lo, hi, strict=True))
        weights.extend((float(w), float(1 - w)) for w in weight)
        groups.append(size)
        offset += size
    return _ViewPlan(tuple(order), tuple(neighbors), tuple(weights), tuple(groups), bool(closing))


def _open_neighbors(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Each of ``size`` angle-sorted views' two neighbors; an end view's one, twice."""
    i = np.arange(size)
    lo, hi = np.maximum(i - 1, 0), np.minimum(i + 1, size - 1)
    lo[0], hi[-1] = hi[0], lo[-1]
    return lo, hi


def _bins(
    groups: tuple[int, ...], views: int, *, cyclic: bool = False
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """About ``views`` contiguous angular bins, shared among trajectories by length.

    Returns each view's bin, each bin's two neighboring bins (around the turn when
    ``cyclic``) and the bins per trajectory.
    """
    total, offset, labels, neighbors, counts = sum(groups), 0, [], [], []
    for size in groups:
        k = min(size, max(2, round(views * size / total)))
        labels.append(offset + np.arange(size) * k // size)
        i = np.arange(k)
        lo, hi = ((i - 1) % k, (i + 1) % k) if cyclic else _open_neighbors(k)
        neighbors.append(offset + np.stack([lo, hi], axis=1))
        counts.append(k)
        offset += k
    return np.concatenate(labels), np.concatenate(neighbors), tuple(counts)


@partial(jax.jit, static_argnames=("axis", "groups", "closing", "views", "order"))
def correct_rings(
    batch: jax.Array,
    axis: float | None = None,
    neighbors: jax.Array | None = None,
    weights: jax.Array | None = None,
    groups: tuple[int, ...] = (),
    *,
    closing: bool = False,
    views: int | None = None,
    order: tuple[int, ...] | None = None,
) -> jax.Array:
    """Correct every detector row of a ``(views, rows, columns)`` batch of line integrals.

    ``views`` fits the responses on about that many angular bin means.
    """
    if neighbors is None or weights is None:
        size = batch.shape[0]
        if size < 2:
            raise ValueError("Rings needs at least two views")
        i = np.arange(size)
        lo, hi = _open_neighbors(size) if axis is None else ((i - 1) % size, (i + 1) % size)
        neighbors = jnp.asarray(np.stack([lo, hi], axis=1), jnp.int32)
        weights = jnp.full((size, 2), 0.5, batch.dtype)
        groups = (size,)
    # Only the fitting samples need angular order. Apply the column responses
    # to the original batch rather than materializing and undoing a full-view gather.
    fitted = batch if order is None else batch[jnp.asarray(order, jnp.int32)]
    passes = 2
    if closing:
        # A repeated closing phase is one observation for the fits.
        fitted = jnp.concatenate([0.5 * (fitted[:1] + fitted[-1:]), fitted[1:-1]])
    if views is not None and views < sum(groups):
        # Fit on angular bin means, which keep every view's photons; an additive column
        # error commutes with the mean, and neighboring bins share no views.
        labels, bin_neighbors, groups = _bins(groups, views, cyclic=axis is not None)
        sizes = np.bincount(labels)
        starts = np.cumsum(sizes) - sizes
        means = fitted[starts]
        for j in range(1, sizes.max()):
            means += fitted[starts + np.minimum(j, sizes - 1)] * (j < sizes)[:, None, None]
        fitted = means / sizes[:, None, None].astype(batch.dtype)
        neighbors = jnp.asarray(bin_neighbors, jnp.int32)
        # equal weights: the instrument need only be independent of each bin's noise
        weights = jnp.full(bin_neighbors.shape, 0.5, batch.dtype)
        # bin means are quiet enough for one fit to remove what two do on every view
        passes = 1 if axis is None else 2
    rows = (jnp.swapaxes(fitted, 0, 1), batch.min(0))
    alpha, beta, corrected = jax.lax.map(
        lambda row: _row_response(row[0], neighbors, weights, groups, row[1], passes, axis), rows
    )
    if corrected.shape[1] == batch.shape[0]:
        # every view was fitted on its own, so the fits have already corrected them
        out = jnp.swapaxes(corrected, 0, 1)
        return out if order is None else out[jnp.asarray(np.argsort(order), jnp.int32)]
    for k in range(alpha.shape[1]):
        batch = _invert(batch, alpha[None, :, k], beta[None, :, k])
    return batch


def mirror_axis(geometry: object) -> float | None:
    """The rotation axis's detector column, or None.

    None unless ``geometry`` is a stationary parallel-beam scan over one evenly sampled full turn.
    """
    if type(geometry) is not ParallelGeometry:
        return None
    if _periodic_order(np.asarray(geometry.angles, np.float64)) is None:
        return None
    detector = geometry.detector
    return detector.nu / 2.0 - 0.5 - float(detector.center[0]) / float(detector.du)


@dataclass(frozen=True)
class Rings(Step):
    """Correct ring artifacts from each detector column's own response.

    A column whose gain or response the flat field does not describe reports
    ``q + a + b (1 - exp(-q))`` instead of the line integral ``q``: a stripe in
    the sinogram, a ring in the volume, stronger where the sample absorbs more
    when ``b`` is not zero. The fitted response is inverted in two passes.
    This step does not interpolate detector columns: low angular variation
    alone cannot distinguish a defective pixel from object structure. Raw
    flat/dark calibration interpolates bracketed pixels without measured
    illumination; other defective measurements need independent calibration
    evidence and repair before this step.

    After statistical regularization, a constrained coefficient refit ensures
    that every observation has an inverse on a branch with response derivative
    at least one half. It minimizes coefficient displacement in the reference
    regression metric subject to a conservative linear feasibility constraint;
    corrected line integrals are not clipped. This also modifies the parity
    prior where it would otherwise produce a noninvertible response.

    On a stationary parallel-beam scan over one evenly sampled full turn,
    short angular arcs measure each column's mean response and its change
    with transmission. A cubic interpolation compares opposing arcs at the
    reflected detector coordinates. Repeatable differences between the arcs
    estimate the detector power in a neighborhood around each column. The
    offset fit weights the arc mean by its uncertainty where reflected
    neighborhoods overlap and share the same detector columns.

    A weighted median over views provides a more precise intercept on locally
    quadratic rays. The angular scatter of the local spatial contrast and
    the residual variation across arcs weight the two estimates. The local
    field prior uses reflection on the nearest half-pixel grid. The arc
    comparison uses the recorded physical axis. A symmetric sample feature
    and a symmetric detector offset can still give identical measurements.
    The fit averages a repeated closing phase and uses cyclic angular neighbors.

    Otherwise, each unsorted view is compared with a quadratic through six
    neighboring columns on either side. Independent neighboring views supply
    an instrument for the response fit, avoiding correlation between reference
    noise and regression error. Angular covariance and the unfiltered scene's
    spatial power regularize the local response contrast before inversion.
    Angular rank profiles estimate the column offsets at several spatial
    scales. Persistent jumps between adjacent columns locate the boundaries
    of wider defects. The variation between disjoint angular arcs sets the
    weight of these measurements in the offset fit.
    Smooth object structure is retained rather than interpreted as calibration
    error. This is a statistical separation, not a no-change guarantee: fine
    stationary object structure and broad detector errors can be ambiguous.

    Built-in angular geometries order the views by angle within each trajectory;
    single turns are unwrapped across the largest unsampled arc, explicit
    multi-turn angle labels are retained, and independent cone trajectories
    are kept separate even when their calibration segments are subdivided.
    Without such geometry, views must follow a continuous acquisition trajectory.
    At least two views per trajectory are needed. Output view order is preserved.

    Parameters
    ----------
    geometry : object, optional
        The scan's geometry, with the detector the data were recorded on.
        Default is None.
    views : int or None, optional
        Fit the responses on the means of about this many contiguous angular bins of
        views, neighboring bins as instrument, and correct every view. Default 512.
        Scans without a mirror axis then fit once rather than twice. None fits every
        view: several times slower, for scans whose structure moves across columns
        within a bin.

    Raises
    ------
    ValueError
        If a geometry trajectory contains fewer than two views, or ``views`` is not
        an integer of at least 2.

    Examples
    --------
    >>> import tomojax as tj
    >>> from tomojax.corrections import Rings
    >>> grid = tj.Grid(64, 64, 1, 1.0, 1.0, 1.0)
    >>> turn = tj.ParallelGeometry(
    ...     grid, tj.Detector(64, 1, 1.0, 1.0), [i * 2.0 for i in range(180)]
    ... )
    >>> Rings(turn).mirror_axis
    31.5
    >>> Rings().mirror_axis is None
    True
    """

    domain: ClassVar[Domain] = "line integrals"
    whole_rows: ClassVar[bool] = True
    geometry: object = field(default=None, repr=False, compare=False)
    views: int | None = field(default=512, kw_only=True, metadata={"unit": "view"})
    # what the geometry decides, compared and hashed so compiled kernels are not shared
    _axis: float | None = field(default=None, init=False, repr=False)
    _columns: int | None = field(default=None, init=False, repr=False)
    _plan: _ViewPlan | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.views is not None and (type(self.views) is not int or self.views < 2):
            raise InvalidParameter(
                "views", f"Rings views must be an integer of at least 2, not {self.views!r}"
            )
        axis = mirror_axis(self.geometry)
        object.__setattr__(self, "_axis", axis)
        if axis is not None and isinstance(self.geometry, ParallelGeometry):
            object.__setattr__(self, "_columns", int(self.geometry.detector.nu))
        object.__setattr__(self, "_plan", _view_plan(self.geometry, cyclic=axis is not None))

    def retained(self, kept: np.ndarray) -> Rings:
        """These rings for the views ``kept``, replanned from their geometry.

        Parameters
        ----------
        kept : numpy.ndarray
            Increasing indices of the views that remain.

        Returns
        -------
        Rings
            The same settings, with the angular plan (and full-turn mirror
            eligibility) of the retained views.
        """
        if self.geometry is None:
            return self
        return replace(self, geometry=views_of(cast("ScanGeometry", self.geometry), kept))

    @property
    def mirror_axis(self) -> float | None:
        """The rotation-axis column the mirror stages use.

        Returns
        -------
        float or None
            Physical axis position in detector columns, or None when the geometry is
            not a stationary parallel-beam full turn.
        """
        return self._axis

    def record(self) -> Correction:
        """Build this step's record: the mirror axis used, or None, and the bin count.

        Returns
        -------
        Correction
            Step name and settings, with no findings yet.
        """
        return Correction("rings", {"mirror_axis": self.mirror_axis, "views": self.views})

    @partial(jax.jit, static_argnums=0)
    def apply(self, batch: jax.Array) -> jax.Array:
        """Correct the rings of a batch of detector rows.

        Parameters
        ----------
        batch : jax.Array
            Float32 line integrals of shape ``(views, rows, columns)``, every
            view of each row.

        Returns
        -------
        jax.Array
            Corrected line integrals of the same shape.

        Raises
        ------
        ValueError
            If the geometry's detector is not as wide as the data, the view
            count differs, or fewer than two views are supplied.
        """
        if self._columns is not None and self._columns != batch.shape[2]:
            raise ValueError(
                f"Rings geometry has {self._columns} detector columns, the data {batch.shape[2]}"
            )
        if self._plan is None:
            return correct_rings(batch, self._axis, views=self.views)
        plan = self._plan
        if len(plan.order) != batch.shape[0]:
            raise ValueError(
                f"Rings geometry has {len(plan.order)} views, the data {batch.shape[0]}"
            )
        return correct_rings(
            batch,
            order=plan.order,
            axis=self._axis,
            neighbors=jnp.asarray(plan.neighbors, jnp.int32),
            weights=jnp.asarray(plan.weights, batch.dtype),
            groups=plan.groups,
            closing=plan.closing,
            views=self.views,
        )
