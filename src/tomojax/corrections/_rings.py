"""Detector-response correction with geometric or statistical separation from the object.

Each column reports ``q + a + b (1 - exp(-q))``. Independent-view regressions
fit neighbouring-column references. Parallel full turns use opposed-ray parity;
other scans use angular covariance and a spatial scene-power prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache, partial
from itertools import pairwise
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.geometry import (
    ConeGeometry,
    ConeSegments,
    LaminographyGeometry,
    ParallelGeometry,
    RotationAxisGeometry,
)

from ._records import Correction
from ._ring_parity import regularize
from ._steps import Domain, Step, bitonic_median

WINDOWS = (30, 20, 14, 9, 6)


@lru_cache(maxsize=16)
def _quadratic_taps(window: int, guard: int) -> np.ndarray:
    """Weights giving each column the local quadratic through its neighbours (guard left out)."""
    j = np.arange(-window, window + 1, dtype=np.float64)
    keep = np.abs(j) > guard
    design = np.stack([np.ones_like(j), j, j * j], axis=1)[keep]
    taps = np.zeros_like(j)
    taps[keep] = np.linalg.pinv(design)[0]
    return taps


def _shifts(x: jax.Array, half: int) -> list[jax.Array]:
    """The columns of ``x`` shifted by ``-half..half``, edge columns repeated."""
    padded = jnp.pad(x, ((0, 0), (half, half)), mode="edge")
    n = x.shape[1]
    return [padded[:, half + j : half + j + n] for j in range(-half, half + 1)]


def _correlate(x: jax.Array, kernel: np.ndarray) -> jax.Array:
    """Apply a symmetric stencil, accumulating reflected samples together."""
    half = len(kernel) // 2
    shifted = _shifts(x, half)
    out = float(kernel[half]) * x
    for j in range(1, half + 1):
        k = float((kernel[half - j] + kernel[half + j]) * 0.5)
        if k:
            out = out + k * (shifted[half - j] + shifted[half + j])
    return out


def _median(x: jax.Array, footprint: np.ndarray) -> jax.Array:
    """The rank ``size // 2`` value over a 1-D footprint along columns."""
    taken = [s for keep, s in zip(footprint, _shifts(x, len(footprint) // 2), strict=True) if keep]
    return bitonic_median(taken)


def _weighted_quadratic(
    values: jax.Array, weights: jax.Array, window: int, guard: int
) -> jax.Array:
    """Each column's weighted least-squares quadratic through its neighbours, at the column.

    Offsets are scaled to [-1, 1] and neighbours taken relative to the column's own value, then
    the 3x3 normal equations are solved by Cholesky, so float32 keeps the local variation's digits.
    """
    values_shifted, weights_shifted = _shifts(values, window), _shifts(weights, window)
    s = [jnp.zeros_like(values) for _ in range(5)]
    t = [jnp.zeros_like(values) for _ in range(3)]
    # Pair the even/odd moments too: tiny asymmetric residuals otherwise steer
    # Tukey weights very differently in nearly flat, reflected neighbourhoods.
    for j in range(guard + 1, window + 1):
        u = j / window
        left, right = weights_shifted[window - j], weights_shifted[window + j]
        dl = left * (values_shifted[window - j] - values)
        dr = right * (values_shifted[window + j] - values)
        s = [s[k] + u**k * (right + left if k % 2 == 0 else right - left) for k in range(5)]
        t = [t[k] + u**k * (dr + dl if k % 2 == 0 else dr - dl) for k in range(3)]
    l00 = jnp.sqrt(s[0])
    l10, l20 = s[1] / l00, s[2] / l00
    l11 = jnp.sqrt(s[2] - l10 * l10)
    l21 = (s[3] - l20 * l10) / l11
    l22 = jnp.sqrt(s[4] - l20 * l20 - l21 * l21)
    y0 = t[0] / l00
    y1 = (t[1] - l10 * y0) / l11
    c2 = (t[2] - l20 * y0 - l21 * y1) / (l22 * l22)
    c1 = (y1 - l21 * c2) / l11
    return values + (y0 - l10 * c1 - l20 * c2) / l00


def _reference(ranked: jax.Array, gamma: float = 2.0, local: int = 64) -> jax.Array:
    """The adaptive spatial reference for rows of line integrals.

    It starts from the local quadratic through +-30 columns and moves toward the +-20, 14, 9 and 6
    ones only as far as each differs from it by more than ``gamma`` times its noise at that view
    and place, so roughness of the object's values (an edge turning at some radius) is
    followed rather than turned into a column correction. Neighbours are Tukey-weighted by their
    misfit to the +-6 quadratic so that a defective column does not steer them.
    """
    taps = [_quadratic_taps(w, 1) for w in WINDOWS]
    r = ranked - _correlate(ranked, taps[-1])
    sigma = 1.4826 * jnp.median(jnp.abs(r), axis=1, keepdims=True)
    u = r / (4.685 * jnp.maximum(sigma, 1e-3 * jnp.median(sigma) + 1e-12))
    weights = 1e-3 + jnp.where(jnp.abs(u) < 1, (1 - u * u) ** 2, 0.0)
    refs = [_weighted_quadratic(ranked, weights, w, 1) for w in WINDOWS]
    # the noise: local scatter about the narrowest fit (median of every 8th column within
    # +-local), carried through each pair of references with the columns' correlation
    r = ranked - refs[-1]
    shifts = _shifts(jnp.abs(r), local)
    scatter = bitonic_median([shifts[local + k] for k in range(-local, local + 1, 8)])
    scatter = 1.4826 * scatter / np.sqrt(1 + (taps[-1] ** 2).sum())
    r = r - r.mean(1, keepdims=True)
    var = jnp.maximum((r * r).mean(1, keepdims=True), jnp.finfo(r.dtype).tiny)
    rho = [jnp.clip((r[:, k:] * r[:, :-k]).mean(1, keepdims=True) / var, 0, 1) for k in (1, 2)]
    out = refs[0]
    for w, t, ref in zip(WINDOWS[1:], taps[1:], refs[1:], strict=True):
        d = np.pad(t, WINDOWS[0] - w) - taps[0]
        lags = [float((d[k:] * d[:-k]).sum()) for k in (1, 2)]
        noise = scatter * jnp.sqrt(float((d * d).sum()) + 2 * (rho[0] * lags[0] + rho[1] * lags[1]))
        diff = ref - out
        out = out + jnp.sign(diff) * jnp.maximum(jnp.abs(diff) - gamma * noise, 0.0)
    return out


def _fit(ranked: jax.Array, ref: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per column, the rank differences' offset, slope on ``1 - T`` and mean ``1 - T``.

    Centred weighted least squares with weights ``T`` against ``ref``.
    """
    x = -jnp.expm1(-ref)
    w = jnp.exp(-jnp.minimum(ref, 50.0))
    d = ranked - ref
    sw = w.sum(0)
    xm, dm = (w * x).sum(0) / sw, (w * d).sum(0) / sw
    xc = x - xm
    # a slope is measured only over the spread of 1 - T the column sees (a ridge for columns that
    # see one transmission), and a detector response stays monotone only for beta > -1
    beta = (w * xc * (d - dm)).sum(0) / ((w * xc * xc).sum(0) + 1e-4 * sw)
    beta = jnp.maximum(beta, -0.5)
    return dm - beta * xm, beta, xm


def _invert(q: jax.Array, alpha: jax.Array, beta: jax.Array) -> jax.Array:
    """The line integrals ``p`` with ``p + alpha + beta (1 - exp(-p)) = q``, by Newton."""
    p = q - alpha - beta * -jnp.expm1(-q)
    for _ in range(3):
        t = jnp.exp(-p)
        p = p - (p + alpha + beta * (1.0 - t) - q) / (1.0 + beta * t)
    return p


def _fit_unsorted(
    q: jax.Array, ref: jax.Array, neighbours: jax.Array, weights: jax.Array
) -> tuple[jax.Array, ...]:
    """Response contrasts and their influence, using independent neighbouring-view noise.

    Regressing on the noisy spatial reference itself correlates predictor and error.
    The angular interpolation is an instrument, including distinct endpoint views.
    """
    instrument = (ref[neighbours] * weights[:, :, None]).sum(1)
    x, z = -jnp.expm1(-ref), -jnp.expm1(-instrument)
    w = jnp.exp(-jnp.minimum(instrument, 50.0))
    d, sw = q - ref, w.sum(0)
    xm, zm, level = (w * x).sum(0) / sw, (w * z).sum(0) / sw, (w * d).sum(0) / sw
    xc, zc = x - xm, z - zm
    denominator = jnp.maximum((w * xc * zc).sum(0), 0) + 1e-4 * sw
    beta = jnp.maximum((w * zc * (d - level)).sum(0) / denominator, -0.5)
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
    factor restores the iid variance removed by centring each trajectory.
    """
    variance = jnp.zeros(influence.shape[1] // 2 + 1, influence.dtype)
    offset = 0
    for size in groups:
        part = influence[offset : offset + size]
        offset += size
        length = max(1, int(np.sqrt(size)))
        centered = part - part.mean(0, keepdims=True)
        integrated = jnp.cumsum(jnp.pad(centered, ((length, length - 1), (0, 0))), axis=0)
        windows = integrated[length:] - integrated[:-length]
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
) -> jax.Array:
    """Regularise a local response contrast by angular noise and spatial scene power.

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
    return jnp.fft.irfft(spectrum * gain, n=profile.size).astype(profile.dtype)


def _unsorted_row(
    q: jax.Array, neighbours: jax.Array, weights: jax.Array, groups: tuple[int, ...]
) -> jax.Array:
    out = _fill(q)
    taps = _quadratic_taps(6, 0)
    for _ in range(2):
        level, beta, xm, il, ib, ml, mb = _fit_unsorted(
            out, _correlate(out, taps), neighbours, weights
        )
        level = _response_posterior(level, il, ml, taps, groups)
        beta = jnp.maximum(_response_posterior(beta, ib, mb, taps, groups), -0.5)
        out = _invert(out, level - beta * xm, beta)
    return out


def _fill(q: jax.Array, window: int = 30) -> jax.Array:
    """Columns that do not follow the scene, blended into their neighbours' interpolation.

    Each column's sorted values are regressed on the median of its neighbours on either side at
    the same ranks; the better side counts, so a column beside a sample edge still follows its air
    side, and a column whose neighbours never vary is taken as working (slope 1).
    """
    ranked = jnp.sort(q, axis=0)
    # Measure variation relative to an observation before centring. Subtracting
    # rounded absolute means fabricates variance in view-constant columns.
    centered = ranked - ranked[:1]
    centered = centered - centered.mean(0)
    slopes = []
    for side in (-1, 1):
        foot = np.zeros(2 * window + 1, bool)
        foot[window + side * np.arange(2, window + 1)] = True
        rc = _median(ranked, foot)
        rc = rc - rc[:1]
        rc = rc - rc.mean(0)
        spread = (rc * rc).sum(0)
        prior = 1e-3 * jnp.median(spread) + jnp.finfo(q.dtype).tiny
        slopes.append(((rc * centered).sum(0) + prior) / (spread + prior))
    weight = jnp.clip(jnp.maximum(*slopes) / jnp.median(jnp.maximum(*slopes)), 0.0, 1.0)
    good = weight > 0.5
    n = q.shape[1]
    idx = jnp.arange(n)
    left = jax.lax.cummax(jnp.where(good, idx, -1))
    right = jax.lax.cummin(jnp.where(good, idx, n), reverse=True)
    lo = jnp.clip(jnp.where(left < 0, right, left), 0, n - 1)
    hi = jnp.clip(jnp.where(right >= n, left, right), 0, n - 1)
    frac = jnp.where(hi > lo, (idx - lo) / jnp.maximum(hi - lo, 1), 0.0).astype(q.dtype)
    filled = q[:, lo] + frac * (q[:, hi] - q[:, lo])
    usable = (~jnp.all(good)) & (jnp.sum(good) >= 2)
    return jnp.where(usable, weight * q + (1 - weight) * filled, q)


def _large_stripes(q: jax.Array, window: int = 100, ratio: float = 3.0) -> jax.Array:
    """Vo et al. 2018, algorithms 4 and 5, with our response model as the correction.

    Each column's level (mean of its sorted values, 5% of the views cut at either end) less its
    median over +-window columns is sorted, a line fitted through the middle half, and the values
    beyond the fitted ends by more than ``ratio``/2 spans flagged on each side whose extreme lies
    beyond them by ``ratio`` spans. Only the flagged columns, grown by two, lose their level
    against the +-window quadratic of their neighbours, faded out over eight more columns.
    """
    views, n = q.shape
    ranked = jnp.sort(q, axis=0)
    cut = int(0.05 * views)
    level = ranked[cut : views - cut].mean(0)
    padded = jnp.pad(level, window, mode="edge")
    stack = jnp.stack([padded[k : k + n] for k in range(2 * window + 1)])
    dev = level - jnp.sort(stack, axis=0)[window]
    s = jnp.sort(dev)
    i = np.arange(n, dtype=np.float64)[n // 4 : n - n // 4]
    y = s[n // 4 : n - n // 4]
    slope = ((i - i.mean()) * (y - y.mean())).sum() / float(((i - i.mean()) ** 2).sum())
    first = y.mean() - slope * i.mean()
    last = first + slope * (n - 1)
    span = jnp.maximum(last - first, 1e-12 * jnp.maximum(jnp.abs(s).max(), 1e-30))
    upper = ((s[-1] - last) / span > ratio) & (dev > last + 0.5 * ratio * span)
    lower = ((first - s[0]) / span > ratio) & (dev < first - 0.5 * ratio * span)
    grown = jnp.pad((upper | lower).astype(q.dtype), 10)
    mask = jnp.max(jnp.stack([grown[8 + j : 8 + j + n] for j in range(5)]), axis=0)
    wide = jnp.pad(mask, 8)
    weight = jnp.max(
        jnp.stack([wide[8 + j : 8 + j + n] * (1 - abs(j) / 9) for j in range(-8, 9)]), axis=0
    )
    alpha, beta, xm = _fit(ranked, _correlate(ranked, _quadratic_taps(window, 1)))
    return q - (weight * (alpha + beta * xm))[None, :]


def _row(
    q: jax.Array,
    axis: float,
    neighbours: jax.Array,
    weights: jax.Array,
    *,
    closing: bool,
) -> jax.Array:
    """Full-turn fits without sorting away the spatial smoothness at the axis."""
    out = _large_stripes(_fill(q))
    for _ in range(2):
        fitted = jnp.concatenate([0.5 * (out[:1] + out[-1:]), out[1:-1]]) if closing else out
        level, beta, xm, *_ = _fit_unsorted(fitted, _reference(fitted), neighbours, weights)
        level = regularize(level, axis, "significance")
        beta = regularize(beta, axis, "significance")
        out = _invert(out, level - beta * xm, beta)
    ranked = jnp.sort(out, axis=0)
    alpha, beta, xm = _fit(ranked, _correlate(ranked, _quadratic_taps(100, 1)))
    level = regularize(alpha + beta * xm, axis, "wiener")
    beta = regularize(beta, axis, "wiener")
    return _invert(out, level - beta * xm, beta)


@dataclass(frozen=True)
class _ViewPlan:
    order: tuple[int, ...]
    neighbours: tuple[tuple[int, int], ...]
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
    # the angle labels. A tie favours the smallest phase for deterministic order.
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
    """Angular neighbours within each fixed trajectory, never between cone segments."""
    if isinstance(geometry, ConeSegments):
        segments = geometry.segments
    elif isinstance(
        geometry, ParallelGeometry | ConeGeometry | LaminographyGeometry | RotationAxisGeometry
    ):
        segments = (geometry,)
    else:
        return None
    order, neighbours, weights, groups = [], [], [], []
    offset = 0
    closing = False
    for segment in segments:
        angles = np.asarray(segment.angles, np.float64)
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
            lo, hi = np.maximum(i - 1, 0), np.minimum(i + 1, size - 1)
            lo[0], hi[-1] = hi[0], lo[-1]
            gap = angles[hi] - angles[lo]
            weight = np.divide(angles[hi] - angles, gap, out=np.full(size, 0.5), where=gap != 0)
        neighbours.extend((int(a + offset), int(b + offset)) for a, b in zip(lo, hi, strict=True))
        weights.extend((float(w), float(1 - w)) for w in weight)
        groups.append(size)
        offset += size
    return _ViewPlan(tuple(order), tuple(neighbours), tuple(weights), tuple(groups), bool(closing))


@partial(jax.jit, static_argnames=("axis", "groups", "closing"))
def correct_rings(
    batch: jax.Array,
    axis: float | None = None,
    neighbours: jax.Array | None = None,
    weights: jax.Array | None = None,
    groups: tuple[int, ...] = (),
    *,
    closing: bool = False,
) -> jax.Array:
    """Correct every detector row of a ``(views, rows, columns)`` batch of line integrals."""
    if neighbours is None or weights is None:
        size = batch.shape[0]
        if size < 2:
            raise ValueError("Rings needs at least two views")
        i = np.arange(size)
        if axis is None:
            lo, hi = np.maximum(i - 1, 0), np.minimum(i + 1, size - 1)
            lo[0], hi[-1] = hi[0], lo[-1]
        else:
            lo, hi = (i - 1) % size, (i + 1) % size
        neighbours = jnp.asarray(np.stack([lo, hi], axis=1), jnp.int32)
        weights = jnp.full((size, 2), 0.5, batch.dtype)
        groups = (size,)
    if axis is not None:
        rows = jax.lax.map(
            lambda q: _row(q, axis, neighbours, weights, closing=closing), jnp.swapaxes(batch, 0, 1)
        )
        return jnp.swapaxes(rows, 0, 1)
    rows = jax.lax.map(
        lambda q: _unsorted_row(q, neighbours, weights, groups), jnp.swapaxes(batch, 0, 1)
    )
    return jnp.swapaxes(rows, 0, 1)


def mirror_axis(geometry: object) -> float | None:
    """The rotation axis's detector column on the half-pixel grid, or None.

    None unless ``geometry`` is a stationary parallel-beam scan over one evenly sampled full turn.
    """
    if type(geometry) is not ParallelGeometry:
        return None
    if _periodic_order(np.asarray(geometry.angles, np.float64)) is None:
        return None
    detector = geometry.detector
    column = detector.nu / 2.0 - 0.5 - float(detector.center[0]) / float(detector.du)
    return round(2.0 * column) / 2.0


@dataclass(frozen=True)
class Rings(Step):
    """Correct ring artefacts from each detector column's own response.

    A column whose gain or response the flat field does not describe reports
    ``q + a + b (1 - exp(-q))`` instead of the line integral ``q``: a stripe in
    the sinogram, a ring in the volume, stronger where the sample absorbs more
    when ``b`` is not zero. The fitted response is inverted in two passes;
    columns that do not follow the scene (stuck pixels) are first filled from
    their neighbours.

    On a stationary parallel-beam scan over one evenly sampled full turn,
    each view is compared with a local quadratic through 30 columns either
    side, narrowed toward 6 where the object changes faster. Independent
    neighbouring views provide the instrument for the response fit. Keeping
    these references unsorted preserves their spatial smoothness at the axis.
    The fine fits use cyclic angular neighbours and average a repeated closing
    phase. Large stripes are detected first using the sorted-column procedure
    of Vo, Atwood and Drakopoulos (2018, Opt. Express 26, 28396, algorithms 4
    and 5); a last sorted fit over 100 columns either side addresses broad rings.

    Opposed columns see the same lines over a full turn. Their fitted odd
    component therefore supplies a stationary detector-field spectrum; the
    even component is regularised locally in an undecimated multiscale frame,
    accounting for correlation between reflected neighbours. Fine corrections
    use per-scale simultaneous significance and broad corrections a Wiener gain.
    The variance and effective degrees are estimated, not a calibrated
    scan-wide confidence bound. DC power is inferred from the lowest measured
    odd-frequency band. Purely even detector errors and rotationally symmetric
    object structure remain ambiguous, especially at broad scales; this is
    not a no-change guarantee. The axis used, or none, is recorded.

    Otherwise, each unsorted view is compared with a quadratic through six
    neighbouring columns on either side. Independent neighbouring views supply
    an instrument for the response fit, avoiding correlation between reference
    noise and regression error. Angular covariance and the unfiltered scene's
    spatial power regularise the local response contrast before inversion.
    Smooth object structure is retained rather than interpreted as calibration
    error. This is a statistical separation, not a no-change guarantee: fine
    stationary object structure and broad detector errors can be ambiguous.

    Built-in angular geometries order the views by angle within each trajectory;
    single turns are unwrapped across the largest unsampled arc, explicit
    multi-turn angle labels are retained, and cone segments are kept separate.
    Without such geometry, views must follow a continuous acquisition trajectory.
    At least two views per trajectory are needed. Output view order is preserved.

    Parameters
    ----------
    geometry : object, optional
        The scan's geometry, with the detector the data were recorded on.
        Default is None.

    Raises
    ------
    ValueError
        If a geometry trajectory contains fewer than two views.

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
    # what the geometry decides, compared and hashed so compiled kernels are not shared
    _axis: float | None = field(default=None, init=False, repr=False)
    _columns: int | None = field(default=None, init=False, repr=False)
    _plan: _ViewPlan | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        axis = mirror_axis(self.geometry)
        object.__setattr__(self, "_axis", axis)
        if axis is not None and isinstance(self.geometry, ParallelGeometry):
            object.__setattr__(self, "_columns", int(self.geometry.detector.nu))
        object.__setattr__(self, "_plan", _view_plan(self.geometry, cyclic=axis is not None))

    @property
    def mirror_axis(self) -> float | None:
        """The rotation-axis column the mirror stages use.

        Returns
        -------
        float or None
            Detector column on the half-pixel grid, or None when the geometry is
            not a stationary parallel-beam full turn.
        """
        return self._axis

    def record(self) -> Correction:
        """Build this step's record: the mirror axis used, or None.

        Returns
        -------
        Correction
            Step name and settings, with no findings yet.
        """
        return Correction("rings", {"mirror_axis": self.mirror_axis})

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
            return correct_rings(batch, self._axis)
        plan = self._plan
        if len(plan.order) != batch.shape[0]:
            raise ValueError(
                f"Rings geometry has {len(plan.order)} views, the data {batch.shape[0]}"
            )
        order = jnp.asarray(plan.order, jnp.int32)
        out = correct_rings(
            batch[order],
            axis=self._axis,
            neighbours=jnp.asarray(plan.neighbours, jnp.int32),
            weights=jnp.asarray(plan.weights, batch.dtype),
            groups=plan.groups,
            closing=plan.closing,
        )
        return out[jnp.asarray(np.argsort(plan.order), jnp.int32)]
