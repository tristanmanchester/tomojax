"""Ring correction from each detector column's response, fitted on the view-sorted sinogram.

Each column ``c`` of a detector row is modelled as reporting ``q + a_c + b_c (1 - exp(-q))`` for
the true line integral ``q``: an offset and a transmission-dependent error. Sorting a column's
values over the views lines up equal ranks of neighbouring columns (Vo, Atwood and Drakopoulos
2018, Opt. Express 26, 28396, algorithm 3), so at every rank a column can be compared with a
reference built from its neighbours, and ``a_c, b_c`` fitted to the rank differences and inverted
exactly. Two passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.geometry import ParallelGeometry

from ._records import Correction
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
    out = jnp.zeros_like(x)
    for k, shifted in zip(kernel, _shifts(x, len(kernel) // 2), strict=True):
        if k != 0.0:
            out = out + float(k) * shifted
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
    u = np.arange(-window, window + 1, dtype=np.float64) / window
    keep = np.abs(u) > guard / window + 1e-12
    s = [jnp.zeros_like(values) for _ in range(5)]
    t = [jnp.zeros_like(values) for _ in range(3)]
    for uj, kj, sw, sv in zip(
        u, keep, _shifts(weights, window), _shifts(values, window), strict=True
    ):
        if kj:
            delta = sw * (sv - values)
            s = [s[m] + float(uj**m) * sw for m in range(5)]
            t = [t[m] + float(uj**m) * delta for m in range(3)]
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
    """The adaptive reference for view-sorted line integrals (ranks, columns).

    It starts from the local quadratic through +-30 columns and moves toward the +-20, 14, 9 and 6
    ones only as far as each differs from it by more than ``gamma`` times its noise at that rank
    and place, so roughness of the object's sorted values (an edge turning at some radius) is
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


def _fill(q: jax.Array, window: int = 30) -> jax.Array:
    """Columns that do not follow the scene, blended into their neighbours' interpolation.

    Each column's sorted values are regressed on the median of its neighbours on either side at
    the same ranks; the better side counts, so a column beside a sample edge still follows its air
    side, and a column whose neighbours never vary is taken as working (slope 1).
    """
    ranked = jnp.sort(q, axis=0)
    slopes = []
    for side in (-1, 1):
        foot = np.zeros(2 * window + 1, bool)
        foot[window + side * np.arange(2, window + 1)] = True
        rc = _median(ranked, foot)
        rc = rc - rc.mean(0)
        spread = (rc * rc).sum(0)
        prior = 1e-3 * jnp.median(spread) + jnp.finfo(q.dtype).tiny
        slopes.append(((rc * (ranked - ranked.mean(0))).sum(0) + prior) / (spread + prior))
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


def _mirror(profile: jax.Array, axis: float, keep_even: str, bands: int = 24) -> jax.Array:
    """Keep a per-column profile's odd part about ``axis`` and its even part band by band.

    The odd part is pure detector error on a parallel full turn. The even part is kept by the odd
    part's share of the power (``"wiener"``) or whole unless its power beats the odd part's by
    more than a 95% F test allows (``"significance"``).
    """
    from scipy.stats import f as fisher

    n = profile.shape[0]
    c = np.arange(n)
    u = np.flatnonzero((2 * axis - c >= 0) & (2 * axis - c <= n - 1) & (c >= axis))
    if u.size < 8:
        return profile
    m = (2 * axis - u).astype(int)
    even, odd = 0.5 * (profile[u] + profile[m]), 0.5 * (profile[u] - profile[m])
    fe, fo = jnp.fft.rfft(even), jnp.fft.rfft(odd)
    edges = np.unique(np.geomspace(1, fe.shape[0], bands).astype(int))
    gains = [jnp.ones(fe.shape[0] - int(edges[-1]), profile.dtype)]
    for lo, hi in reversed(list(zip(np.r_[0, edges[:-1]], edges, strict=True))):
        pe = jnp.maximum(jnp.sum(jnp.abs(fe[lo:hi]) ** 2), jnp.finfo(profile.dtype).tiny)
        scale = (
            1.0 if keep_even == "wiener" else float(fisher.ppf(0.95, 2 * (hi - lo), 2 * (hi - lo)))
        )
        gain = jnp.minimum(scale * jnp.sum(jnp.abs(fo[lo:hi]) ** 2) / pe, 1.0)
        gains.insert(0, jnp.full(hi - lo, gain, profile.dtype))
    kept = jnp.fft.irfft(fe * jnp.concatenate(gains), n=even.shape[0]).astype(profile.dtype)
    return profile.at[u].set(kept + odd).at[m].set(kept - odd)


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


def _row(q: jax.Array, axis: float | None, passes: int = 2, window: int = 100) -> jax.Array:
    out = _fill(q)
    if axis is not None:
        out = _large_stripes(out, window)
    for _ in range(passes):
        ranked = jnp.sort(out, axis=0)
        alpha, beta, xm = _fit(ranked, _reference(ranked))
        if axis is not None:
            level = _mirror(alpha + beta * xm, axis, "significance")
            beta = _mirror(beta, axis, "significance")
            alpha = level - beta * xm
        out = _invert(out, alpha[None, :], beta[None, :])
    if axis is None:
        return out
    ranked = jnp.sort(out, axis=0)
    alpha, beta, xm = _fit(ranked, _correlate(ranked, _quadratic_taps(window, 1)))
    level = _mirror(alpha + beta * xm, axis, "wiener")
    beta = _mirror(beta, axis, "wiener")
    return _invert(out, (level - beta * xm)[None, :], beta[None, :])


@partial(jax.jit, static_argnames=("axis",))
def correct_rings(batch: jax.Array, axis: float | None = None) -> jax.Array:
    """Correct every detector row of a ``(views, rows, columns)`` batch of line integrals."""
    rows = jax.lax.map(lambda q: _row(q, axis), jnp.swapaxes(batch, 0, 1))
    return jnp.swapaxes(rows, 0, 1)


def mirror_axis(geometry: object) -> float | None:
    """The rotation axis's detector column on the half-pixel grid, or None.

    None unless ``geometry`` is a stationary parallel-beam scan over one evenly sampled full turn.
    """
    if type(geometry) is not ParallelGeometry:
        return None
    angles = np.sort(np.asarray(geometry.angles, np.float64))
    steps = np.diff(angles)
    if angles.size < 8 or not np.all(np.abs(steps - steps.mean()) <= 0.1 * steps.mean()):
        return None
    # one turn, with or without a last view repeating the first angle
    cover = angles[-1] - angles[0] + steps.mean()
    if not 360.0 - 0.5 * steps.mean() <= cover <= 360.0 + 1.5 * steps.mean():
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
    when ``b`` is not zero. Each detector row's values are sorted over views
    (Vo, Atwood and Drakopoulos 2018, Opt. Express 26, 28396, algorithm 3), and
    at every rank a column is compared with a reference built from its
    neighbours: the local quadratic through 30 columns either side, narrowed
    toward 6 where the sample's own sorted values change faster than that, so
    that sample edges are not mistaken for stripes. ``a`` and ``b`` are fitted
    to the differences and inverted exactly, in two passes; columns that do not
    follow the scene (stuck pixels) are first filled from their neighbours.

    Given the scan's ``geometry``, the step checks whether it is a stationary
    parallel-beam scan over one evenly sampled full turn. Every column and its
    mirror about the rotation axis then see the same lines, so the odd part of
    the fitted errors is pure detector error: large stripes are detected and
    corrected first (algorithms 4 and 5 of the same paper, on the flagged
    columns only), the fine fit keeps the errors' even part only as far as the
    odd part supports it, and a last pass over 100 columns either side removes
    broad rings the same way. An even error profile on its own is
    indistinguishable from a rotationally symmetric sample, so what is kept of
    it rests on the odd part showing detector errors at that scale. The axis
    used, or none, is recorded.

    Otherwise (no geometry, half turns, cone beam), a defect wider than about
    twelve columns is followed like sample structure and left, and on a scan
    without rings the correction still changes the image slightly, most where
    the sample has sharp edges: use it when rings are present.

    Parameters
    ----------
    geometry : object, optional
        The scan's geometry, with the detector the data were recorded on.
        Default is None.

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

    def __post_init__(self) -> None:
        axis = mirror_axis(self.geometry)
        object.__setattr__(self, "_axis", axis)
        if axis is not None and isinstance(self.geometry, ParallelGeometry):
            object.__setattr__(self, "_columns", int(self.geometry.detector.nu))

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
            If the geometry's detector is not as wide as the data.
        """
        if self._columns is not None and self._columns != batch.shape[2]:
            raise ValueError(
                f"Rings geometry has {self._columns} detector columns, the data {batch.shape[2]}"
            )
        return correct_rings(batch, self._axis)
