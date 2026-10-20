"""Polychromatic reconstruction: the image and its beam hardening curve fitted together.

The measured line integrals of one material are ``q = g(A x)``, with ``g`` the
single-material curve of :mod:`tomojax.core.hardening`: a positive mixture of
exponentials, here with its spectrum of rates a mixture of a few log-normal
components, each taken at its Gauss-Hermite nodes, scaled so that ``g'(0) = 1``.
A component can be as narrow as one rate (no hardening) or as broad as an
unfiltered tube's spectrum.
:func:`polychromatic` minimises the weighted least-squares misfit

    ``0.5 sum_i v_i (g(A x)_i - q_i)^2``,  ``v_i = exp(-q_i)``,

each ray weighted by its Poisson precision up to the unknown dose, over the
image ``x`` and the spectrum ``beta`` together, by Gauss-Newton. Each step
linearises ``g`` about ``u = A x`` and solves the linear least-squares problem
in both unknowns by conjugate gradients on the joint operator
``[v^1/2 g'(u) A, v^1/2 dg/dbeta]``. The image block is the exact matched
projector and its transpose applied to pointwise weights; the spectrum block is
pointwise and costs no projection.

Starting from the identity curve the spectrum's gradient vanishes, so the
fit starts from a first estimate: the segment-reproject self-calibration
(:func:`segmented_curve`) unless a curve is given.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from itertools import pairwise
import logging
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core import hardening as mixture

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomojax.geometry import Detector, Geometry, Grid

LOG = logging.getLogger(__name__)

# Each log-normal component of the spectrum is taken at these nodes and weights
# of the standard normal.
_NODES, _NODE_WEIGHTS = np.polynomial.hermite_e.hermegauss(9)
_NODE_WEIGHTS = _NODE_WEIGHTS / _NODE_WEIGHTS.sum()
# Log-normal components of the fitted spectrum.
_TERMS = 3
# Levels of the self-calibration: at most _LEVELS, each attenuating more than
# _DENSER times the next lighter and holding at least _SMALLEST_LEVEL of the voxels.
_LEVELS = 5
_DENSER = 1.5
_SMALLEST_LEVEL = 0.01
# A level is a material, not the partial volumes along an edge, if this share
# of its voxels has all its neighbours in it.
_INTERIOR = 0.25
# Rays fitting the curve cross other levels for at most this share of their material.
_MIXED = 0.02
# Rays fitting the curve cross at least this many voxels of material.
_GRAZING = 3.0
# A voxel is in the field of view if it takes this share of the largest
# backprojection of ones.
_SEEN = 0.9
_FIELD_VOXELS = 1 << 18
# Histogram bins per standard deviation of the image's noise, and the most bins.
_BINS_PER_NOISE = 2
_MOST_BINS = 1 << 14
# Voxels the histogram samples, and the smallest step a Gauss-Newton update halves to.
_SAMPLES = 4_000_000
_SMALLEST_STEP = 1e-3
# Levenberg-Marquardt damping of the spectrum, relative to its best-seen parameter.
_DAMPING = 1e-4


@dataclass(frozen=True, kw_only=True)
class PolychromaticConfig:
    """Settings of the joint image and beam hardening fit.

    ``iterations`` Gauss-Newton steps each run ``inner_iterations`` conjugate
    gradient updates of the image and the spectrum together, then step to the
    linearised solution, halving the step until the misfit falls. ``rates`` and
    ``weights`` give the starting curve (as in
    :class:`tomojax.corrections.BeamHardening`); None starts from the
    segment-reproject self-calibration of ``BeamHardening.fit``, which takes
    ``threshold``.
    The image is unconstrained least squares, so like CGLS it fits
    noise as the total count of updates grows.

    Parameters
    ----------
    iterations : int, optional
        Gauss-Newton steps. Default is 8.
    inner_iterations : int, optional
        Conjugate-gradient updates per step. Default is 8.
    rates : tuple of float or None, optional
        Rates of the starting curve; None self-calibrates.
    weights : tuple of float or None, optional
        Weights of the starting curve, with ``rates``.
    threshold : float or None, optional
        Attenuation separating material from background in the
        self-calibration; None finds it in the histogram.
    """

    iterations: int = 8
    inner_iterations: int = 8
    rates: tuple[float, ...] | None = None
    weights: tuple[float, ...] | None = None
    threshold: float | None = None

    def __post_init__(self) -> None:
        if self.iterations < 0 or self.inner_iterations < 1:
            raise ValueError("PolychromaticConfig needs iterations >= 0 and inner_iterations >= 1")
        if (self.rates is None) != (self.weights is None):
            raise ValueError("PolychromaticConfig takes rates and weights together")


def _spectrum(params: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Log weights and rates (unit initial slope) of a log-normal mixture spectrum.

    ``params`` holds, for each of ``m`` components, its log share (up to a
    common constant), log median rate and log width, in three blocks of ``m``.
    """
    m = params.shape[0] // 3
    share = params[:m] - jax.scipy.special.logsumexp(params[:m])
    nodes = jnp.asarray(_NODES, params.dtype)
    log_k = (params[m : 2 * m, None] + jnp.exp(params[2 * m :, None]) * nodes[None, :]).ravel()
    log_w = (share[:, None] + jnp.log(jnp.asarray(_NODE_WEIGHTS, params.dtype))[None, :]).ravel()
    log_mean = jax.scipy.special.logsumexp(log_w + log_k)
    return log_w, jnp.exp(log_k - log_mean)


def _curve(u: jax.Array, params: jax.Array) -> jax.Array:
    """``g(u)`` of the spectrum ``params``, in the unit-initial-slope gauge."""
    log_w, rates = _spectrum(params)
    return -jax.scipy.special.logsumexp(log_w - u[..., None] * rates, axis=-1)


def spectrum_weights(params: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Rates and weights (unit initial slope) of the spectrum ``params``, in float64."""
    params = np.asarray(params, np.float64)
    m = params.size // 3
    share = np.exp(params[:m] - params[:m].max())
    rates = np.exp(params[m : 2 * m, None] + np.exp(params[2 * m :, None]) * _NODES[None, :])
    weights = share[:, None] * _NODE_WEIGHTS[None, :]
    return mixture.normalise(tuple(rates.ravel()), tuple(weights.ravel()), unit_slope=True)


def _project_spectrum(
    rates: tuple[float, ...], weights: tuple[float, ...], terms: int, upto: float
) -> np.ndarray:
    """The log-normal mixture whose curve is closest to a given one on ``[0, upto]``."""
    from scipy.optimize import least_squares

    k, w = mixture.normalise(rates, weights, unit_slope=True)
    p = np.linspace(0.0, upto, 129)
    target = mixture.curve(k, w, p)
    log_k = np.log(np.asarray(k))
    mean = float(np.dot(w, log_k))
    spread = float(np.sqrt(max(np.dot(w, (log_k - mean) ** 2), 1e-4)))
    # Components spread across the given spectrum, each about as wide as it.
    offsets = np.linspace(-1.0, 1.0, terms) if terms > 1 else np.zeros(1)
    start = np.concatenate(
        [np.zeros(terms), mean + spread * offsets, np.full(terms, np.log(spread / terms))]
    )

    def residual(params: np.ndarray) -> np.ndarray:
        kk, ww = spectrum_weights(params)
        return mixture.curve(kk, ww, p) - target

    low = np.concatenate([np.full(terms, -30.0), np.full(terms, -8.0), np.full(terms, -8.0)])
    high = np.concatenate([np.full(terms, 30.0), np.full(terms, 8.0), np.full(terms, 1.5)])
    start = np.clip(start, low + 1e-6, high - 1e-6)
    return least_squares(residual, start, bounds=(low, high), max_nfev=100 * start.size).x


def segmented_curve(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: np.ndarray,
    *,
    threshold: float | None = None,
    iterations: int = 2,
) -> tuple[tuple[float, ...], tuple[float, ...], dict[str, object]]:
    """Self-calibrate a curve by segmenting a reconstruction and reprojecting it.

    Each pass reconstructs by FBP (FDK for cone beams) the data corrected by
    the last pass's curve and divides it into a background and material levels
    (:func:`_levels`), each material flattened to its mean, so that shading
    within a material is not taken for its density. The exact projector turns
    this into each ray's material line integral, in units of one level's
    attenuation, and the curve is fitted to the measured line integrals
    against those (:func:`tomojax.core.hardening.fit`). Levels are found
    among the voxels every view sees; the others count as their own value.
    The curve is the most populous level's: only rays that cross it alone
    (other levels and voxels outside the field make at most ``_MIXED`` of
    theirs) for more than ``_GRAZING`` voxels enter the fit, and other
    materials, which share its spectral shape only approximately, are
    corrected as denser or lighter versions of it. Returns rates, weights
    (unit initial slope) and the last pass's levels and rms misfit.
    """
    from ._operators import project
    from .api import method_config, reconstruct_arrays

    if iterations < 1:
        raise ValueError("the self-calibration needs at least one iteration")
    measured = np.asarray(projections, np.float32)
    seen = _field(geometry, grid, measured.shape)
    rates: tuple[float, ...] = (1.0,)
    weights: tuple[float, ...] = (1.0,)
    found: dict[str, object] = {}
    for _ in range(iterations):
        data = measured if rates == (1.0,) else _inverted(rates, weights, measured)
        volume, _ = reconstruct_arrays(
            "fbp", geometry, grid, detector, data, config=method_config("fbp")
        )
        image = np.asarray(volume)
        level, means, unit = _levels(image, threshold, seen)
        # Material levels are flattened; a voxel outside the field counts as its
        # own value, and any ray crossing much of it is left out below.
        flat = np.asarray([0.0, *means])[np.maximum(level, 0)]
        occupancy = np.where(level < 0, image / unit, flat).astype(np.float32)
        length = np.asarray(project(geometry, occupancy, grid=grid))
        shortest = _GRAZING * min(grid.vx, grid.vy, grid.vz)
        # The curve is the most common material's: rays mostly through other
        # levels or voxels outside the field would make the density-scaled
        # materials' different spectra part of it, and rays grazing an edge,
        # where detector blur mixes in their neighbours' attenuation, would
        # misplace its start.
        part = length
        if len(means) > 1 or (level < 0).any():
            common = 1 + int(np.argmax(np.bincount(level[level > 0])[1:]))
            main = np.where(level == common, occupancy, 0.0).astype(np.float32)
            part = np.asarray(project(geometry, main, grid=grid))
        alone = (part > shortest) & (part >= (1 - _MIXED) * length)
        if not alone.any():
            raise ValueError("no ray crosses the most common material alone")
        rates, weights, fit = mixture.fit(length[alone], measured[alone])
        rates, weights = mixture.normalise(rates, weights, unit_slope=True)
        found = {"levels": means, "rms": fit["rms"]}
        LOG.info("segmented beam hardening fit: %s", found)
    return rates, weights, found


def _field(geometry: Geometry, grid: Grid, shape: tuple[int, ...]) -> np.ndarray:
    """The voxels every view sees: a corner outside the scanned field has values of its own.

    A voxel is in the field when the backprojection of ones reaches ``_SEEN``
    of its largest value. The backprojection is computed on a grid of at most
    ``_FIELD_VOXELS`` covering the same box and interpolated to the voxels.
    """
    from scipy.ndimage import zoom

    from ._operators import backproject

    factor = max(1, int(np.ceil((grid.nx * grid.ny * grid.nz / _FIELD_VOXELS) ** (1 / 3))))
    counts = (max(1, grid.nx // factor), max(1, grid.ny // factor), max(1, grid.nz // factor))
    reach = np.asarray(
        backproject(geometry, np.ones(shape, np.float32), grid=grid.resampled(counts))
    )
    scale = [n / c for n, c in zip((grid.nx, grid.ny, grid.nz), counts, strict=True)]
    reach = zoom(reach, scale, order=1, mode="nearest", grid_mode=True)
    return reach >= _SEEN * reach.max()


def _levels(
    image: np.ndarray, threshold: float | None, seen: np.ndarray
) -> tuple[np.ndarray, list[float], float]:
    """The background (level 0) and material levels of a reconstruction, voxel by voxel.

    The voxel values fall into levels, each a peak of their histogram, binned
    at the image's noise (:func:`_valley`): a level is split at a valley
    between two peaks that stand at least twice as high as it, when the denser
    peak's voxels attenuate more than ``_DENSER`` times the lighter's. Shading
    within one material and partial volumes at edges make no peak of their
    own, so they stay in a level; air and a material, or two materials, do
    not. A level must also have an interior (``_INTERIOR``), which a ring of
    edge voxels lacks. The lowest level is the background: below
    ``threshold`` when one is given, else below the lowest such valley.
    Voxels outside ``seen``, and those of a phase too rare to be a level (a
    thin metal rod), take level -1. Returns each voxel's level, the material
    levels' mean attenuations relative to the most populous one's, and that one's.
    """
    smooth = np.nan_to_num(image)
    finite = smooth[np.isfinite(image) & seen]
    values = finite[:: max(1, finite.size // _SAMPLES)]
    width = _noise(smooth, seen) / _BINS_PER_NOISE
    bounds = [float(values.min()) - 1.0, float(values.max()) + 1.0]
    # Lower bounds of the ranges of rare phases.
    rare: set[float] = set()
    pool, background = values, threshold
    while background is None:
        found = _valley(pool, values.size, width)
        if found is None:
            raise ValueError(
                "no material stands out from the background in the reconstruction; "
                "give the attenuation that separates them as threshold="
            )
        if found[1]:
            rare.add(found[0])
            bounds.insert(1, found[0])
            pool = pool[pool <= found[0]]
        else:
            background = found[0]
    bounds = sorted([*bounds, float(background)])
    settled: set[tuple[float, float]] = set()
    while len(bounds) <= _LEVELS:
        first, found = (1 if threshold is not None else 0), None
        for low, high in list(pairwise(bounds))[first:]:
            if (low, high) in settled or low in rare:
                continue
            found = _valley(values[(values > low) & (values <= high)], values.size, width)
            if found is not None and (
                found[1]
                or min(
                    _interior((smooth > low) & (smooth <= found[0]) & seen),
                    _interior((smooth > found[0]) & (smooth <= high) & seen),
                )
                >= _INTERIOR
            ):
                break
            settled.add((low, high))
            found = None
        if found is None:
            break
        if found[1]:
            rare.add(found[0])
        bounds = sorted([*bounds, found[0]])
    kept = np.asarray([low not in rare for low in bounds[:-1]])
    renumber = np.where(kept, np.cumsum(kept) - 1, -1)
    level = np.where(seen, renumber[np.digitize(smooth, bounds[1:-1])], -1)
    inside = level >= 0
    counts = np.bincount(level[inside], minlength=int(kept.sum()))
    means = np.bincount(level[inside], np.nan_to_num(image)[inside], minlength=int(kept.sum()))
    means = means / np.maximum(counts, 1)
    reference = means[1 + np.argmax(counts[1:])]
    return level, [float(m / reference) for m in means[1:]], float(reference)


def _noise(image: np.ndarray, seen: np.ndarray) -> float:
    """The standard deviation of the image's noise, from differences of neighbouring voxels.

    Edges make few of the differences, so their median absolute value is the noise's.
    """
    step = max(1, image.shape[2] // 16)
    planes, inside = image[:, :, ::step], seen[:, :, ::step]
    difference = np.diff(planes, axis=0)[inside[1:] & inside[:-1]]
    return 1.4826 * float(np.median(np.abs(difference))) / np.sqrt(2)


def _interior(mask: np.ndarray) -> float:
    """The share of ``mask``'s voxels whose neighbours along every long axis are in it too."""
    return float(_interior_mask(mask).sum()) / max(int(mask.sum()), 1)


def _interior_mask(mask: np.ndarray) -> np.ndarray:
    """``mask``'s voxels whose neighbours along every long axis are in it too."""
    inside = mask.copy()
    for axis in range(mask.ndim):
        if mask.shape[axis] > 1:
            inside[(slice(None),) * axis + (slice(1, None),)] &= mask[
                (slice(None),) * axis + (slice(None, -1),)
            ]
            inside[(slice(None),) * axis + (slice(None, -1),)] &= mask[
                (slice(None),) * axis + (slice(1, None),)
            ]
    return inside


def _valley(values: np.ndarray, total: int, width: float) -> tuple[float, bool] | None:
    """The lowest split of ``values`` between two distinct peaks of their histogram, if any.

    The histogram has bins of ``width``, smoothed over ``_BINS_PER_NOISE``
    of them. A valley is a run of bins under half the highest bin on either
    side, split at its lowest bin; the voxels above the split must attenuate
    more than ``_DENSER`` times those below. Each side must hold
    ``_SMALLEST_LEVEL`` of the ``total`` voxels, except that a denser side
    holding fewer is a rare phase, returned with True.
    """
    from scipy.ndimage import gaussian_filter1d

    if values.size < _SMALLEST_LEVEL * total:
        return None
    span = (float(values.min()), float(values.max()))
    bins = int(np.clip(np.ceil((span[1] - span[0]) / max(width, 1e-30)), 2, _MOST_BINS))
    counts, edges = np.histogram(values, bins, span)
    moments, _ = np.histogram(values, bins, span, weights=values.astype(np.float64))
    smooth = gaussian_filter1d(counts.astype(np.float64), _BINS_PER_NOISE)
    rise = np.maximum.accumulate(smooth)
    fall = np.maximum.accumulate(smooth[::-1])[::-1]
    deep = np.minimum(rise, fall) >= 2 * smooth
    below, moment = np.cumsum(counts), np.cumsum(moments)
    runs = np.flatnonzero(np.diff(np.r_[0, deep.astype(np.int8), 0])).reshape(-1, 2)
    for start, stop in runs:
        bottom = int(start + np.argmin(smooth[start:stop]))
        light, dense = int(below[bottom]), values.size - int(below[bottom])
        if light < _SMALLEST_LEVEL * total or dense == 0:
            continue
        light_mean = moment[bottom] / light
        if light_mean > 0 and (moment[-1] - moment[bottom]) / dense <= _DENSER * light_mean:
            continue
        return float(edges[bottom + 1]), dense < _SMALLEST_LEVEL * total
    return None


def _inverted(
    rates: tuple[float, ...], weights: tuple[float, ...], measured: np.ndarray
) -> np.ndarray:
    out = np.empty_like(measured)
    for first in range(0, len(measured), 64):
        batch = jnp.asarray(measured[first : first + 64])
        out[first : first + 64] = np.asarray(mixture.invert(rates, weights, batch))
    return out


@partial(jax.jit, static_argnames=("forward", "adjoint", "inner"))
def _gauss_newton_step(
    volume: jax.Array,
    beta: jax.Array,
    projected: jax.Array,
    measured: jax.Array,
    *,
    forward: Callable[[jax.Array], jax.Array],
    adjoint: Callable[[jax.Array], jax.Array],
    inner: int,
) -> tuple[jax.Array, jax.Array]:
    """The linearised least-squares step in (image, spectrum), by CGLS."""
    root = jnp.sqrt(jnp.clip(jnp.exp(-measured), 1e-6, 1.0))
    curve = _curve
    value, slope = jax.jvp(lambda u: curve(u, beta), (projected,), (jnp.ones_like(projected),))
    _, pullback = jax.vjp(lambda b: curve(projected, b), beta)

    def spectrum_columns(direction: jax.Array) -> jax.Array:
        return jax.jvp(lambda b: curve(projected, b), (beta,), (direction,))[1]

    # Each spectrum column scaled to unit norm, with Levenberg-Marquardt damping
    # in the spectrum's own units so that a parameter the data hardly see (the
    # width of a component as narrow as one rate) cannot take a long step.
    eye = jnp.eye(beta.shape[0], dtype=beta.dtype)
    norms = jax.vmap(lambda e: jnp.sqrt(jnp.sum((root * spectrum_columns(e)) ** 2)))(eye)
    scale = 1.0 / jnp.maximum(norms, 1e-12)
    damping = jnp.sqrt(_DAMPING) * jnp.max(norms) * scale

    def apply(dx: jax.Array, db: jax.Array) -> tuple[jax.Array, jax.Array]:
        return root * (slope * forward(dx) + spectrum_columns(db * scale)), damping * db

    def transpose(s: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        weighted = root * s[0]
        return adjoint(slope * weighted), pullback(weighted)[0] * scale + damping * s[1]

    def dot(a: tuple[jax.Array, jax.Array], b: tuple[jax.Array, jax.Array]) -> jax.Array:
        return jnp.vdot(a[0], b[0]) + jnp.vdot(a[1], b[1])

    residual = (root * (measured - value), jnp.zeros_like(beta))
    sx, sb = transpose(residual)

    def body(_: int, state: tuple) -> tuple:
        dx, db, res, px, pb, gamma = state
        mp = apply(px, pb)
        alpha = gamma / jnp.maximum(dot(mp, mp), 1e-30)
        dx, db = dx + alpha * px, db + alpha * pb
        res = (res[0] - alpha * mp[0], res[1] - alpha * mp[1])
        sx, sb = transpose(res)
        new = jnp.vdot(sx, sx) + jnp.vdot(sb, sb)
        ratio = new / jnp.maximum(gamma, 1e-30)
        return dx, db, res, sx + ratio * px, sb + ratio * pb, new

    gamma = jnp.vdot(sx, sx) + jnp.vdot(sb, sb)
    state = (jnp.zeros_like(volume), jnp.zeros_like(beta), residual, sx, sb, gamma)
    dx, db, *_ = jax.lax.fori_loop(0, inner, body, state)
    return dx, db * scale


@partial(jax.jit, static_argnames=("forward",))
def _misfit(
    volume: jax.Array,
    beta: jax.Array,
    measured: jax.Array,
    *,
    forward: Callable[[jax.Array], jax.Array],
) -> tuple[jax.Array, jax.Array]:
    projected = forward(volume)
    weight = jnp.clip(jnp.exp(-measured), 1e-6, 1.0)
    residual = _curve(projected, beta) - measured
    return 0.5 * jnp.sum(weight * residual * residual), projected


def polychromatic(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jax.Array | np.ndarray,
    *,
    config: PolychromaticConfig | None = None,
) -> tuple[jax.Array, dict[str, object]]:
    """Fit an image and its single-material beam hardening curve to line integrals.

    See :class:`PolychromaticConfig`. The record holds the fitted curve
    (``beam_hardening``: rates and weights in the unit-initial-slope gauge, as
    :class:`tomojax.corrections.BeamHardening` takes them), the misfit after
    each step and the number of rays.

    Parameters
    ----------
    geometry : Geometry
        Scan geometry with its views.
    grid : Grid
        Reconstruction grid.
    detector : Detector
        Detector of the projections.
    projections : array-like
        Measured line integrals ``-ln(I / I0)``, shape ``(views, nv, nu)``.
    config : PolychromaticConfig or None, optional
        Settings; None uses the defaults.

    Returns
    -------
    volume : jax.Array
        The image, in the unit of the unhardened beam's mean attenuation.
    record : dict
        Fitted curve and misfit history.
    """
    from ._operators import matched_operators
    from .api import method_config, reconstruct_arrays

    cfg = PolychromaticConfig() if config is None else config
    measured_host = np.asarray(projections, np.float32)
    if cfg.rates is None:
        rates, weights, found = segmented_curve(
            geometry, grid, detector, measured_host, threshold=cfg.threshold
        )
    else:
        rates, weights = mixture.normalise(cfg.rates, tuple(cfg.weights or ()), unit_slope=True)
        found = {}
    upto = float(mixture.inverse(rates, weights, np.asarray(np.percentile(measured_host, 99.9))))
    beta = jnp.asarray(_project_spectrum(rates, weights, _TERMS, upto), jnp.float32)
    start_rates, start_weights = spectrum_weights(np.asarray(beta, np.float64))
    corrected = _inverted(start_rates, start_weights, measured_host)
    volume, _ = reconstruct_arrays(
        "fbp", geometry, grid, detector, corrected, config=method_config("fbp")
    )
    del corrected
    _, _, (forward, adjoint) = matched_operators(geometry, grid, "polychromatic")
    measured = jnp.asarray(measured_host)
    volume = jnp.asarray(volume, jnp.float32)
    value, projected = _misfit(volume, beta, measured, forward=forward)
    history = [float(value)]
    for _ in range(cfg.iterations):
        dx, db = _gauss_newton_step(
            volume, beta, projected, measured,
            forward=forward, adjoint=adjoint, inner=cfg.inner_iterations,
        )  # fmt: skip
        step = 1.0
        while True:
            trial, trial_projected = _misfit(
                volume + step * dx, beta + step * db, measured, forward=forward
            )
            if float(trial) < history[-1] or step < _SMALLEST_STEP:
                break
            step *= 0.5
        if not float(trial) < history[-1]:
            break
        volume, beta, projected = volume + step * dx, beta + step * db, trial_projected
        history.append(float(trial))
    rates, weights = spectrum_weights(np.asarray(beta, np.float64))
    record: dict[str, object] = {
        "beam_hardening": {"rates": list(rates), "weights": list(weights)},
        "misfit": history,
        "rays": int(measured.size),
        "start": found,
    }
    return volume, record
