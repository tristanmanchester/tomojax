"""Each view's weights of its two bracketing flat sets, measured from the views.

Two flat sets around a run of views differ either because the illumination changed
between them or because one of them does not show the illumination the views had
(something stood in the beam while it was taken). The acquisition times cannot tell
these apart; the views can. With ``L = log(later) - log(earlier)`` and a view whose
illumination is ``earlier * exp(w L)``, ``log(view) - log(earlier) = w L - p`` for the
line integrals ``p``. The fine structure of ``L`` along detector columns is unrelated
to the object, so regressing the view on it over a neighborhood measures ``w`` there.

A sample of views gives ``w`` through the interval in each detector cell, fitted as a
line in the views' positions between the sets. The flat sets' own noise is in ``L``
and in no view: its expected products are subtracted, so a difference that is only
noise or smooth across columns carries no information and the weights stay those of
linear interpolation in position.

The weights describe structure, not level. Where they change along a row while the
sets differ, blending the sets would put a step in the flat that neither the views
nor the weights support (the far side of an edge that only one set has). A level
term removes that step: it keeps the blended flat where a set matches the views or
the views say nothing, and lets the level carry across where neither set matches.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, replace
from functools import partial
import logging
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax import progress

from ._batches import correction_group, view_batches

if TYPE_CHECKING:
    from ._fields import CalibrationFrames, Fields
    from ._records import Json
    from ._steps import Step

LOG = logging.getLogger(__name__)

# Detector pixels along each side of a weight cell; a weight is fitted to the cells around it.
_CELL = 16
# Columns of the running mean that the high-pass along a detector row removes.
_WIDTH = 15
# Cells each way of the window whose weights set the flat's level.
_BROAD = 4
# A weight known only to lie between 0 and 1 has variance 1/12.
_PRIOR_PRECISION = 12.0
# Linear interpolation in position: none of the later set at the earlier one, all of it at itself.
_LINEAR = np.array([0.0, 1.0]).reshape(2, 1, 1)
# Median of a chi-square variable with one degree of freedom.
_CHI2_MEDIAN = 0.4549
_VIEW_BYTES = 32 * 2**20


@dataclass(frozen=True)
class FlatWeights:
    """What the views show of the later flat set at both ends of each interval.

    ``ends`` has shape ``(sets - 1, 2, cells_v, cells_u)``: row ``i`` holds the
    weights of set ``i + 1`` at set ``i`` and at set ``i + 1``, on cells of
    ``_CELL`` detector pixels. ``levels``, ``(sets - 1, 2, cells_v, columns)``, is
    the log factor on the blended flat at the same two ends. ``rows`` is the
    window of detector rows these are read through.
    """

    ends: np.ndarray
    levels: np.ndarray
    shape: tuple[int, int]
    rows: tuple[int, int]

    def window(self, rows: slice) -> FlatWeights:
        """The same weights seen through ``rows`` of the current window."""
        first, stop, _ = rows.indices(self.rows[1] - self.rows[0])
        return replace(self, rows=(self.rows[0] + first, self.rows[0] + stop))

    def at(self, rows: slice, earlier: np.ndarray | None = None) -> np.ndarray:
        """Both ends' weights, then levels, at every pixel of ``rows``.

        ``earlier`` are the earlier sets of the intervals wanted, by default every
        interval. One interval gives ``(4, rows, columns)``; several give
        ``(intervals, 4, rows, columns)`` in the order asked. Values are constant
        across the columns of a cell and linear between the cells' middle rows.
        """
        first, stop, _ = rows.indices(self.rows[1] - self.rows[0])
        cells = self.ends.shape[2]
        pixels = np.arange(self.rows[0] + first, self.rows[0] + stop)
        place = np.clip((pixels + 0.5) / _CELL - 0.5, 0, cells - 1)
        lower = np.minimum(place.astype(np.intp), max(cells - 2, 0))
        upper = np.minimum(lower + 1, cells - 1)
        a = (place - lower).astype(np.float32)[:, None]
        which = np.arange(len(self.ends)) if earlier is None else np.asarray(earlier)
        ends, levels = self.ends[which], self.levels[which]
        weights = _along_columns(ends[:, :, lower] * (1 - a) + ends[:, :, upper] * a, self.shape[1])
        level = levels[:, :, lower] * (1 - a) + levels[:, :, upper] * a
        found = np.concatenate([weights, level], axis=1).astype(np.float32)
        return np.ascontiguousarray(found[0] if len(found) == 1 else found)


def _along_columns(cells: np.ndarray, cols: int) -> np.ndarray:
    """Values on cells along the last axis, at every detector column of each cell."""
    return np.repeat(cells, _CELL, axis=-1)[..., :cols]


def view_weights(
    counts: CalibrationFrames,
    fields: Fields,
    interpolation: tuple[np.ndarray, ...],
    counts_steps: tuple[Step, ...] = (),
) -> tuple[FlatWeights | None, dict[str, Json]]:
    """The flat-set weights the views support, and what was found.

    ``fields`` are the whole detector's. Returns ``None`` when no view lies between
    two flat sets.
    """
    lo, hi, t, scales = interpolation
    sets = len(fields.positions)
    rows, cols = fields.dark.shape
    cells = (-(-rows // _CELL), -(-cols // _CELL))
    ends = np.zeros((sets - 1, 2, *cells), np.float32)
    ends[:, 1] = 1.0
    levels = np.zeros((sets - 1, 2, cells[0], cols), np.float32)
    tally = np.zeros(5)  # the views' share, unmatched energy, pixels, the overall weights
    intervals = 0
    for i in range(sets - 1):
        inside = np.flatnonzero((lo == i) & (hi == i + 1) & (t > 0) & (t < 1))
        if not inside.size:
            continue
        count = min(inside.size, max(64, -(-inside.size // 64)))
        picks = inside[np.unique(np.linspace(0, inside.size - 1, count).round().astype(np.intp))]
        sums, ratio = _sampled_sums(counts, fields, i, picks, scales[picks], counts_steps)
        ends[i], levels[i], found = _fitted(sums, t[picks].astype(np.float64), ratio)
        tally += found
        intervals += 1
    if not intervals:
        return None, {}
    pixels = max(float(tally[2]), np.finfo(np.float64).tiny)
    share, unmatched = float(tally[0]) / pixels, float(np.sqrt(tally[1] / pixels))
    start, end = (round(float(w) / intervals, 4) for w in tally[3:])
    LOG.info(
        "flat sets weighted by the views over %.1f%% of the detector (later set %.2f at the "
        "start, %.2f at the end); fine flat structure the views do not show: %.2g RMS",
        100 * share, start, end, unmatched,
    )  # fmt: skip
    return FlatWeights(ends, levels, (rows, cols), (0, rows)), {
        "flat_weights_from_views": round(share, 4),
        "later_flat_weight": [start, end],
        "flat_structure_unmatched": unmatched,
    }


def _sampled_sums(
    counts: CalibrationFrames,
    fields: Fields,
    earlier: int,
    picks: np.ndarray,
    scales: np.ndarray,
    counts_steps: tuple[Step, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Cell sums of views ``picks`` between sets ``earlier`` and the next, and the sets' log ratio.

    The sums are ``(6, views, cells_v, cells_u)`` (:func:`_view_sums`); the ratio
    is the mean over each cell's rows, ``(cells_v, columns)``.
    """
    rows, cols = fields.dark.shape
    dark, usable = jax.device_put(fields.dark), jax.device_put(fields.usable)
    flat = jax.device_put(fields.flats[earlier])
    tiny = jnp.finfo(jnp.float32).tiny
    later = jnp.maximum(jax.device_put(fields.flats[earlier + 1]), tiny)
    ratio = jnp.where(usable, jnp.log(later) - jnp.log(jnp.maximum(flat, tiny)), 0.0)
    noise = jax.device_put(fields.noise[earlier]), jax.device_put(fields.noise[earlier + 1])
    batch = max(1, _VIEW_BYTES // (4 * rows * cols))
    sums: list[np.ndarray] = []
    with (
        correction_group(("flat weights",), range(0, len(picks), batch), "view batches"),
        closing(view_batches(counts, len(picks), batch, picks)) as batches,
    ):
        for start, stop, raw in batches:
            progress.check_canceled()
            done = _view_sums(
                jnp.asarray(raw), jax.device_put(scales[start:stop]), dark, flat, ratio, *noise,
                usable, counts_steps=counts_steps,
            )  # fmt: skip
            sums.append(np.asarray(done))
            progress.step("view batches")
    # The ratio's mean over each cell's rows, at every column.
    padded = np.pad(np.asarray(ratio, np.float64), ((0, -rows % _CELL), (0, 0)))
    among = np.minimum(_CELL, rows - _CELL * np.arange(len(padded) // _CELL))[:, None]
    return np.concatenate(sums, axis=1), padded.reshape(-1, _CELL, cols).sum(1) / among


def _fitted(
    cell_sums: np.ndarray, t: np.ndarray, ratio: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weights and levels of one interval from its views' cell sums, and what they found.

    Four scales are fitted, each with the scale above as its prior: the whole
    detector (whose prior is linear interpolation), ``_BROAD`` cells each way,
    one cell each way, and the cell alone. Returns ``(2, cells_v, cells_u)``
    weights, ``(2, cells_v, columns)`` levels, and a tally of the views' share of
    the broad weights, the fine-structure energy that matches neither set, the
    pixels, and the overall weight at each end.
    """
    around = _windows(cell_sums, 1)
    normal = _normal(around, t)
    overall, _ = _solve(normal.sum((-2, -1), keepdims=True) / 9, _LINEAR)
    broad, share = _solve(_windows(normal, _BROAD) / 9, overall)
    near, _ = _solve(normal, broad)
    ends, _ = _solve(_normal(cell_sums.astype(np.float64), t), near)
    area = around[4].mean(0)
    neither = (broad * (1 - broad)).mean(0)  # 0 where each end matches one set
    energy = np.maximum(around[1].mean(0), 0.0)
    found = np.array(
        [
            (share * area).sum(),
            (share * energy * neither).sum(),
            area.sum(),
            overall[0].item(),
            overall[1].item(),
        ]
    )
    return ends.astype(np.float32), _levels(ends, broad, share, ratio), found


@partial(jax.jit, static_argnames=("counts_steps",))
def _view_sums(
    raw: jax.Array,
    scale: jax.Array,
    dark: jax.Array,
    earlier: jax.Array,
    ratio: jax.Array,
    noise_earlier: jax.Array,
    noise_later: jax.Array,
    usable: jax.Array,
    *,
    counts_steps: tuple[Step, ...],
) -> jax.Array:
    """Cell sums of each view's products with the flat sets' difference.

    Pixels are weighted by the transmission around them (the inverse variance of
    a logged count), in the regression and in the running mean the high-pass
    removes, so opaque regions neither enter the fit nor leak into their neighbors.
    Returns ``(6, views, cells_v, cells_u)``: the product less the earlier set's
    noise, the difference's energy less both sets' noise, that energy, the
    view's own energy, the weights, and the variance of the noise energy removed.
    """
    x = raw.astype(jnp.float32)
    for step in counts_steps:
        x = step.apply(x)
    tiny = jnp.finfo(jnp.float32).tiny
    signal = (x - dark) * scale[:, None, None]
    flat = jnp.where(usable, jnp.maximum(earlier, tiny), 1.0)
    lit = usable & (signal > 0)
    # The mean transmission around a pixel: a weight that follows neither the object's
    # fine structure nor the flat's, whose products with them would bias the fit.
    around = _row_sum(jnp.where(lit, signal, 0.0)) / jnp.maximum(_row_sum(lit * flat), tiny)
    weight = jnp.where(lit, jnp.clip(around, 0.0, 1.0), 0.0)
    y = jnp.log(jnp.maximum(signal, tiny) / flat)
    total = jnp.maximum(_row_sum(weight), tiny)

    def high(values: jax.Array) -> jax.Array:
        return values - _row_sum(weight * values) / total

    view, difference = high(y), high(jnp.broadcast_to(ratio, y.shape))
    gain = 1 - 1 / _WIDTH  # white noise left by removing a running mean
    energy = weight * difference * difference
    noise = gain * weight * (noise_earlier + noise_later)
    return jnp.stack(
        [
            _cells(weight * view * difference - gain * weight * noise_earlier),
            _cells(energy - noise),
            _cells(energy),
            _cells(weight * view * view),
            _cells(weight),
            _cells(2 * noise * noise),  # a squared Gaussian's variance is twice its mean squared
        ]
    )


def _row_sum(values: jax.Array) -> jax.Array:
    """Sum over ``_WIDTH`` columns centered on each pixel, zero beyond the detector."""
    half = _WIDTH // 2
    return jax.lax.reduce_window(
        values, 0.0, jax.lax.add, (1, 1, _WIDTH), (1, 1, 1), ((0, 0), (0, 0), (half, half))
    )


def _cells(values: jax.Array) -> jax.Array:
    views, rows, cols = values.shape
    padded = jnp.pad(values, ((0, 0), (0, -rows % _CELL), (0, -cols % _CELL)))
    return padded.reshape(views, -1, _CELL, padded.shape[2] // _CELL, _CELL).sum((2, 4))


def _windows(sums: np.ndarray, reach: int) -> np.ndarray:
    """Sums over the cells within ``reach`` of each cell along the last two axes, in float64."""
    total = sums.astype(np.float64)
    for axis in (total.ndim - 2, total.ndim - 1):
        pad = [(0, 0)] * total.ndim
        pad[axis] = (reach + 1, reach)
        along = np.cumsum(np.pad(total, pad), axis=axis)
        size = total.shape[axis]
        upper = np.take(along, np.arange(2 * reach + 1, 2 * reach + 1 + size), axis=axis)
        total = upper - np.take(along, np.arange(size), axis=axis)
    return total


def _normal(sums: np.ndarray, t: np.ndarray) -> np.ndarray:
    """What the views say of the later set's weight at both ends of an interval.

    ``sums`` are window sums of :func:`_view_sums`. In each cell, for view ``k``
    at ``t[k]`` of the interval,
    ``product[k] = signal[k] * ((1 - t[k]) * start + t[k] * end)`` plus an error
    whose variance is the difference's energy times the view's mean square. That
    variance assumes independent pixels; the scatter of the views about the fit
    inflates it where the object's structure correlates them, and a Huber weight
    bounds single views. ``signal`` is itself an estimate with variance ``doubt``,
    the same in every view: the views fix a weight no better than ``signal``
    stands above three of its standard errors, so a difference that is noise
    says nothing at all.

    Returns the normal equations of ``(start, end)`` without a prior,
    ``(5, cells_v, cells_u)``: the three entries of the symmetric matrix, then
    the right side.
    """
    product, signal, energy, residual, mass, doubt = sums
    tiny = np.finfo(np.float64).tiny
    design = np.maximum(signal, 0.0) * np.stack([1 - t, t])[:, :, None, None]
    variance = np.maximum(energy * residual / np.maximum(mass, tiny), tiny)
    error = np.sqrt(np.maximum(doubt.mean(0), tiny))
    known = np.maximum(signal.mean(0) / error - 3.0, 0.0) ** 2
    huber = np.ones_like(product)
    inflation = bound = np.ones(product.shape[1:])
    normal = np.zeros((5, *product.shape[1:]))
    for _ in range(4):
        q = bound * huber / (variance * inflation)
        normal = np.stack(
            [
                (q * design[0] ** 2).sum(0),
                (q * design[0] * design[1]).sum(0),
                (q * design[1] ** 2).sum(0),
                (q * design[0] * product).sum(0),
                (q * design[1] * product).sum(0),
            ]
        )
        (start, end), _ = _solve(normal, _LINEAR, clip=False)
        squared = (product - start * design[0] - end * design[1]) ** 2 / variance
        if len(t) > 3:
            scatter = np.median(squared, axis=0) / _CHI2_MEDIAN * len(t) / (len(t) - 2)
            inflation = np.maximum(1.0, scatter)
        huber = np.minimum(1.0, 1.345 / np.sqrt(np.maximum(squared / inflation, tiny)))
        data = (normal[0] + normal[2]) / np.maximum(bound, tiny)
        bound = known / np.maximum(known + data, tiny)
    return normal


def _solve(
    normal: np.ndarray, prior: np.ndarray, *, clip: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Weights at both ends from normal equations and a prior, and the views' share.

    The prior's two means are each as uncertain as a weight known only to lie in
    [0, 1]. Returns ``(2, ...)`` weights, clipped to [0, 1] unless told otherwise,
    and the fraction of their precision that came from the views.
    """
    m00, m01, m11, b0, b1 = normal
    a00, a11 = m00 + _PRIOR_PRECISION, m11 + _PRIOR_PRECISION
    b0, b1 = b0 + _PRIOR_PRECISION * prior[0], b1 + _PRIOR_PRECISION * prior[1]
    determinant = a00 * a11 - m01**2
    ends = np.stack([(a11 * b0 - m01 * b1) / determinant, (a00 * b1 - m01 * b0) / determinant])
    if clip:
        ends = np.clip(ends, 0.0, 1.0)
    return ends, (m00 + m11) / (m00 + m11 + 2 * _PRIOR_PRECISION)


def _levels(
    ends: np.ndarray, broad: np.ndarray, share: np.ndarray, ratio: np.ndarray
) -> np.ndarray:
    """The log factor on the blended flat that sets its level.

    The weights ``ends`` describe the sets' fine structure and are noisy from cell
    to cell; multiplying a level difference between the sets by them would print
    that noise, and every change of weight, into the flat as steps. For each end
    of the interval and each row of cells, with ``w`` the weights along the row
    and ``r = exp(ratio)``, the blend ``1 + w (r - 1)`` steps by
    ``log(1 + w[u + 1] (r[u + 1] - 1)) - log(1 + w[u] (r[u + 1] - 1))`` from the
    change of weight alone. The level ``d`` minimizes the squares of
    ``d[u + 1] - d[u]`` plus that step, and of the distance of the blend's level
    from that of the ``broad`` weights (fitted over ``_BROAD`` cells each way),
    held as far as those are to be believed: fully where the views match a set (a
    weight of 0 or 1 at both ends) or say nothing, not at all where they match
    neither, so the level reached at an edge carries across such a region.
    ``ends``, ``broad`` and ``share`` are on cells, ``ratio`` on
    ``(cells_v, columns)``; returns ``(2, cells_v, columns)``.
    """
    cols = ratio.shape[1]
    unmatched = 4 * np.max(broad * (1 - broad), axis=0)
    hold = _along_columns((1 - share * unmatched) ** 8, cols) / _WIDTH**2 + 1e-9
    change = np.expm1(ratio.astype(np.float64))[None]
    weights = _along_columns(ends.astype(np.float64), cols)
    step = np.log1p(weights[..., 1:] * change[..., 1:]) - np.log1p(
        weights[..., :-1] * change[..., 1:]
    )
    target = np.log1p(_along_columns(broad.astype(np.float64), cols) * change) - np.log1p(
        weights * change
    )
    if cols < 2 or not (step.any() or target.any()):
        return np.zeros((2, *ratio.shape), np.float32)
    # Tridiagonal normal equations, -1 off the diagonal, solved along each row at once.
    diagonal = np.broadcast_to(2.0 + hold, target.shape).copy()
    diagonal[..., [0, -1]] -= 1.0
    right = hold * target
    right[..., :-1] += step
    right[..., 1:] -= step
    over = np.empty_like(diagonal)
    over[..., 0] = -1.0 / diagonal[..., 0]
    right[..., 0] /= diagonal[..., 0]
    for u in range(1, cols):
        pivot = diagonal[..., u] + over[..., u - 1]
        over[..., u] = -1.0 / pivot
        right[..., u] = (right[..., u] + right[..., u - 1]) / pivot
    for u in range(cols - 2, -1, -1):
        right[..., u] -= over[..., u] * right[..., u + 1]
    return right.astype(np.float32)
