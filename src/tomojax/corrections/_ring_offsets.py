"""Estimate the additive detector field from angular ranks and adjacent-column contrasts."""

from functools import lru_cache
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.robust import MAD_TO_SD

from ._steps import bitonic_median, odd_even_median


def window_sum(values: jax.Array, width: int, padding: Literal["SAME", "VALID"]) -> jax.Array:
    """Sum contiguous samples along the leading axis."""
    shape = (width,) + (1,) * (values.ndim - 1)
    return jax.lax.reduce_window(values, 0.0, jax.lax.add, shape, (1,) * values.ndim, padding)


def shifted_columns(x: jax.Array, half: int) -> list[jax.Array]:
    """The columns of ``x`` shifted by ``-half..half``, edge columns repeated."""
    padded = jnp.pad(x, ((0, 0), (half, half)), mode="edge")
    n = x.shape[1]
    return [padded[:, half + j : half + j + n] for j in range(-half, half + 1)]


@lru_cache(maxsize=16)
def quadratic_taps(window: int, guard: int) -> np.ndarray:
    """Weights giving each column the local quadratic through its neighbors (guard left out)."""
    j = np.arange(-window, window + 1, dtype=np.float64)
    keep = np.abs(j) > guard
    design = np.stack([np.ones_like(j), j, j * j], axis=1)[keep]
    taps = np.zeros_like(j)
    taps[keep] = np.linalg.pinv(design)[0]
    return taps


def correlate(x: jax.Array, kernel: np.ndarray) -> jax.Array:
    """Apply a symmetric stencil, accumulating reflected samples together."""
    half = len(kernel) // 2
    shifted = shifted_columns(x, half)
    out = float(kernel[half]) * x
    for j in range(1, half + 1):
        k = float((kernel[half - j] + kernel[half + j]) * 0.5)
        if k:
            out = out + k * (shifted[half - j] + shifted[half + j])
    return out


def _block_means(values: jax.Array, scale: int) -> tuple[jax.Array, jax.Array]:
    """Means over ``scale``-column blocks centered on the detector middle, edges repeated.

    Block ``j`` covers ``[m + j scale - scale/2, m + j scale + scale/2]`` with
    ``m`` the middle column, so the blocks, and the correction, are symmetric
    under reversal of the detector. A boundary through a column's middle takes
    half of it. Returns the block centers in columns and the means,
    ``(rows, blocks)``.
    """
    columns = values.shape[1]
    middle = (columns - 1) / 2
    reach = -(-int(np.ceil(middle)) // scale)
    centers = middle + scale * np.arange(-reach, reach + 1)
    # Columns are [i - 1/2, i + 1/2]: blocks end on column edges for an even
    # count and on column middles for an odd one.
    halves = columns % 2
    taps = scale + halves
    weights = np.ones(taps)
    if halves:
        weights[[0, -1]] = 0.5
    first = round(centers[0] - scale / 2 + (0.5 if not halves else 0.0))
    pad = max(0, -first, first + (len(centers) - 1) * scale + taps - columns)
    padded = jnp.pad(values, ((0, 0), (pad, pad)), mode="edge")
    start = first + pad
    stop = start + (len(centers) - 1) * scale + 1
    total = weights[0] * padded[:, start:stop:scale]
    for k in range(1, taps):
        total = total + weights[k] * padded[:, start + k : stop + k : scale]
    return jnp.asarray(centers, values.dtype), total / scale


def _ranked_offsets(ranked: jax.Array) -> jax.Array:
    """Fit offsets on four-column blocks, then two-column blocks, then pixels."""
    columns = jnp.arange(ranked.shape[1], dtype=ranked.dtype)
    offsets = jnp.zeros_like(columns)
    for scale in (4, 2):
        centers, blocks = _block_means(ranked - offsets, scale)
        reference = odd_even_median(shifted_columns(blocks, 4))
        offsets += jnp.interp(columns, centers, jnp.median(blocks - reference, axis=0))
    residual = ranked - offsets
    reference = odd_even_median(shifted_columns(residual, 4))
    return offsets + jnp.median(residual - reference, axis=0)


def offset_prior(row: jax.Array, groups: tuple[int, ...]) -> jax.Array:
    """A row's repeatable detector offsets, up to an unobservable constant.

    Rank profiles give offsets relative to local neighbors. At the edge of a wide
    constant defect, both neighborhoods can be internally consistent. Differences
    of adjacent columns still measure that boundary, regardless of the width.
    Disjoint angular arcs provide a variance for each measurement.
    """
    views, columns = row.shape
    if columns < 2:
        return jax.lax.full((columns,), 0, row.dtype)
    parts: list[int] = []
    for size in groups:
        count = max(1, min(size, round(8 * size / views)))
        parts.extend((i + 1) * size // count - i * size // count for i in range(count))
    lengths = np.asarray(parts, dtype=np.int32)
    starts = np.cumsum(lengths) - lengths
    k = np.arange(max(parts))
    indices = starts[:, None] + np.minimum(k, lengths[:, None] - 1)
    valid = k < lengths[:, None]
    arc_dof = max(len(lengths) - 1, 1)

    def ordered_arcs(values: jax.Array) -> jax.Array:
        return jnp.sort(jnp.where(valid[:, :, None], values[indices], jnp.inf), axis=1)

    # Sixteen equiprobable ranks per arc describe its range of transmissions.
    ranks = ((np.arange(16) + 0.5) * lengths[:, None] / 16).astype(int)
    quantiles = jnp.take_along_axis(ordered_arcs(row), ranks[:, :, None], axis=1)
    fields = jax.vmap(_ranked_offsets)(quantiles)
    level = fields.mean(0)
    variance = fields.var(0) / arc_dof
    power = jnp.maximum(MAD_TO_SD**2 * jnp.median(level**2) - jnp.median(variance), 0)
    level = level * power / (power + variance + 1e-20)

    # A persistent jump in the gain changes the local slope at the same column
    # in every view. A moving sample edge crosses different columns. The median
    # across views estimates the jump with a Laplace prior in gradient space.
    slope = jnp.diff(row - level, axis=1)
    residual = slope - bitonic_median(shifted_columns(slope, 4))
    ordered = ordered_arcs(residual)
    mid = np.stack(((lengths - 1) // 2, lengths // 2), axis=1)
    arc_jumps = jnp.take_along_axis(ordered, mid[:, :, None], axis=1).mean(1)
    variance = arc_jumps.var(0) / arc_dof
    jump = jnp.median(residual, axis=0)
    jump = jnp.sign(jump) * jnp.maximum(
        jnp.abs(jump) - variance / (jnp.abs(jump).mean() + 1e-15), 0
    )

    # The integrated measurement has variance proportional to 1 / |D(f)|².
    # For an exponentially correlated detector field, lag-two over lag-one
    # increment power is 1 + rho. This gives the Wiener integral without a
    # chosen stripe width. Reflect at the detector edges and leave DC alone.
    v1 = jnp.mean(jump**2)
    v2 = jnp.mean((jump[:-1] + jump[1:]) ** 2) if columns > 2 else v1
    rho = jnp.clip(v2 / (v1 + 1e-20) - 1, 0, 1)
    strength = v1 * (1 + rho) / 2
    h2 = (4 * np.sin(np.pi * np.fft.rfftfreq(2 * columns)) ** 2).astype(row.dtype)
    gain = strength * h2 / (strength * h2 + variance.mean() * ((1 - rho) ** 2 + rho * h2) + 1e-20)
    field = jnp.concatenate([jax.lax.full((1,), 0, row.dtype), jnp.cumsum(jump)])
    spectrum = jnp.fft.rfft(jnp.concatenate([field, field[::-1]]))
    return level + jnp.fft.irfft(spectrum * gain, n=2 * columns)[:columns]
