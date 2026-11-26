"""Estimate the additive detector field from angular ranks and adjacent-column contrasts."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.robust import MAD_TO_SD

from ._steps import bitonic_median, ranked_offsets


def shifted_columns(x: jax.Array, half: int) -> list[jax.Array]:
    """The columns of ``x`` shifted by ``-half..half``, edge columns repeated."""
    padded = jnp.pad(x, ((0, 0), (half, half)), mode="edge")
    n = x.shape[1]
    return [padded[:, half + j : half + j + n] for j in range(-half, half + 1)]


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
    fields = jax.vmap(partial(ranked_offsets, width=33))(quantiles)
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
