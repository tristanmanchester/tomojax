"""Filtering of the measured columns and the two exterior profiles shared across views."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ._continuation import Continuation


def filter_rows(rows: jax.Array, kernel: jax.Array, nu: int, pad_lo: int, pad_hi: int) -> jax.Array:
    """Ramp-filter continued rows, keeping ``pad_lo``/``pad_hi`` columns beyond the detector.

    Filtered rows extend past the data; an offset detector's backprojection
    needs that tail on its short side.
    """
    n_fft = 2 * (int(kernel.shape[0]) - 1)
    out = jnp.fft.irfft(jnp.fft.rfft(rows, n=n_fft, axis=-1) * kernel, n=n_fft, axis=-1)
    return output_columns(out, (rows.shape[-1] - nu) // 2 - pad_lo, pad_lo + nu + pad_hi)


def output_columns(out: jax.Array, first: int, count: int) -> jax.Array:
    """Read a contiguous column interval of the circular transform, crossing zero if needed."""
    if first >= 0:
        return out[..., first : first + count]
    return jnp.concatenate([out[..., first:], out[..., : first + count]], axis=-1)


@partial(jax.jit, static_argnames=("distance", "pad_lo", "pad_hi"))
def filtered_profiles(
    radius: jax.Array,
    cosine: jax.Array | None,
    spectrum: jax.Array,
    distance: tuple[float, ...],
    pad_lo: int,
    pad_hi: int,
) -> jax.Array:
    """Filter the two unit-amplitude exterior profiles when every view shares its cosine image."""
    nu = len(distance) // 3
    fits = Continuation(np.asarray(distance), nu, nu)
    unit = jnp.ones_like(radius)
    low, high = fits.tails(partial(jnp.asarray, dtype=radius.dtype), unit, unit, radius)
    zeros = jnp.zeros_like(low)
    pair = jnp.stack(
        (
            jnp.concatenate((low, zeros, zeros), axis=-1),
            jnp.concatenate((zeros, zeros, high), axis=-1),
        )
    )
    if cosine is not None:
        pair *= cosine
    return filter_rows(pair, spectrum, nu, pad_lo, pad_hi)
