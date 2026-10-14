"""GPU evaluation of the full-resolution Vo centre metric."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def _metrics(
    sino: jax.Array,
    offsets: jax.Array,
    folded: jax.Array,
    freq: jax.Array,
    count: jax.Array,
) -> jax.Array:
    views, cols = sino.shape
    mirror = sino[:, ::-1]
    spectrum = jnp.fft.rfft(mirror, axis=1)
    columns = jnp.arange(cols)

    def one(offset: jax.Array) -> jax.Array:
        shift = 2 * offset

        def fractional(_: None) -> jax.Array:
            phase = jnp.exp(-2j * jnp.pi * freq * shift).astype(jnp.complex64)
            return jnp.fft.irfft(spectrum * phase, n=cols, axis=1)

        moved = jax.lax.cond(
            shift == jnp.floor(shift),
            lambda _: jnp.roll(mirror, shift.astype(jnp.int32), axis=1),
            fractional,
            None,
        )
        edge = jnp.ceil(jnp.abs(shift)).astype(jnp.int32)
        fill = jnp.where(shift > 0, columns < edge, columns >= cols - edge)
        moved = jnp.where(fill[None], sino[::-1], moved)
        energy = jnp.abs(jnp.fft.rfft2(jnp.concatenate((sino, moved), axis=0)))
        return jnp.sum(energy * folded) / (2 * views * cols)

    # A runtime count shares one executable between both refinement passes,
    # without evaluating padded candidates or batching their large FFT buffers.
    def body(i: jax.Array, out: jax.Array) -> jax.Array:
        return out.at[i].set(one(offsets[i]))

    return jax.lax.fori_loop(0, count, body, jnp.zeros(offsets.shape, dtype=sino.dtype))


def gpu_metrics(sino: np.ndarray, offsets: np.ndarray, folded: np.ndarray) -> np.ndarray | None:
    """The metric for up to thirteen offsets on the GPU, or None when not on a GPU."""
    if jax.default_backend() != "gpu":
        return None
    padded = np.zeros(13, dtype=np.float32)
    padded[: len(offsets)] = offsets
    freq = np.fft.rfftfreq(sino.shape[1]).astype(np.float32)
    out = _metrics(sino, padded, folded, freq, np.int32(len(offsets)))
    return np.asarray(out)[: len(offsets)]
