"""Bounded GPU evaluation of the full-resolution Vo centre metric."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.backends.api import device_free_memory_bytes, is_oom_error


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


def try_metrics(sino: np.ndarray, offsets: np.ndarray, folded: np.ndarray) -> np.ndarray | None:
    """Return GPU metrics, or None when this search should retain CPU execution.

    Unknown/tight allocator headroom skips execution. Compiled buffer sizes
    exclude some cuFFT scratch, so reserve twice their total and retain a narrow
    allocation-failure fallback. No measurements or failure state persist here.
    """
    if jax.config.jax_disable_jit:
        return None
    device = jax.config.jax_default_device
    if device is None or isinstance(device, str):
        device = next(iter(jax.devices(device or jax.default_backend())), None)
    if device is None or device.platform != "gpu":
        return None
    free = device_free_memory_bytes(device=device)
    if free is None:
        return None
    # The fine search has at most thirteen candidates. Its second pass reuses
    # the same array shape, even when several candidates were already scored.
    padded = np.zeros(13, dtype=np.float32)
    padded[: len(offsets)] = offsets
    args = (
        sino,
        padded,
        folded,
        np.fft.rfftfreq(sino.shape[1]).astype(np.float32),
        np.int32(len(offsets)),
    )
    try:
        compiled = _metrics.lower(*args).compile()
        workspace = compiled.memory_analysis()
        reserve = (
            None
            if workspace is None
            else 2
            * (
                workspace.temp_size_in_bytes
                + workspace.argument_size_in_bytes
                + workspace.output_size_in_bytes
            )
        )
        free = device_free_memory_bytes(device=device)
        if reserve is None or free is None or free < reserve:
            return None
        return np.asarray(compiled(*args))[: len(offsets)]
    except Exception as exc:
        if not is_oom_error(exc):
            raise
        return None
