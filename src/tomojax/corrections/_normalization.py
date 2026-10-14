"""Shared raw normalization and log boundary for full frames and calibrated windows."""

from functools import partial

import jax
import jax.numpy as jnp

from ._steps import Step


@partial(jax.jit, static_argnames=("epsilon", "counts_steps"))
def normalize(
    raw,
    lo,
    hi,
    t,
    flats,
    dark,
    flat_variance,
    dark_variance,
    *,
    epsilon: float,
    counts_steps: tuple[Step, ...] = (),
) -> jax.Array:
    x = raw.astype(jnp.float32)
    for step in counts_steps:
        x = step.apply(x)
    weight = t[:, None, None]
    # An unused calibration must not contaminate an endpoint through 0 * NaN
    # (or 0 * Inf). Interior views still need both bracketing fields.
    flat = jnp.where(
        weight == 0, flats[lo],
        jnp.where(weight == 1, flats[hi], (1 - weight) * flats[lo] + weight * flats[hi]),
    )  # fmt: skip
    variance = jnp.where(
        weight == 0, flat_variance[lo],
        jnp.where(weight == 1, flat_variance[hi],
                  (1 - weight) ** 2 * flat_variance[lo] + weight**2 * flat_variance[hi]),
    ) + dark_variance  # fmt: skip
    light = flat - dark
    reliable = (
        jnp.isfinite(light) & jnp.isfinite(variance)
        & (light > jnp.maximum(epsilon, 3 * jnp.sqrt(variance)))
    )  # fmt: skip
    # A calibration indistinguishable from darkness is an unmeasured pixel,
    # not a denominator of epsilon. Use neutral transmission before spatial
    # steps, so an FFT cannot spread enormous counts across the whole image.
    return jnp.where(reliable, (x - dark) / jnp.maximum(light, epsilon), 1.0)


def log_transmission(
    transmission: jax.Array,
    *,
    epsilon: float,
    line_steps: tuple[Step, ...],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    nonpositive = jnp.sum(transmission <= 0)
    p = -jnp.log(jnp.maximum(transmission, epsilon))
    for step in line_steps:
        p = step.apply(p)
    bad = ~jnp.isfinite(p)
    return jnp.where(bad, 0.0, p), nonpositive, jnp.sum(bad)


@partial(jax.jit, static_argnames=("epsilon", "counts_steps", "transmission_steps", "line_steps"))
def first_pass(
    raw: jax.Array,
    lo: jax.Array,
    hi: jax.Array,
    t: jax.Array,
    flats: jax.Array,
    dark: jax.Array,
    flat_variance: jax.Array,
    dark_variance: jax.Array,
    *,
    epsilon: float,
    counts_steps: tuple[Step, ...],
    transmission_steps: tuple[Step, ...],
    line_steps: tuple[Step, ...],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    transmission = normalize(
        raw,
        lo,
        hi,
        t,
        flats,
        dark,
        flat_variance,
        dark_variance,
        epsilon=epsilon,
        counts_steps=counts_steps,
    )
    for step in transmission_steps:
        transmission = step.apply(transmission)
    return log_transmission(transmission, epsilon=epsilon, line_steps=line_steps)
