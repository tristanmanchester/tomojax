"""Projection weighting shared by nuisance fits."""

import jax
import jax.numpy as jnp


def mask_weights(mask: jax.Array | None, predicted: jax.Array) -> jax.Array:
    """Broadcast detector weights over views while preserving explicit stack weights."""
    if mask is None:
        return jnp.ones_like(predicted, dtype=jnp.float32)
    weights = jnp.asarray(mask, dtype=jnp.float32)
    if weights.shape == predicted.shape:
        return weights
    if weights.ndim == predicted.ndim - 1:
        return jnp.broadcast_to(weights, predicted.shape)
    raise ValueError("mask must match projection shape or detector shape")
