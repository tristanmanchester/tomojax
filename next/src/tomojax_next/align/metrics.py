from __future__ import annotations

import jax.numpy as jnp


def rmse_deg(pred_deg: jnp.ndarray, true_deg: jnp.ndarray) -> float:
    d = jnp.asarray(pred_deg) - jnp.asarray(true_deg)
    return float(jnp.sqrt(jnp.mean(d * d)))


def rmse_pixels(pred: jnp.ndarray, true: jnp.ndarray) -> float:
    d = jnp.asarray(pred) - jnp.asarray(true)
    return float(jnp.sqrt(jnp.mean(d * d)))

