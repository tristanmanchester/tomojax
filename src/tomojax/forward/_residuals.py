"""Masked robust projection residuals."""
# pyright: reportAny=false, reportUnknownMemberType=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ResidualResult:
    """Store whitened residuals and their reduced robust loss.

    Parameters
    ----------
    residual : jax.Array
        Masked, whitened differences with the projection shape.
    loss : jax.Array
        Scalar sum of elementwise loss divided by max(valid_count, 1).
    weights : jax.Array
        Elementwise robust weights, or ones for L2, with the residual shape.
    valid_count : jax.Array
        Sum of mask values, or sample count when no mask was supplied.
    """

    residual: jax.Array
    loss: jax.Array
    weights: jax.Array
    valid_count: jax.Array


ResidualLossMode = Literal["l2", "pseudo_huber"]


def masked_whitened_residual(
    predicted: jax.Array,
    observed: jax.Array,
    *,
    mask: jax.Array | None = None,
    sigma: float | jax.Array = 1.0,
) -> jax.Array:
    """Whiten differences, excluding zero-mask entries before unsafe arithmetic.

    Parameters
    ----------
    predicted, observed : jax.Array
        Prediction and measured projections with identical shapes, usually
        (views, nv, nu). Values are converted to float32.
    mask : jax.Array or None, optional
        Multiplicative weights of the same shape. Zero entries are excluded
        before subtraction and division; None keeps every sample.
    sigma : float or jax.Array, optional
        Positive noise standard deviations, broadcastable to the projection
        shape; default is 1.

    Returns
    -------
    jax.Array
        Float32 (predicted-observed)/sigma multiplied by mask, or unmasked
        when mask is None. The shape matches the inputs.

    Raises
    ------
    ValueError
        If prediction, observation or mask shapes disagree.
    """
    pred = jnp.asarray(predicted, dtype=jnp.float32)
    obs = jnp.asarray(observed, dtype=jnp.float32)
    if pred.shape != obs.shape:
        raise ValueError("predicted and observed projections must have matching shapes")
    sigma_arr = jnp.asarray(sigma, dtype=jnp.float32)
    if mask is None:
        return (pred - obs) / sigma_arr
    mask_arr = jnp.asarray(mask, dtype=jnp.float32)
    if mask_arr.shape != pred.shape:
        raise ValueError("mask must match projection shape")
    active = mask_arr != 0
    pred = jnp.where(active, pred, 0.0)
    obs = jnp.where(active, obs, 0.0)
    sigma_arr = jnp.where(active, sigma_arr, 1.0)
    return ((pred - obs) / sigma_arr) * mask_arr


def pseudo_huber_loss(residual: jax.Array, *, delta: float = 1.0) -> jax.Array:
    """Compute elementwise pseudo-Huber loss without small-residual cancellation.

    Parameters
    ----------
    residual : jax.Array
        Residual array of any shape, converted to float32.
    delta : float, optional
        Positive transition scale in residual units; default is 1.

    Returns
    -------
    jax.Array
        Elementwise delta**2 * (sqrt(1 + (residual/delta)**2) - 1),
        with the residual shape.
    """
    r = jnp.asarray(residual, dtype=jnp.float32)
    d = jnp.asarray(delta, dtype=jnp.float32)
    # d^2 (sqrt(1 + (r/d)^2) - 1), rewritten so that |r| << d does not give 0.
    return r * r / (1.0 + jnp.sqrt(1.0 + (r / d) ** 2))


def pseudo_huber_weights(residual: jax.Array, *, delta: float = 1.0) -> jax.Array:
    """Compute the iteratively reweighted least-squares weights for pseudo-Huber.

    Parameters
    ----------
    residual : jax.Array
        Residual array of any shape, converted to float32.
    delta : float, optional
        Positive transition scale in residual units; default is 1.

    Returns
    -------
    jax.Array
        Elementwise 1 / sqrt(1 + (residual/delta)**2), with the residual shape.
    """
    r = jnp.asarray(residual, dtype=jnp.float32)
    d = jnp.asarray(delta, dtype=jnp.float32)
    return 1.0 / jnp.sqrt(1.0 + (r / d) ** 2)


def robust_residual_scale(residual: jax.Array, *, mask: jax.Array | None = None) -> jax.Array:
    """Estimate normal-consistent median absolute deviation under JAX transforms.

    Empty, constant or nonfinite-MAD samples retain the fallback scale of one.

    Parameters
    ----------
    residual : jax.Array
        Residual samples of any shape, converted to float32.
    mask : jax.Array or None, optional
        Same-shaped sample selector; nonzero entries select samples and their
        magnitudes do not weight the medians. None selects all samples.

    Returns
    -------
    jax.Array
        Scalar 1.4826 times the selected median absolute deviation about the
        selected median, or one when that estimate is unusable.

    Raises
    ------
    ValueError
        If mask and residual shapes differ.
    """
    values = jnp.asarray(residual, dtype=jnp.float32)
    valid = jnp.asarray(True)
    if mask is not None:
        mask_arr = jnp.asarray(mask, dtype=bool)
        if mask_arr.shape != values.shape:
            raise ValueError("mask must match residual shape")
    if values.size == 0:
        return jnp.asarray(1.0, dtype=jnp.float32)
    if mask is None:
        median = jnp.median(values)
        mad = jnp.median(jnp.abs(values - median))
    else:
        # Missing NaN/Inf samples must be removed before centering, not merely
        # hidden after an unsafe subtraction. Sorting sentinels keeps the shape
        # fixed while the selected count and median indices stay dynamic.
        values = jnp.where(mask_arr, values, 0.0)
        valid = ~jnp.any(jnp.isnan(values))
        count = jnp.sum(mask_arr)
        median = _selected_median(values, mask_arr, count)
        mad = _selected_median(jnp.abs(values - median), mask_arr, count)
    scale = jnp.asarray(1.4826, dtype=jnp.float32) * mad
    return jnp.where(
        valid & jnp.isfinite(scale) & (scale > 0.0), scale, jnp.asarray(1.0, dtype=jnp.float32)
    )


def _selected_median(values: jax.Array, mask: jax.Array, count: jax.Array) -> jax.Array:
    ordered = jnp.sort(jnp.where(mask, values, jnp.inf).ravel())
    lower = jnp.where(count > 0, ordered[jnp.maximum((count - 1) // 2, 0)], 0.0)
    upper = jnp.where(count > 0, ordered[count // 2], 0.0)
    return lower * 0.5 + upper * 0.5


def residual_loss(
    predicted: jax.Array,
    observed: jax.Array,
    *,
    mask: jax.Array | None = None,
    sigma: float | jax.Array = 1.0,
    delta: float = 1.0,
    mode: ResidualLossMode = "pseudo_huber",
) -> ResidualResult:
    """Compute masked whitened residuals and their mean robust or L2 loss.

    Parameters
    ----------
    predicted, observed : jax.Array
        Prediction and measured projections with identical shapes, usually
        (views, nv, nu). Values are converted to float32.
    mask : jax.Array or None, optional
        Multiplicative weights of the same shape. Zero entries are excluded
        before subtraction and division; None keeps every sample.
    sigma : float or jax.Array, optional
        Positive noise standard deviations, broadcastable to the projection
        shape; default is 1.
    delta : float, optional
        Positive pseudo-Huber transition in whitened residual units; default is 1.
    mode : {'l2', 'pseudo_huber'}, optional
        Elementwise loss; default is 'pseudo_huber'. L2 uses half squared error.

    Returns
    -------
    ResidualResult
        Residuals, scalar reduced loss, robust weights and normalization count.
        A nonbinary mask weights residuals and uses its sum for normalization.

    Raises
    ------
    ValueError
        If array shapes disagree or the loss mode is unsupported.
    """
    residual = masked_whitened_residual(predicted, observed, mask=mask, sigma=sigma)
    if mode == "pseudo_huber":
        loss_map = pseudo_huber_loss(residual, delta=delta)
        weights = pseudo_huber_weights(residual, delta=delta)
    elif mode == "l2":
        loss_map = jnp.asarray(0.5, dtype=jnp.float32) * residual * residual
        weights = jnp.ones_like(residual, dtype=jnp.float32)
    else:
        raise ValueError(f"unknown residual loss mode {mode!r}")
    if mask is None:
        valid_count = jnp.asarray(residual.size, dtype=jnp.float32)
    else:
        valid_count = jnp.sum(jnp.asarray(mask, dtype=jnp.float32))
    loss = jnp.sum(loss_map) / jnp.maximum(valid_count, 1.0)
    return ResidualResult(residual=residual, loss=loss, weights=weights, valid_count=valid_count)
