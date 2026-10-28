"""Projection-domain residual filters."""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

import jax
import jax.numpy as jnp

ResidualFilterKind = Literal["raw", "lowpass_gaussian", "bandpass_difference_of_gaussians"]


@dataclass(frozen=True)
class ResidualFilterConfig:
    """Configure a periodic detector-plane residual filter.

    Parameters
    ----------
    kind : {'raw', 'lowpass_gaussian', 'bandpass_difference_of_gaussians'}, optional
        Filter family; default is 'raw'. Filtering acts on the last two axes.
    weight : float, optional
        Multiplicative output weight; default is 1.
    sigma_px : float, optional
        Gaussian width, or inner bandpass width, in pixels; default is 1.
    outer_sigma_px : float, optional
        Outer bandpass Gaussian width in pixels; default is 2.
    """

    kind: ResidualFilterKind = "raw"
    weight: float = 1.0
    sigma_px: float = 1.0
    outer_sigma_px: float = 2.0


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ResidualFilterResult:
    """Store a summed residual-filter schedule and its individual outputs.

    Parameters
    ----------
    residual : jax.Array
        Sum of all weighted, masked filter outputs, with the input shape.
    components : tuple of jax.Array
        Individual weighted, masked outputs in schedule order.
    configs : tuple of ResidualFilterConfig
        Settings used for each component, retained as static JAX metadata.
    """

    residual: jax.Array
    components: tuple[jax.Array, ...]
    configs: tuple[ResidualFilterConfig, ...] = field(metadata={"static": True})


def apply_residual_filter(
    residual: jax.Array,
    config: ResidualFilterConfig,
    *,
    mask: jax.Array | None = None,
) -> jax.Array:
    """Apply a weighted residual filter over the last two detector axes.

    Gaussian filters use periodic boundaries. The mask is applied after filtering,
    not before it, so mask invalid input samples before calling when necessary.

    Parameters
    ----------
    residual : jax.Array
        Residual array with at least two dimensions, ending in (nv, nu).
    config : ResidualFilterConfig
        Filter family, width and multiplicative weight.
    mask : jax.Array or None, optional
        Same-shaped multiplicative output mask; None leaves the output unmasked.

    Returns
    -------
    jax.Array
        Float32 filtered residual with the input shape.

    Raises
    ------
    ValueError
        If dimensions, filter name or mask shape are unsupported.
    """
    arr = jnp.asarray(residual, dtype=jnp.float32)
    if arr.ndim < 2:
        raise ValueError("residual must have at least two detector dimensions")

    if config.kind == "raw":
        filtered = arr
    elif config.kind == "lowpass_gaussian":
        filtered = _gaussian_lowpass(arr, sigma_px=config.sigma_px)
    elif config.kind == "bandpass_difference_of_gaussians":
        inner = _gaussian_lowpass(arr, sigma_px=config.sigma_px)
        outer = _gaussian_lowpass(arr, sigma_px=config.outer_sigma_px)
        filtered = inner - outer
    else:
        raise ValueError(f"unknown residual filter kind: {config.kind}")

    filtered = jnp.asarray(config.weight, dtype=jnp.float32) * filtered
    if mask is None:
        return filtered
    mask_arr = jnp.asarray(mask, dtype=jnp.float32)
    if mask_arr.shape != arr.shape:
        raise ValueError("mask must match residual shape")
    return filtered * mask_arr


def apply_residual_filter_schedule(
    residual: jax.Array,
    configs: tuple[ResidualFilterConfig, ...],
    *,
    mask: jax.Array | None = None,
) -> ResidualFilterResult:
    """Apply and sum a deterministic residual-filter schedule.

    Parameters
    ----------
    residual : jax.Array
        Residual array ending in the detector axes (nv, nu).
    configs : tuple of ResidualFilterConfig
        Nonempty sequence of weighted filter settings.
    mask : jax.Array or None, optional
        Same-shaped multiplicative mask applied to every filtered output.

    Returns
    -------
    ResidualFilterResult
        Summed residual, individual components and their settings.

    Raises
    ------
    ValueError
        If the schedule is empty or any filter input is unsupported.
    """
    if not configs:
        raise ValueError("at least one residual filter config is required")
    components = tuple(apply_residual_filter(residual, config, mask=mask) for config in configs)
    total = jnp.sum(jnp.stack(components, axis=0), axis=0)
    return ResidualFilterResult(residual=total, components=components, configs=configs)


def _gaussian_lowpass(residual: jax.Array, *, sigma_px: float) -> jax.Array:
    sigma = max(float(sigma_px), 1e-6)
    radius = max(1, math.ceil(2.0 * sigma))
    offsets = range(-radius, radius + 1)
    kernel = _gaussian_kernel(radius=radius, sigma_px=sigma)
    # The 2D Gaussian is an outer product. Two 1D passes preserve the periodic
    # boundaries while avoiding a quadratic number of weighted image shifts.
    for axis in (-2, -1):
        filtered = jnp.zeros_like(residual, dtype=jnp.float32)
        for index, offset in enumerate(offsets):
            filtered = filtered + kernel[index] * jnp.roll(residual, shift=offset, axis=axis)
        residual = filtered
    return residual


def _gaussian_kernel(*, radius: int, sigma_px: float) -> jax.Array:
    offsets = jnp.arange(-radius, radius + 1, dtype=jnp.float32)
    kernel = jnp.exp(-0.5 * (offsets / jnp.asarray(sigma_px, dtype=jnp.float32)) ** 2)
    return kernel / jnp.sum(kernel)
