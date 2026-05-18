"""Projection-domain residual filters."""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import jax
import jax.numpy as jnp

ResidualFilterKind = Literal["raw", "lowpass_gaussian", "bandpass_difference_of_gaussians"]


@dataclass(frozen=True)
class ResidualFilterConfig:
    kind: ResidualFilterKind = "raw"
    weight: float = 1.0
    sigma_px: float = 1.0
    outer_sigma_px: float = 2.0


@dataclass(frozen=True)
class ResidualFilterResult:
    residual: jax.Array
    components: tuple[jax.Array, ...]
    configs: tuple[ResidualFilterConfig, ...]


def apply_residual_filter(
    residual: jax.Array,
    config: ResidualFilterConfig,
    *,
    mask: jax.Array | None = None,
) -> jax.Array:
    """Apply a named projection-domain residual filter."""
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
    """Apply and sum a deterministic residual-filter schedule."""
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
    filtered = jnp.zeros_like(residual, dtype=jnp.float32)
    for row_index, row_offset in enumerate(offsets):
        row_weight = kernel[row_index]
        row_shifted = jnp.roll(residual, shift=row_offset, axis=-2)
        for col_index, col_offset in enumerate(offsets):
            filtered = filtered + row_weight * kernel[col_index] * jnp.roll(
                row_shifted,
                shift=col_offset,
                axis=-1,
            )
    return filtered


def _gaussian_kernel(*, radius: int, sigma_px: float) -> jax.Array:
    offsets = jnp.arange(-radius, radius + 1, dtype=jnp.float32)
    kernel = jnp.exp(-0.5 * (offsets / jnp.asarray(sigma_px, dtype=jnp.float32)) ** 2)
    return kernel / jnp.sum(kernel)
