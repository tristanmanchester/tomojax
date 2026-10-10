"""Projection residual filters match independent periodic convolutions."""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.ndimage import convolve1d

from tomojax.forward.api import (
    ResidualFilterConfig,
    apply_residual_filter,
    apply_residual_filter_schedule,
)

pytestmark = pytest.mark.numerical


def _reference_filter(residual: np.ndarray, config: ResidualFilterConfig) -> np.ndarray:
    def gaussian(sigma: float) -> np.ndarray:
        sigma = max(sigma, 1e-6)
        radius = max(1, math.ceil(2.0 * sigma))
        kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
        kernel /= kernel.sum()
        rows = convolve1d(residual.astype(np.float64), kernel, axis=-2, mode="wrap")
        return convolve1d(rows, kernel, axis=-1, mode="wrap")

    if config.kind == "raw":
        return config.weight * residual
    inner = gaussian(config.sigma_px)
    if config.kind == "lowpass_gaussian":
        return config.weight * inner
    return config.weight * (inner - gaussian(config.outer_sigma_px))


@pytest.mark.parametrize("shape", [(7, 8), (2, 9, 7), (2, 3, 1, 5)])
@pytest.mark.parametrize("sigma", [0.2, 1.0, 2.3, 4.0])
@pytest.mark.parametrize("kind", ["raw", "lowpass_gaussian", "bandpass_difference_of_gaussians"])
def test_residual_filters_match_periodic_reference(shape, sigma, kind) -> None:
    residual = np.random.default_rng(73).normal(size=shape).astype(np.float32)
    config = ResidualFilterConfig(kind=kind, sigma_px=sigma, outer_sigma_px=2 * sigma, weight=0.7)
    operation = jax.jit(functools.partial(apply_residual_filter, config=config))
    actual = operation(jnp.asarray(residual))
    np.testing.assert_allclose(actual, _reference_filter(residual, config), rtol=2e-5, atol=5e-7)


@pytest.mark.parametrize("kind", ["lowpass_gaussian", "bandpass_difference_of_gaussians"])
def test_masked_residual_filter_gradient_matches_reference(kind) -> None:
    rng = np.random.default_rng(38)
    residual = rng.normal(size=(2, 9, 7)).astype(np.float32)
    mask = rng.integers(0, 2, size=residual.shape).astype(np.float32)
    config = ResidualFilterConfig(kind=kind, sigma_px=1.3, outer_sigma_px=2.6, weight=0.7)

    def loss(x: jax.Array) -> jax.Array:
        filtered = apply_residual_filter(x, config, mask=jnp.asarray(mask))
        return 0.5 * jnp.vdot(filtered, filtered)

    # Each unmasked periodic filter is symmetric; the mask is applied after it.
    filtered = _reference_filter(residual, config)
    expected = _reference_filter(mask**2 * filtered, config)
    actual = jax.jit(jax.grad(loss))(jnp.asarray(residual))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=5e-7)


def test_residual_filter_schedule_preserves_components_and_mask() -> None:
    residual = np.random.default_rng(4).normal(size=(3, 11, 13)).astype(np.float32)
    mask = np.zeros_like(residual)
    mask[:, 1:-1, 2:-2] = 1
    configs = (
        ResidualFilterConfig(kind="raw", weight=0.2),
        ResidualFilterConfig(kind="lowpass_gaussian", sigma_px=2.3, weight=0.8),
    )
    actual = apply_residual_filter_schedule(jnp.asarray(residual), configs, mask=jnp.asarray(mask))
    expected = [_reference_filter(residual, config) * mask for config in configs]
    for component, reference in zip(actual.components, expected, strict=True):
        np.testing.assert_allclose(component, reference, rtol=2e-5, atol=5e-7)
    np.testing.assert_allclose(actual.residual, sum(expected), rtol=2e-5, atol=5e-7)


def test_residual_filter_result_supports_compilation_batching_and_tree_roundtrip() -> None:
    residual = jnp.asarray(np.random.default_rng(51).normal(size=(3, 5, 7)), jnp.float32)
    configs = (
        ResidualFilterConfig(kind="raw", weight=0.2),
        ResidualFilterConfig(kind="lowpass_gaussian", sigma_px=1.3, weight=0.8),
    )
    operation = functools.partial(apply_residual_filter_schedule, configs=configs)
    actual = jax.jit(jax.vmap(operation))(residual)
    reference = operation(residual)
    assert actual.configs == configs
    np.testing.assert_allclose(actual.residual, reference.residual, rtol=2e-5, atol=5e-7)
    assert len(actual.components) == len(configs)
    leaves, tree = jax.tree.flatten(actual)
    assert len(leaves) == 1 + len(configs)
    restored = jax.tree.unflatten(tree, leaves)
    assert restored.configs == configs
    np.testing.assert_array_equal(restored.residual, actual.residual)
