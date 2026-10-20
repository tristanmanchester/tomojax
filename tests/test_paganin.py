"""Real-input phase retrieval against a full-complex float64 host reference."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.fft import fft2, ifft2

from tomojax.corrections import Paganin


def _reference(values: np.ndarray, step: Paganin, guard: int = 512) -> np.ndarray:
    """The filter on views extended by ``guard`` repeated-edge pixels, far beyond its range.

    A one-pixel axis stays unpadded: its edge padding is constant, so only its
    zero frequency would contribute, exactly as without the padding.
    """
    _, rows, cols = values.shape
    guard_v, guard_u = (guard if rows > 1 else 0), (guard if cols > 1 else 0)
    padded = np.pad(
        values.astype(np.float64), ((0, 0), (guard_v, guard_v), (guard_u, guard_u)), mode="edge"
    )
    kv = 2 * np.pi * np.fft.fftfreq(padded.shape[1], step.pixel_size)
    ku = 2 * np.pi * np.fft.fftfreq(padded.shape[2], step.pixel_size)
    wavelength = 12.398419843320026e-10 / step.energy_kev
    scale = step.distance * wavelength * step.delta_beta / (4 * np.pi)
    response = 1 / (1 + scale * (kv[:, None] ** 2 + ku[None, :] ** 2))
    spectrum = np.asarray(fft2(padded), dtype=np.complex128)
    filtered = np.asarray(ifft2(spectrum * response), dtype=np.complex128).real
    return filtered[:, guard_v : guard_v + rows, guard_u : guard_u + cols]


@pytest.mark.parametrize("shape", [(1, 1, 33), (1, 33, 1)])
def test_paganin_broad_filter_does_not_wrap_opposite_detector_edges(
    shape: tuple[int, ...],
) -> None:
    # About 30 px decay length on a 33 px detector: half-width padding lets
    # transmission from the opposite edge change a pixel by more than 0.1.
    step = Paganin(delta_beta=300.0, distance=0.6, energy_kev=20.0, pixel_size=1e-6)
    values = np.where(np.arange(33) < 16, 0.1, 0.9).astype(np.float32).reshape(shape)
    actual = jax.jit(step.apply)(jnp.asarray(values))
    np.testing.assert_allclose(actual, _reference(values, step, 4096), rtol=3e-6, atol=1e-5)


@pytest.mark.parametrize(
    "shape", [(3, 1, 17), (3, 17, 1), (3, 16, 20), (3, 17, 21), (3, 16, 21), (3, 17, 20)]
)
@pytest.mark.parametrize("distance", [0.001, 0.05, 0.5])
def test_paganin_matches_full_complex_reference(shape: tuple[int, ...], distance: float) -> None:
    step = Paganin(delta_beta=300.0, distance=distance, energy_kev=20.0, pixel_size=1e-6)
    values = np.random.default_rng(51).uniform(0.05, 1.0, shape).astype(np.float32)
    actual = jax.jit(step.apply)(jnp.asarray(values))
    assert actual.shape == values.shape and actual.dtype == values.dtype
    # Short filters on tiny views have a slowly decaying sampled kernel: 1e-4 is the guard's.
    np.testing.assert_allclose(actual, _reference(values, step), rtol=3e-6, atol=1e-4)


@pytest.mark.parametrize("shape", [(2, 1, 1), (2, 5, 7), (2, 6, 8)])
def test_paganin_preserves_a_constant_without_mixing_views(shape: tuple[int, ...]) -> None:
    step = Paganin(delta_beta=300.0, distance=0.05, energy_kev=20.0, pixel_size=1e-6)
    values = np.broadcast_to(np.asarray([0.3, 0.8], np.float32)[:, None, None], shape)
    actual = jax.jit(step.apply)(jnp.asarray(values))
    np.testing.assert_allclose(actual, values, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("shape", [(2, 5, 7), (2, 6, 8), (2, 5, 8)])
def test_paganin_directional_derivative_matches_reference(shape: tuple[int, ...]) -> None:
    step = Paganin(delta_beta=300.0, distance=0.05, energy_kev=20.0, pixel_size=1e-6)
    rng = np.random.default_rng(52)
    values = rng.uniform(0.05, 1.0, shape).astype(np.float32)
    direction = rng.normal(size=shape).astype(np.float32)
    _, actual = jax.jit(lambda x, d: jax.jvp(step.apply, (x,), (d,)))(
        jnp.asarray(values), jnp.asarray(direction)
    )
    # Phase retrieval is linear in the transmission, including its edge padding.
    np.testing.assert_allclose(actual, _reference(direction, step), rtol=3e-6, atol=1e-5)


@pytest.mark.parametrize("shape", [(1, 1, 7), (1, 5, 1), (1, 5, 7), (1, 6, 8)])
def test_paganin_reverse_derivative_matches_dense_transpose(shape: tuple[int, ...]) -> None:
    step = Paganin(delta_beta=300.0, distance=0.05, energy_kev=20.0, pixel_size=1e-6)
    values = np.random.default_rng(53).uniform(0.05, 1.0, shape).astype(np.float32)
    cotangent = np.random.default_rng(54).normal(size=shape).astype(np.float32)
    basis = np.eye(values.size).reshape(values.size, *shape[1:])
    # One padded view at a time: the dense matrix is small, its padded basis is not.
    forward = np.stack([_reference(b[None], step).ravel() for b in basis]).T
    expected = (forward.T @ cotangent.ravel()).reshape(shape)
    actual = jax.jit(jax.grad(lambda x: jnp.vdot(step.apply(x), cotangent)))(jnp.asarray(values))
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-5)
