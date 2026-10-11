"""Real-input phase retrieval against a full-complex float64 host reference."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.fft import fft2, ifft2, next_fast_len

from tomojax.corrections import Paganin


def _reference(values: np.ndarray, step: Paganin, *, guard: int | None = None) -> np.ndarray:
    _, rows, cols = values.shape
    wavelength = 12.398419843320026e-10 / step.energy_kev
    scale = step.distance * wavelength * step.delta_beta / (4 * np.pi)
    decay_guard = int(np.ceil(16 * np.sqrt(scale) / step.pixel_size))
    guards = [max(min(n // 2, 128), decay_guard) if guard is None else guard for n in (rows, cols)]
    sizes = [
        1 if n == 1 else next_fast_len(n + 2 * p, real=True)
        for n, p in zip((rows, cols), guards, strict=True)
    ]
    pads = [((s - n) // 2, s - n - (s - n) // 2) for n, s in zip((rows, cols), sizes, strict=False)]
    pr, pc = pads[0][0], pads[1][0]
    padded = np.pad(values.astype(np.float64), ((0, 0), *pads), mode="edge")
    kv = 2 * np.pi * np.fft.fftfreq(padded.shape[1], step.pixel_size)
    ku = 2 * np.pi * np.fft.fftfreq(padded.shape[2], step.pixel_size)
    response = 1 / (1 + scale * (kv[:, None] ** 2 + ku[None, :] ** 2))
    spectrum = np.asarray(fft2(padded), dtype=np.complex128)
    filtered = np.asarray(ifft2(spectrum * response), dtype=np.complex128).real
    return filtered[:, pr : pr + rows, pc : pc + cols]


@pytest.mark.parametrize("shape", [(1, 1, 33), (1, 33, 1)])
def test_paganin_broad_filter_does_not_wrap_opposite_detector_edges(
    shape: tuple[int, ...],
) -> None:
    # About 30 px decay length on a 33 px detector: half-width padding lets
    # transmission from the opposite edge change a pixel by more than 0.1.
    step = Paganin(delta_beta=300.0, distance=0.6, energy_kev=20.0, pixel_size=1e-6)
    values = np.where(np.arange(33) < 16, 0.1, 0.9).astype(np.float32).reshape(shape)
    expected = _reference(values, step, guard=4096)
    actual = jax.jit(step.apply)(jnp.asarray(values))
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("length", [0.1, 0.5, 1.0, 3.0, 10.0, 30.0])
def test_paganin_guard_matches_long_edge_extension_for_a_wide_image(length: float) -> None:
    # Test the approximation independently of the production padding rule,
    # including short filters whose sampled spectrum has non-exponential tails.
    pixel, energy, delta_beta = 1e-6, 20.0, 300.0
    wavelength = 12.398419843320026e-10 / energy
    distance = (length * pixel) ** 2 * (4 * np.pi) / (wavelength * delta_beta)
    step = Paganin(delta_beta, distance, energy, pixel)
    values = np.random.default_rng(55).uniform(0.05, 1.0, (1, 1, 641)).astype(np.float32)
    expected = _reference(values, step, guard=4096)
    actual = jax.jit(step.apply)(jnp.asarray(values))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)


@pytest.mark.parametrize(
    "shape", [(3, 1, 17), (3, 17, 1), (3, 16, 20), (3, 17, 21), (3, 16, 21), (3, 17, 20)]
)
@pytest.mark.parametrize("distance", [0.001, 0.05, 0.5])
def test_paganin_matches_full_complex_reference(shape: tuple[int, ...], distance: float) -> None:
    step = Paganin(delta_beta=300.0, distance=distance, energy_kev=20.0, pixel_size=1e-6)
    values = np.random.default_rng(51).uniform(0.05, 1.0, shape).astype(np.float32)
    actual = jax.jit(step.apply)(jnp.asarray(values))
    assert actual.shape == values.shape and actual.dtype == values.dtype
    np.testing.assert_allclose(actual, _reference(values, step), rtol=3e-6, atol=3e-7)


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
    np.testing.assert_allclose(actual, _reference(direction, step), rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("shape", [(1, 1, 7), (1, 5, 1), (1, 5, 7), (1, 6, 8)])
def test_paganin_reverse_derivative_matches_dense_transpose(shape: tuple[int, ...]) -> None:
    step = Paganin(delta_beta=300.0, distance=0.05, energy_kev=20.0, pixel_size=1e-6)
    values = np.random.default_rng(53).uniform(0.05, 1.0, shape).astype(np.float32)
    cotangent = np.random.default_rng(54).normal(size=shape).astype(np.float32)
    basis = np.eye(values.size).reshape(values.size, *shape[1:])
    forward = _reference(basis, step).reshape(values.size, values.size).T
    expected = (forward.T @ cotangent.ravel()).reshape(shape)
    actual = jax.jit(jax.grad(lambda x: jnp.vdot(step.apply(x), cotangent)))(jnp.asarray(values))
    # The adjoint folds a much larger repeated-edge guard onto a few boundary
    # pixels. Its FP32 reductions incur more absolute roundoff than the FFT.
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-6)
