"""Phase retrieval uses the physical pitch of each detector axis."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.fft import fft2, ifft2, next_fast_len

import tomojax as tj
from tomojax.corrections import Paganin


def _step(pitches: tuple[float, float]) -> Paganin:
    return Paganin(300.0, 0.05, 20.0, pitches)


def _reference(values: np.ndarray, pitches: tuple[float, float]) -> np.ndarray:
    scale = 0.05 * (12.398419843320026e-10 / 20.0) * 300.0 / (4 * np.pi)
    pads = []
    for length, pitch in zip(values.shape[-2:], pitches[::-1], strict=True):
        guard = max(min(length // 2, 128), int(np.ceil(16 * np.sqrt(scale) / pitch)))
        extent = 1 if length == 1 else next_fast_len(length + 2 * guard, real=True)
        pads.append(((extent - length) // 2, extent - length - (extent - length) // 2))
    padded = np.pad(values.astype(np.float64), ((0, 0), *pads), mode="edge")
    ku = 2 * np.pi * np.fft.fftfreq(padded.shape[2], pitches[0])
    kv = 2 * np.pi * np.fft.fftfreq(padded.shape[1], pitches[1])
    filtered = ifft2(fft2(padded) / (1 + scale * (kv[:, None] ** 2 + ku[None, :] ** 2))).real
    return filtered[:, pads[0][0] : pads[0][0] + values.shape[1],
                    pads[1][0] : pads[1][0] + values.shape[2]]  # fmt: skip


@pytest.mark.parametrize("shape", [(2, 1, 17), (2, 17, 1), (2, 16, 21), (2, 17, 20)])
@pytest.mark.parametrize("pitches", [(0.7e-6, 1.3e-6), (1.3e-6, 0.7e-6)])
def test_rectangular_pitches_match_full_complex_fp64_filter(
    shape: tuple[int, ...], pitches: tuple[float, float]
) -> None:
    values = np.random.default_rng(891).uniform(0.05, 1.0, shape).astype(np.float32)
    actual = jax.jit(_step(pitches).apply)(jnp.asarray(values))
    np.testing.assert_allclose(actual, _reference(values, pitches), rtol=3e-6, atol=3e-7)


def test_rectangular_phase_filter_respects_detector_axis_exchange() -> None:
    values = np.random.default_rng(892).uniform(0.05, 1.0, (2, 17, 32)).astype(np.float32)
    pitches = (0.7e-6, 1.3e-6)
    actual = jax.jit(_step(pitches).apply)(jnp.asarray(values))
    swapped = jax.jit(_step(pitches[::-1]).apply)(jnp.asarray(values.swapaxes(1, 2)))
    np.testing.assert_allclose(actual, np.asarray(swapped).swapaxes(1, 2), rtol=3e-6, atol=3e-7)


def test_square_pitch_pair_preserves_scalar_outputs_and_saved_context() -> None:
    values = np.random.default_rng(893).uniform(0.05, 1.0, (2, 17, 21)).astype(np.float32)
    scalar = Paganin(300.0, 0.05, 20.0, 1e-6)
    pair = _step((1e-6, 1e-6))
    np.testing.assert_array_equal(scalar.apply(values), pair.apply(values))
    settings = json.loads(json.dumps(_step((0.7e-6, 1.3e-6)).record().settings))
    assert settings["pixel_size"] == [0.7e-6, 1.3e-6]
    restored = Paganin(**settings)
    assert restored.pixel_size == (0.7e-6, 1.3e-6)
    assert hash(restored) == hash(_step((0.7e-6, 1.3e-6)))


def test_rectangular_pitch_workspace_counts_both_filter_ranges() -> None:
    scalar = Paganin(300.0, 0.6, 20.0, 1e-6)
    fine_rows = Paganin(300.0, 0.6, 20.0, (1e-6, 0.25e-6))
    assert fine_rows.workspace_pixels(128, 512) > 3 * scalar.workspace_pixels(128, 512)
    # A constant singleton axis needs no padding regardless of its pitch.
    assert fine_rows.workspace_pixels(1, 512) == scalar.workspace_pixels(1, 512)


@pytest.mark.parametrize(
    "pitches", [(1e-6,), (1e-6, 2e-6, 3e-6), (0.0, 1e-6), (1e-6, -1e-6), (np.nan, 1e-6)]
)
def test_rectangular_pitch_needs_two_positive_finite_lengths(pitches: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="pixel_size"):
        Paganin(300.0, 0.05, 20.0, pitches)


@pytest.mark.parametrize("v_pitch", [0.5, 2.0])
@pytest.mark.parametrize("noisy", [False, True])
def test_rectangular_phase_retrieval_recovers_independent_3d_attenuation(
    v_pitch: float, noisy: bool
) -> None:
    columns, rows, views = 96, 65, 120
    theta = np.linspace(0, np.pi, views, endpoint=False)
    axis = np.arange(columns) - (columns - 1) / 2
    zaxis = (np.arange(rows) - (rows - 1) / 2) * v_pitch
    cx, cy, cz, sigma, sigma_z, amplitude = 10.0, -7.0, -1.5, 5.0, 3.0, 0.025
    u = axis[None, None, :] - (cx * np.cos(theta) - cy * np.sin(theta))[:, None, None]
    z = zaxis[None, :, None] - cz
    p = (
        amplitude
        * np.sqrt(2 * np.pi)
        * sigma
        * np.exp(-0.5 * (u / sigma) ** 2 - 0.5 * (z / sigma_z) ** 2)
    )
    truth = amplitude * np.exp(
        -0.5 * ((axis[:, None, None] - cx) ** 2 + (axis[None, :, None] - cy) ** 2) / sigma**2
        - 0.5 * ((zaxis[None, None, :] - cz) / sigma_z) ** 2
    )
    # Independent continuous 2D transport-of-intensity propagation.
    pu, pv = -u / sigma**2 * p, -z / sigma_z**2 * p
    laplacian = ((u**2 / sigma**4 - 1 / sigma**2) + (z**2 / sigma_z**4 - 1 / sigma_z**2)) * p
    alpha = 2.0
    transmission = np.exp(-p) * (1 - alpha * (pu**2 + pv**2 - laplacian))
    assert transmission.min() > 0
    counts = (
        np.random.default_rng(1451).poisson(transmission * 10000) if noisy else transmission * 10000
    ).astype(np.float32)
    geometry = tj.ParallelGeometry(
        tj.Grid(columns, columns, rows, 1.0, 1.0, v_pitch),
        tj.Detector(columns, rows, 1.0, v_pitch),
        np.rad2deg(theta),
    )
    frames = tj.Frames(counts, geometry, white_level=10000.0)
    pixel, energy, delta_beta = 1e-6, 20.0, 300.0
    distance = alpha * pixel**2 * 4 * np.pi / ((12.398419843320026e-10 / energy) * delta_beta)
    errors = []
    for pitches in (pixel, (pixel, pixel * v_pitch)):
        scan = frames.corrected(Paganin(delta_beta, distance, energy, pitches))
        volume = np.asarray(tj.reconstruct(scan).volume)
        errors.append(float(np.linalg.norm(volume - truth) / np.linalg.norm(truth)))
        if isinstance(pitches, tuple) and not noisy:
            assert np.linalg.norm(scan.projections - p) / np.linalg.norm(p) < 3e-5
    assert errors[1] < errors[0]
    if not noisy:
        assert errors[1] < 0.006


def test_rectangular_phase_reverse_derivative_matches_dense_transpose() -> None:
    pitches = (0.7e-6, 1.3e-6)
    shape = (1, 3, 4)
    values = np.random.default_rng(894).uniform(0.05, 1.0, shape).astype(np.float32)
    cotangent = np.random.default_rng(895).normal(size=shape).astype(np.float32)
    basis = np.eye(values.size).reshape(values.size, *shape[1:])
    matrix = _reference(basis, pitches).reshape(values.size, values.size).T
    expected = (matrix.T @ cotangent.ravel()).reshape(shape)
    actual = jax.jit(jax.grad(lambda x: jnp.vdot(_step(pitches).apply(x), cotangent)))(values)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-6)
