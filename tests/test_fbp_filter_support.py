"""Retain the full output convolution with known centred input support."""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from tomojax.geometry import Detector, Grid, ParallelGeometry
from tomojax.recon import FBPConfig, fbp
from tomojax.recon.filters import fft_filter_rows, rfft_filter_array


def _spatial_ramp(distance, pitch):
    values = np.zeros(distance.shape, np.float64)
    values[distance == 0] = 0.25 / pitch
    odd = distance % 2 != 0
    values[odd] = -1 / (np.pi**2 * distance[odd] ** 2 * pitch)
    return values


def _spatial_filter(distance, pitch, filter):
    if filter == "shepp-logan":
        return 2 / (np.pi**2 * pitch * (1 - 4 * distance.astype(np.float64) ** 2))
    kernel = _spatial_ramp(distance, pitch)
    if filter == "hann":
        kernel = 0.5 * kernel + 0.25 * (
            _spatial_ramp(distance - 1, pitch) + _spatial_ramp(distance + 1, pitch)
        )
    return kernel


@pytest.mark.parametrize("distance", [0, 1, 2, 7, 31])
def test_shepp_logan_impulse_matches_independent_nyquist_frequency_integral(distance):
    # Integrate the declared |f| * sinc(f*du) response, not the FFT implementation.
    pitch = 0.7
    integral = (
        2
        * quad(
            lambda f: f * np.sinc(f * pitch) * np.cos(2 * np.pi * f * distance * pitch),
            0,
            0.5 / pitch,
            epsabs=1e-12,
        )[0]
        * pitch
    )
    actual = _spatial_filter(np.asarray(distance), pitch, "shepp-logan")
    assert actual == pytest.approx(integral, abs=1e-12)


@pytest.mark.parametrize("raw_width", [1, 2, 17, 18, 31, 32, 33])
@pytest.mark.parametrize("padding", [0, 12, 15, 31, 40, 200])
@pytest.mark.parametrize("filter", ["ramp", "hann", "shepp-logan"])
def test_short_spectrum_matches_independent_whole_output_convolution(raw_width, padding, filter):
    width, pitch = raw_width + 2 * padding, 0.7
    random = np.random.default_rng(519 + raw_width + padding).normal(size=(2, raw_width))
    source = np.vstack([random, np.eye(1, raw_width, 0), np.eye(1, raw_width, raw_width - 1)])
    source = source.astype(np.float32)
    rows = np.pad(source, ((0, 0), (padding, padding)))
    distance = np.arange(width)[:, None] - (padding + np.arange(raw_width)[None, :])
    kernel = _spatial_filter(distance, pitch, filter)
    truth = source.astype(np.float64) @ kernel.T
    spectrum = rfft_filter_array(filter, width, pitch, jnp.float32, source_nu=raw_width)
    actual = np.asarray(fft_filter_rows(jnp.asarray(rows), spectrum))
    assert 2 * (spectrum.size - 1) >= width + raw_width
    assert np.linalg.norm(actual - truth) / np.linalg.norm(truth) < 2e-6


@pytest.mark.parametrize(
    ("raw_width", "padding"), [(1, 0), (1, 200), (17, 12), (18, 15), (31, 200), (32, 40), (33, 31)]
)
@pytest.mark.parametrize("filter", ["ramp", "hann", "shepp-logan"])
def test_short_filter_pullback_matches_independent_dense_transpose(raw_width, padding, filter):
    width, pitch = raw_width + 2 * padding, 0.7
    distance = np.arange(width)[:, None] - (padding + np.arange(raw_width)[None, :])
    kernel = _spatial_filter(distance, pitch, filter)
    cotangent = np.random.default_rng(37 + width).normal(size=(2, width)).astype(np.float32)
    spectrum = rfft_filter_array(filter, width, pitch, jnp.float32, source_nu=raw_width)

    def apply(source):
        padded = jnp.pad(source, ((0, 0), (padding, padding)))
        return fft_filter_rows(padded, spectrum)

    _, pullback = jax.vjp(apply, jnp.zeros((2, raw_width), jnp.float32))
    actual = np.asarray(pullback(jnp.asarray(cotangent))[0])
    truth = cotangent.astype(np.float64) @ kernel
    assert np.linalg.norm(actual - truth) / np.linalg.norm(truth) < 2e-6


@pytest.mark.parametrize("filter", ["ramp", "hann", "shepp-logan"])
def test_full_support_keeps_original_spectrum(filter):
    original = rfft_filter_array(filter, 95, 1.4, jnp.float32)
    full = rfft_filter_array(filter, 95, 1.4, jnp.float32, source_nu=95)
    np.testing.assert_array_equal(full, original)
    small = rfft_filter_array(filter, 95, 1.4, jnp.float32, source_nu=33)
    assert small.shape[0] < original.shape[0]
    alternate = rfft_filter_array(" " + filter.upper() + " ", 95, 1.4, jnp.float32, source_nu=33)
    np.testing.assert_array_equal(small, alternate)


@pytest.mark.parametrize("separable", [False, True])
@pytest.mark.parametrize(("nu", "nv", "nx"), [(1536, 22, 1536), (2560, 64, 2560), (2560, 64, 3072)])
def test_support_sizing_preserves_existing_view_batch_bound(monkeypatch, separable, nu, nv, nx):
    implementation = importlib.import_module("tomojax.recon.fbp")
    views = 720
    grid, detector = Grid(nx, nx, nv, 1.0, 1.0, 1.0), Detector(nu, nv, 1.0, 1.0)
    geometry = ParallelGeometry(grid, detector, np.linspace(0, 180, views, endpoint=False))
    monkeypatch.setattr(
        implementation,
        "_view_weights",
        lambda *args: (
            np.ones(views, np.float32),
            np.zeros((views, 6), np.float32),
            np.pi,
            separable,
        ),
    )
    # Only source metadata is read here, not a large real projection stack.
    plan = implementation._plan(
        geometry,
        grid,
        detector,
        jnp.broadcast_to(jnp.eye(4), (views, 4, 4)),
        np.empty((0, nv, nu), np.float32),
        FBPConfig(),
        cuda=True,
    )
    original_length = implementation._fft_length(plan.detector.nu)
    rows = nv * (1 if separable else implementation._fft_length(nv))
    expected_batch = implementation._parallel_filter_batch_size(views, rows, original_length)
    assert plan.batch == expected_batch
    actual_length = 2 * (plan.spectrum.size - 1)
    assert actual_length <= original_length
    if separable and nx != 2560:
        assert actual_length < original_length
    else:
        assert actual_length == original_length


@pytest.mark.gpu
@pytest.mark.parametrize("storage", ["resident", "streamed"])
@pytest.mark.parametrize("filter", ["ramp", "hann", "shepp-logan"])
@pytest.mark.parametrize("sampling", ["regular", "sparse"])
def test_wide_fbp_recovers_independent_off_axis_gaussian(monkeypatch, storage, filter, sampling):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    sparse = sampling == "sparse"
    views, rows, pitch, origin_z, sigma_z = (
        (75, 64, 4.0, -29.29, 14) if sparse else (180, 16, 1.0, -5.29, 6)
    )
    angles = np.linspace(5, 185, views, endpoint=False)
    theta = np.deg2rad(angles)
    detector = Detector(128, rows, 1.0, 1.0, (-7.3, 0.21))
    grid = Grid(192, 192, 8, 1.0, 1.0, pitch, vol_origin=(-95.5, -95.5, origin_z))
    implementation = importlib.import_module("tomojax.recon.fbp")
    pack = implementation._pack_fixed_rows
    calls = []

    # This spy observes Python tracing, not execution of cached JAX programs.
    # Ramp and Hann share spectrum shapes, so each case needs a fresh trace.
    implementation._run_fbp_streamed.clear_cache()
    implementation._fbp_accumulate_batch.clear_cache()

    def record(*args, **kwargs):
        calls.append(args[1].shape[1])
        return pack(*args, **kwargs)

    monkeypatch.setattr(implementation, "_pack_fixed_rows", record)
    geometry = ParallelGeometry(grid, detector, angles)
    u, v = np.arange(128) - 63.5 - 7.3, np.arange(rows) - (rows - 1) / 2 + 0.21
    variance = 25 * np.cos(theta) ** 2 + 49 * np.sin(theta) ** 2
    mean = 10 * np.cos(theta) + 7 * np.sin(theta)
    profile = 0.04 * np.sqrt(2 * np.pi) * 5 * 7 / np.sqrt(variance[:, None])
    profile = profile * np.exp(-0.5 * (u - mean[:, None]) ** 2 / variance[:, None])
    data = (profile[:, None, :] * np.exp(-0.5 * ((v - 0.7) / sigma_z) ** 2)[None, :, None]).astype(
        np.float32
    )
    actual = np.asarray(
        fbp(
            geometry,
            grid,
            detector,
            jnp.asarray(data) if storage == "resident" else data,
            config=FBPConfig(filter=filter, backprojector="pallas"),
        )
    )
    x, z = np.arange(192) - 95.5, origin_z + np.arange(8) * pitch
    truth = 0.04 * np.exp(
        -0.5
        * (
            (x[:, None, None] - 10) ** 2 / 25
            + (x[None, :, None] + 7) ** 2 / 49
            + (z[None, None, :] - 0.7) ** 2 / sigma_z**2
        )
    )
    assert np.linalg.norm(actual - truth) / np.linalg.norm(truth) < 0.02
    if sparse:
        # The shortened transform alone drops below the packing threshold;
        # retain the original decision rather than refilter all64 source rows.
        assert rows in calls


@pytest.mark.parametrize("contrast", [0.01, 0.001])
@pytest.mark.parametrize("filter", ["ramp", "hann", "shepp-logan"])
@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
def test_short_filter_preserves_weak_density_on_wide_shifted_grid(contrast, filter, backend):
    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    from tomojax.recon import FBPHostConfig, fbp_host

    angles = np.linspace(5, 185, 180, endpoint=False)
    theta = np.deg2rad(angles)
    detector = Detector(128, 9, 1.0, 1.0, (-7.3, 0.17))
    grid = Grid(192, 184, 5, 0.9, 1.1, 1.3, vol_origin=(-85.95, -100.65, -2.05))
    geometry = ParallelGeometry(grid, detector, angles)
    u = np.arange(detector.nu) - (detector.nu - 1) / 2 + detector.center[0]
    v = np.arange(detector.nv) - (detector.nv - 1) / 2 + detector.center[1]
    x = np.arange(grid.nx) * grid.vx - 85.95
    y = np.arange(grid.ny) * grid.vy - 100.65
    z = np.arange(grid.nz) * grid.vz - 2.05

    def gaussian(center, sigma):
        cx, cy, cz = center
        sx, sy, sz = sigma
        variance = sx**2 * np.cos(theta) ** 2 + sy**2 * np.sin(theta) ** 2
        mean = cx * np.cos(theta) - cy * np.sin(theta)
        profile = (
            0.04
            * np.sqrt(2 * np.pi)
            * sx
            * sy
            / np.sqrt(variance[:, None, None])
            * np.exp(-0.5 * (u[None, None, :] - mean[:, None, None]) ** 2 / variance[:, None, None])
            * np.exp(-0.5 * ((v[None, :, None] - cz) / sz) ** 2)
        )
        truth = 0.04 * np.exp(
            -0.5
            * (
                ((x[:, None, None] - cx) / sx) ** 2
                + ((y[None, :, None] - cy) / sy) ** 2
                + ((z[None, None, :] - cz) / sz) ** 2
            )
        )
        return profile, truth

    background, background_truth = gaussian((10, -7, 0.7), (5, 7, 7))
    feature, feature_truth = gaussian((-15.66, 13.86, -0.62), (2.3, 1.7, 3.5))
    # The wide grid needs filtered tails beyond the acquired detector, while
    # the fractional z-grid exercises row blending and a partial final slab.
    implementation = importlib.import_module("tomojax.recon.fbp")
    padded = implementation._parallel_filter_detector(grid, detector)
    short = rfft_filter_array(filter, padded.nu, detector.du, jnp.float32, source_nu=detector.nu)
    full = rfft_filter_array(filter, padded.nu, detector.du, jnp.float32)
    assert short.size < full.size
    config = FBPHostConfig(
        slices_per_batch=3, views_per_batch=32, filter=filter, backprojector=backend
    )
    base = fbp_host(geometry, grid, detector, background.astype(np.float32), config=config)
    actual = fbp_host(
        geometry,
        grid,
        detector,
        (background + contrast * feature).astype(np.float32),
        config=config,
    )
    expected = background_truth + contrast * feature_truth
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.02
    signal = (actual.astype(np.float64) - base) / contrast
    assert np.linalg.norm(signal - feature_truth) / np.linalg.norm(feature_truth) < 0.14
    assert np.unravel_index(np.argmax(signal), signal.shape) == np.unravel_index(
        np.argmax(feature_truth), feature_truth.shape
    )
