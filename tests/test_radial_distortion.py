"""A calibrated detector-coordinate pullback before the transmission log."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.ndimage import map_coordinates

import tomojax as tj
from tomojax.corrections import RadialDistortion

if TYPE_CHECKING:
    from pathlib import Path


def _frames(transmission: np.ndarray) -> tj.Frames:
    views, rows, cols = transmission.shape
    geometry = tj.ParallelGeometry(
        tj.Grid(cols, cols, rows, 1.0, 1.0, 1.0),
        tj.Detector(cols, rows, 1.0, 1.0),
        np.linspace(0, 180, views, endpoint=False),
    )
    return tj.Frames(transmission * 1000, geometry, white_level=1000.0)


def _reference(images: np.ndarray, step: RadialDistortion) -> np.ndarray:
    v, u = np.indices(images.shape[-2:], dtype=np.float64)
    du, dv = u - step.center[0], v - step.center[1]
    r = np.hypot(du, dv)
    # Independent power-series evaluation, including an off-image centre.
    scale = sum(c * r**k for k, c in enumerate(step.coefficients))
    source = np.asarray((step.center[1] + dv * scale, step.center[0] + du * scale), np.float32)
    return np.stack([map_coordinates(image, source, order=1, mode="nearest") for image in images])


@pytest.mark.parametrize("coefficients", [(1.0,), (1.0, 0.0, 0.0)])
def test_identity_keeps_every_view_and_pixel(coefficients: tuple[float, ...]) -> None:
    data = np.random.default_rng(5).uniform(0.1, 1, (3, 7, 11)).astype(np.float32)
    step = RadialDistortion(center=(-2.3, 0.7), coefficients=coefficients)
    actual = jax.jit(step.apply)(jnp.asarray(data))
    np.testing.assert_array_equal(actual, data)
    assert step.workspace_pixels(7, 11) == 0


def test_large_pixel_coordinates_keep_subpixel_calibration_without_global_x64() -> None:
    data = np.random.default_rng(4).uniform(0.1, 1, (2, 64, 1024)).astype(np.float32)
    step = RadialDistortion(center=(900.25, 720.75), coefficients=(1.001, -3e-6, 2e-8, -1e-11))
    with jax.enable_x64(False):
        actual = jax.jit(step.apply)(jnp.asarray(data))
        assert not jax.config.x64_enabled
    assert actual.dtype == jnp.float32
    np.testing.assert_allclose(actual, _reference(data, step), rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("center", [(3.2, 4.7), (-5.3, 9.4)])
@pytest.mark.parametrize("coefficients", [(0.85, 0.005, -0.0001), (1.2, -0.002)])
def test_bilinear_pullback_matches_scipy_and_nearest_edges(
    center: tuple[float, float], coefficients: tuple[float, ...]
) -> None:
    data = np.random.default_rng(12).uniform(0.1, 1, (3, 9, 13)).astype(np.float32)
    step = RadialDistortion(center=center, coefficients=coefficients)
    actual = jax.jit(step.apply)(jnp.asarray(data))
    np.testing.assert_allclose(actual, _reference(data, step), rtol=2e-6, atol=2e-7)


def test_distortion_runs_on_transmission_before_log_and_is_recorded(tmp_path: Path) -> None:
    data = np.random.default_rng(1).uniform(0.2, 1, (5, 8, 11)).astype(np.float32)
    step = RadialDistortion(center=[4.2, 5.8], coefficients=[0.9, 0.01])
    scan = _frames(data).corrected(step, batch_views=2)
    expected = -np.log(_reference(data, step))
    np.testing.assert_allclose(scan.projections, expected, rtol=2e-5, atol=5e-7)
    assert [record.name for record in scan.corrections] == ["flat_dark", "radial_distortion", "log"]
    assert scan.corrections[1].settings == {"center": [4.2, 5.8], "coefficients": [0.9, 0.01]}
    tj.save(tmp_path / "corrected.nxs", scan)
    assert tj.load(tmp_path / "corrected.nxs").corrections == scan.corrections
    with pytest.raises(ValueError, match="transmission"):
        scan.corrected(step)


def test_crop_translation_retains_calibration_with_source_halo() -> None:
    data = np.random.default_rng(1).uniform(0.2, 1, (2, 20, 24)).astype(np.float32)
    step = RadialDistortion(center=(12.3, 9.1), coefficients=(0.98, 0.001))
    whole = np.asarray(jax.jit(step.apply)(jnp.asarray(data)))
    cropped = data[:, 3:17, 4:20]
    shifted = replace(step, center=(step.center[0] - 4, step.center[1] - 3))
    actual = np.asarray(jax.jit(shifted.apply)(jnp.asarray(cropped)))
    # Ignore the crop boundary, where a source halo rather than edge extension is needed.
    np.testing.assert_allclose(actual[:, 1:-1, 1:-1], whole[:, 4:16, 5:19], rtol=2e-6, atol=4e-7)


def test_binned_calibration_uses_pixel_centres_and_radius_units() -> None:
    factor = 4
    center = (33.2, 21.7)
    coefficients = (0.94, 0.0002, 0.00001)
    step = RadialDistortion(
        center=tuple((c - (factor - 1) / 2) / factor for c in center),
        coefficients=tuple(c * factor**k for k, c in enumerate(coefficients)),
    )
    # An affine image is integrated exactly by mean binning and bilinear sampling.
    v, u = np.indices((12, 16), dtype=np.float64)
    data = (0.3 + 0.001 * (factor * u + 1.5) + 0.002 * (factor * v + 1.5))[None]
    du, dv = factor * u + 1.5 - center[0], factor * v + 1.5 - center[1]
    r = np.hypot(du, dv)
    scale = sum(c * r**k for k, c in enumerate(coefficients))
    expected = 0.3 + 0.001 * (center[0] + du * scale) + 0.002 * (center[1] + dv * scale)
    actual = np.asarray(jax.jit(step.apply)(jnp.asarray(data, jnp.float32)))
    np.testing.assert_allclose(actual[0, 1:-1, 1:-1], expected[1:-1, 1:-1], rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("center", [(2.3, 3.1), (-4.2, 7.9)])
def test_image_directional_derivative_matches_independent_pullback(
    center: tuple[float, float],
) -> None:
    rng = np.random.default_rng(31)
    values = rng.uniform(0.1, 1.0, (3, 5, 7)).astype(np.float32)
    direction = rng.normal(size=values.shape).astype(np.float32)
    step = RadialDistortion(center=center, coefficients=(1.2, -0.001))
    _, actual = jax.jit(lambda x, d: jax.jvp(step.apply, (x,), (d,)))(
        jnp.asarray(values), jnp.asarray(direction)
    )
    # The calibration is fixed: the intensity pullback is linear, including
    # repeated nearest edges, and each view has its own independent image.
    np.testing.assert_allclose(actual, _reference(direction, step), rtol=3e-6, atol=5e-7)


@pytest.mark.parametrize("center", [(2.3, 3.1), (-4.2, 7.9)])
def test_image_reverse_derivative_matches_dense_transpose(
    center: tuple[float, float],
) -> None:
    rng = np.random.default_rng(32)
    values = rng.uniform(0.1, 1.0, (2, 5, 7)).astype(np.float32)
    cotangent = rng.normal(size=values.shape).astype(np.float32)
    step = RadialDistortion(center=center, coefficients=(1.2, -0.001))
    pixels = values.shape[-2] * values.shape[-1]
    basis = np.eye(pixels, dtype=np.float32).reshape(pixels, *values.shape[-2:])
    forward = _reference(basis, step).reshape(pixels, pixels).T
    expected = (cotangent.reshape(len(values), pixels).astype(np.float64) @ forward).reshape(
        values.shape
    )
    actual = jax.jit(jax.grad(lambda x: jnp.vdot(step.apply(x), cotangent)))(jnp.asarray(values))
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=5e-7)


@pytest.mark.parametrize(
    ("center", "coefficients"),
    [((0,), (1,)), ((0, np.nan), (1,)), ((0, 0), ()), ((0, 0), (1, np.inf))],
)
def test_invalid_calibrations_are_refused(
    center: tuple[float, ...], coefficients: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError, match="RadialDistortion"):
        RadialDistortion(center=center, coefficients=coefficients)


def _raw_gaussian_components(
    spacing: float,
    components: tuple[tuple[float, float, float, float, float, float], ...],
) -> tuple[tj.Frames, RadialDistortion, np.ndarray]:
    """Measured camera counts and continuous density, without TomoJAX's projector."""
    width, height, views = 96, 48, 180
    optical_center = (30.2, 31.1)
    coefficient = 5e-5
    v, u = np.indices((height, width), dtype=np.float64)
    dx, dy = u - optical_center[0], v - optical_center[1]
    measured_radius = np.hypot(dx, dy)
    ideal_radius = measured_radius.copy()
    # Invert rho = r + c2*r**3 to generate the *measured* raw camera image.
    for _ in range(12):
        ideal_radius -= (ideal_radius + coefficient * ideal_radius**3 - measured_radius) / (
            1.0 + 3.0 * coefficient * ideal_radius**2
        )
    ratio = np.divide(
        ideal_radius, measured_radius, out=np.ones_like(ideal_radius), where=measured_radius > 0
    )
    ideal_u = optical_center[0] + ratio * dx
    ideal_v = optical_center[1] + ratio * dy
    axis = -2.3 * spacing
    detector_u = (ideal_u - (width - 1) / 2) * spacing + axis
    detector_v = (ideal_v - (height - 1) / 2) * spacing
    angles = np.arange(views, dtype=np.float64)
    radians = np.deg2rad(angles)
    x, y, z = np.meshgrid(
        (np.arange(48) - 23.5) * spacing,
        (np.arange(48) - 23.5) * spacing,
        (np.arange(16) - 7.5) * spacing,
        indexing="ij",
    )
    line_integrals = np.zeros((views, height, width), dtype=np.float64)
    density = np.zeros(x.shape, dtype=np.float64)
    for amplitude, width_xy, width_z, cx, cy, cz in components:
        sigma, sigma_z = width_xy * spacing, width_z * spacing
        x0, y0, z0 = cx * spacing, cy * spacing, cz * spacing
        shift = x0 * np.cos(radians) - y0 * np.sin(radians)
        line_integrals += (
            amplitude
            * np.sqrt(2 * np.pi)
            * sigma
            * np.exp(
                -((detector_u[None] - shift[:, None, None]) ** 2) / (2 * sigma**2)
                - (detector_v[None] - z0) ** 2 / (2 * sigma_z**2)
            )
        )
        density += amplitude * np.exp(
            -((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2) - (z - z0) ** 2 / (2 * sigma_z**2)
        )
    # Nonuniform raw calibration must be removed before the optical pullback.
    flat = 8000 * (1.0 + 0.25 * np.cos(u / 17) + 0.15 * np.sin(v / 13))
    dark = 100 + 0.6 * u + 0.2 * v
    counts = (dark + (flat - dark) * np.exp(-line_integrals)).astype(np.float32)
    grid = tj.Grid(48, 48, 16, spacing, spacing, spacing)
    detector = tj.Detector(width, height, spacing, spacing, (axis, 0.0))
    frames = tj.Frames(
        counts,
        tj.ParallelGeometry(grid, detector, angles),
        flats=flat[None].astype(np.float32),
        darks=dark[None].astype(np.float32),
    )
    step = RadialDistortion(center=optical_center, coefficients=(1.0, 0.0, coefficient))
    return frames, step, density


@pytest.mark.parametrize("spacing", [0.7, 1.3])
@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
def test_calibrated_raw_gaussian_recovers_independent_density(spacing: float, backend: str) -> None:
    """Exact continuous line integrals, not a forward/backprojector round trip."""
    from tomojax.recon import FBPConfig

    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires a CUDA GPU")
    frames, step, expected = _raw_gaussian_components(spacing, ((0.035, 7, 3, 4, -5, -1),))
    config = FBPConfig(backprojector=backend)
    actual = np.asarray(tj.reconstruct(frames.corrected(step), "fbp", config=config).volume)
    uncorrected = np.asarray(tj.reconstruct(frames.corrected(), "fbp", config=config).volume)
    norm = np.linalg.norm(expected)
    # Bilinear transmission sampling slightly broadens the narrow axial Gaussian.
    assert np.linalg.norm(actual - expected) / norm < 0.02
    assert np.linalg.norm(uncorrected - expected) / norm > 0.025


@pytest.mark.parametrize("contrast", [0.01, 0.001])
@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
def test_raw_lens_pipeline_preserves_weak_density_signal(contrast: float, backend: str) -> None:
    """A small raw-data feature must survive even when background error is low."""
    from tomojax.recon import FBPConfig

    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires a CUDA GPU")
    broad = (0.035, 7.0, 3.0, 4.0, -5.0, -1.0)
    bead = (broad[0] * contrast, 2.0, 1.5, -10.5, 8.5, 1.5)
    background, step, background_density = _raw_gaussian_components(1.0, (broad,))
    combined, _, combined_density = _raw_gaussian_components(1.0, (broad, bead))
    config = FBPConfig(backprojector=backend)
    # A non-divisor view batch also exercises the correction tail.
    volumes = [
        np.asarray(
            tj.reconstruct(frames.corrected(step, batch_views=19), "fbp", config=config).volume
        )
        for frames in (background, combined)
    ]
    signal, expected = volumes[1] - volumes[0], combined_density - background_density
    # Difference-signal error cannot hide behind the much larger background.
    assert np.linalg.norm(signal - expected) / np.linalg.norm(expected) < 0.08
    assert np.unravel_index(np.argmax(signal), signal.shape) == np.unravel_index(
        np.argmax(expected), expected.shape
    )
