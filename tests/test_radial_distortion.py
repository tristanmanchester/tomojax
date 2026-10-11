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


def test_identity_keeps_every_view_and_pixel() -> None:
    data = np.random.default_rng(5).uniform(0.1, 1, (3, 7, 11)).astype(np.float32)
    step = RadialDistortion(center=(-2.3, 0.7), coefficients=(1.0,))
    actual = jax.jit(step.apply)(jnp.asarray(data))
    np.testing.assert_array_equal(actual, data)


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


@pytest.mark.parametrize(
    ("center", "coefficients"),
    [((0,), (1,)), ((0, np.nan), (1,)), ((0, 0), ()), ((0, 0), (1, np.inf))],
)
def test_invalid_calibrations_are_refused(
    center: tuple[float, ...], coefficients: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError, match="RadialDistortion"):
        RadialDistortion(center=center, coefficients=coefficients)
