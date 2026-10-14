"""The calibrated backward radial polynomial unwarps transmission before the log."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.ndimage import map_coordinates

import tomojax as tj
from tomojax.corrections import Correction, RadialDistortion

pytestmark = pytest.mark.surface


def _image(rows: int, cols: int) -> np.ndarray:
    v, u = np.mgrid[:rows, :cols]
    return 0.6 + 0.3 * np.sin(u / 7.0) * np.cos(v / 9.0)


def test_the_unwarped_pixel_samples_the_measured_image_at_the_polynomial_radius() -> None:
    rows, cols = 40, 56
    step = RadialDistortion(center=(27.5, 18.0), coefficients=(1.002, -3e-4, 2e-6))
    image = _image(rows, cols)
    v, u = np.mgrid[:rows, :cols].astype(np.float64)
    du, dv = u - 27.5, v - 18.0
    r = np.hypot(du, dv)
    scale = 1.002 - 3e-4 * r + 2e-6 * r**2
    expected = map_coordinates(
        image, [18.0 + scale * dv, 27.5 + scale * du], order=1, mode="nearest"
    )
    actual = np.asarray(step.apply(jnp.asarray(image[None], jnp.float32)))[0]
    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_the_identity_polynomial_changes_nothing() -> None:
    image = jnp.asarray(_image(30, 41)[None], jnp.float32)
    step = RadialDistortion(center=(13.2, 9.7), coefficients=(1.0,))
    np.testing.assert_allclose(step.apply(image), image, atol=1e-6)


def test_a_distortion_step_normalises_first_and_is_recorded() -> None:
    views, rows, cols = 3, 24, 32
    flat = 900.0 + np.arange(cols) * 2.0  # not described by the polynomial
    transmission = np.broadcast_to(_image(rows, cols), (views, rows, cols))
    counts = (transmission * flat).astype(np.float32)
    geometry = tj.ParallelGeometry(
        tj.Grid(cols, cols, rows, 1.0, 1.0, 1.0),
        tj.Detector(cols, rows, 1.0, 1.0),
        np.arange(views) * 60.0,
    )
    step = RadialDistortion(center=(15.5, 11.5), coefficients=(1.01, -2e-4))
    scan = tj.Frames(counts, geometry, flats=np.broadcast_to(flat, (1, rows, cols))).corrected(step)
    unwarped = step.apply(jnp.asarray(transmission, jnp.float32))
    np.testing.assert_allclose(scan.projections, -np.log(unwarped), rtol=2e-6, atol=2e-6)
    assert scan.corrections[1] == Correction(
        "radial_distortion", {"center": [15.5, 11.5], "coefficients": [1.01, -2e-4]}
    )
    with pytest.raises(ValueError, match="finite"):
        RadialDistortion(center=(np.nan, 0.0), coefficients=(1.0,))
