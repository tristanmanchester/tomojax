"""Measurement noise remains separate from smooth object structure."""

import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.alignment._joint._noise import estimate_noise


def test_fourth_difference_rejects_polynomial_texture_without_changing_noise_scale():
    rng = np.random.default_rng(51)
    shape, sigma = (32, 80, 88), 0.013
    noise = rng.normal(0, sigma, shape)
    u = np.linspace(-1, 1, shape[2])[None, None]
    v = np.linspace(-1, 1, shape[1])[None, :, None]
    texture = 3 + 2 * u - 0.7 * u**3 + 40 * u**2 * v**3 + 2 * v**2
    plain = estimate_noise(noise)
    structured = estimate_noise(noise + texture)
    assert abs(plain / sigma - 1) < 0.025
    np.testing.assert_allclose(structured, plain, atol=2e-14)


@pytest.mark.parametrize("shape", [(40, 1, 80), (40, 80, 1), (40, 2, 75), (40, 4, 3)])
def test_short_and_singleton_axes_keep_unit_noise_calibration(shape):
    sigma = 0.04
    noise = np.random.default_rng(52).normal(0, sigma, shape)
    assert abs(estimate_noise(noise) / sigma - 1) < 0.08


@pytest.mark.parametrize("missing_columns", [False, True])
def test_masked_stencils_use_available_observations(missing_columns):
    sigma = 0.03
    noise = np.random.default_rng(53).normal(0, sigma, (32, 60, 65))
    precision = np.ones_like(noise)
    precision[:, ::3] = 0
    if missing_columns:
        precision[:, :, ::3] = 0
    noise[precision == 0] = np.nan
    estimated = estimate_noise(noise, precision=precision)
    assert abs(estimated / sigma - 1) < 0.06


def test_noise_estimator_indexes_bounded_samples_before_reading_a_large_array():
    class VirtualArray:
        shape = (1000, 800, 900)

        def __init__(self):
            self.reads = []

        def __getitem__(self, coordinates):
            self.reads.append(len(coordinates[0]))
            view, row, column = coordinates
            return np.sin(view * 1.45 + row * 2.67 + column * 3.21)

    source = VirtualArray()
    assert np.isfinite(estimate_noise(source))
    assert len(source.reads) == 25
    assert max(source.reads) <= 65_536


def test_unobserved_or_single_pixel_detector_has_no_spatial_noise_statistic():
    with pytest.raises(ValueError, match="observed detector contrast"):
        estimate_noise(np.ones((5, 1, 1)))
    with pytest.raises(ValueError, match="observed detector contrast"):
        estimate_noise(np.ones((5, 10, 10)), precision=0)


def test_sample_dtype_matches_rounding_the_whole_observation_stack():
    samples = np.random.default_rng(54).normal(size=(3, 12, 13))
    assert estimate_noise(samples, dtype=np.float32) == estimate_noise(samples.astype(np.float32))
