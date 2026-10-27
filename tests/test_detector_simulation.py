"""Physical count response and persistent finite-reference error."""

import numpy as np
import pytest

from tomojax.datasets import DetectorModel


def test_static_response_cancels_and_changed_offset_is_physical():
    signal = np.array([[100.0, 200.0], [300.0, 500.0]])
    dark = np.arange(4.0).reshape(2, 2)
    model = DetectorModel(signal=signal, dark=dark, photon_gain=0, read_noise=0)
    q = np.arange(12.0).reshape(3, 2, 2) / 10
    exact = model.simulate(q, flat_frames=1, dark_frames=1, seed=12)
    np.testing.assert_allclose(exact.projections, q, atol=1e-15)
    changed = model.simulate(
        q, flat_frames=1, dark_frames=1, seed=12, gain_change=1.02, offset_change=2
    )
    truth = -np.log(1.02 * np.exp(-q) + 2 / signal)
    np.testing.assert_allclose(changed.projections, truth)
    np.testing.assert_array_equal(changed.expected, changed.ideal)


def test_finite_references_are_shared_but_sample_noise_is_not():
    model = DetectorModel(
        signal=np.full((4, 32), 10000.0), dark=np.zeros((4, 32)), photon_gain=2, read_noise=3
    )
    q = np.full((400, 4, 32), 0.5)
    first = model.simulate(q, flat_frames=4, dark_frames=3, seed=13)
    again = model.simulate(q, flat_frames=4, dark_frames=3, seed=13)
    np.testing.assert_array_equal(first.projections, again.projections)
    np.testing.assert_array_equal(first.expected[0], first.expected[-1])
    np.testing.assert_allclose(first.ideal, q)
    assert np.std(first.expected - q) > 0.005
    target_variance = 2 * 10000 * np.exp(-0.5) + 9
    assert np.var(first.counts) / target_variance == pytest.approx(1, rel=0.03)
    assert abs(np.corrcoef((first.expected - q)[0, 0], (first.expected - q)[0, 1])[0, 1]) < 0.5


def test_calibration_and_single_dark_requires_explicit_noise():
    rng = np.random.default_rng(14)
    flat = 2 * rng.poisson(1000, size=(100, 8, 16)) + rng.normal(10, 3, (100, 8, 16))
    dark = rng.normal(10, 3, (100, 8, 16))
    model = DetectorModel.from_flats(flat, dark)
    assert model.photon_gain == pytest.approx(2, rel=0.05)
    assert model.read_noise == pytest.approx(3, rel=0.05)
    with pytest.raises(ValueError, match="two darks"):
        DetectorModel.from_flats(flat, dark[:1])
    assert DetectorModel.from_flats(flat, dark[:1], read_noise=3).read_noise == 3
