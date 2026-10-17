"""The shared measured-data challenge must not vary between recipes or batches."""

from bench.nanotomo_motion import injected_motion, shift_complex_objects
import numpy as np


def test_motion_components_reproducible_and_acquisition_ordered():
    angles = np.tile(np.linspace(0, 180, 1000), 10)
    first = injected_motion(angles, seed=1)
    again = injected_motion(angles, seed=1)
    np.testing.assert_array_equal(first.shifts_uv, again.shifts_uv)
    np.testing.assert_array_equal(first.sub_tomogram, np.repeat(np.arange(10), 1000))
    np.testing.assert_allclose(np.max(abs(first.drift_uv), axis=0), 8)
    assert np.max(abs(first.offset_uv)) <= 3
    for group in range(10):
        offsets = first.offset_uv[first.sub_tomogram == group]
        np.testing.assert_array_equal(offsets, np.broadcast_to(offsets[0], offsets.shape))
    np.testing.assert_allclose(first.jitter_uv.std(axis=0), 0.5, atol=0.02)
    assert abs(np.corrcoef(first.jitter_uv.T)[0, 1]) < 0.03
    assert not np.array_equal(first.shifts_uv, injected_motion(angles, seed=2).shifts_uv)


def test_fourier_motion_sign_axes_analytic_wave_and_batch_invariance():
    v, u = np.meshgrid(np.arange(17), np.arange(24), indexing="ij")
    frame = np.exp(2j * np.pi * (3 * v / 17 + 5 * u / 24))
    objects = np.tile(frame.astype(np.complex64), (3, 1, 1))
    shifts = np.array([[0, 0], [2, -3], [-1.3, 0.42]])
    shifted = shift_complex_objects(objects, shifts)
    expected = (
        objects
        * np.exp(-2j * np.pi * (3 * shifts[:, 1] / 17 + 5 * shifts[:, 0] / 24))[:, None, None]
    )
    np.testing.assert_allclose(shifted, expected, atol=5e-7)
    np.testing.assert_allclose(shifted[1], np.roll(objects[1], (-3, 2), axis=(0, 1)), atol=5e-7)
    np.testing.assert_array_equal(shifted[:1], shift_complex_objects(objects[:1], shifts[:1]))
    np.testing.assert_allclose(shift_complex_objects(shifted, -shifts), objects, atol=7e-7)
    assert shifted.dtype == objects.dtype
