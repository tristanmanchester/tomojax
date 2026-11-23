"""Rolled raw detectors measured against the complete calibrated public reconstruction."""

import jax
import numpy as np
import pytest

import tomojax as tj
from tomojax import recon


@pytest.mark.parametrize("method", ["fbp", "fourier", "sirt", "mlem"])
def test_rolled_raw_slabs_match_full_calibrated_reconstruction(method):
    grid = tj.Grid(12, 14, 17, 1, 1, 1)
    detector = tj.Detector(15, 17, 0.9, 1, (0.25, 0))
    geometry = tj.ParallelGeometry(
        grid, detector, np.linspace(0, 180, 19, endpoint=False), detector_roll=18
    )
    truth = np.zeros(grid.shape, np.float32)
    truth[2:6, 1:5, 7:10] = 0.18
    truth[11:14, 8:12, 2:5] = 0.09
    projections = np.maximum(np.asarray(tj.project(geometry, truth)), 0)
    frames = tj.Frames(1000 * np.exp(-projections), geometry, white_level=1000)
    config = {
        "fbp": recon.FBPConfig(),
        "fourier": recon.FourierConfig(slices_per_batch=3),
        "sirt": recon.SIRTConfig(iterations=3, backend="jax"),
        "mlem": recon.MLEMConfig(iterations=3, backend="jax"),
    }[method]
    full = np.asarray(tj.reconstruct(frames.corrected(), method, config=config).volume)
    with tj.reconstruct_slabs(frames, method, config=config) as slabs:
        parts = [result for _, result in slabs]
    actual = np.concatenate([part.volume for part in parts])
    assert np.linalg.norm(full - truth) < np.linalg.norm(truth)
    if jax.default_backend() == "cpu":
        np.testing.assert_array_equal(actual, full)
    else:
        np.testing.assert_allclose(actual, full, atol=8e-7, rtol=1e-5)
    if method == "fourier":
        assert all(part.scan.detector.nv < detector.nv for part in parts)
