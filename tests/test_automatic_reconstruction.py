"""Automatic reconstruction selects a measured inverse without changing geometry."""

from dataclasses import replace

import numpy as np
import pytest

import tomojax as tj
from tomojax.recon import FBPConfig


@pytest.mark.parametrize(("complete", "raw"), [(True, False), (False, False), (True, True)])
def test_default_reconstruction_matches_explicit_inverse_and_reports_coverage(
    complete, raw, tmp_path
):
    grid = tj.Grid(12, 14, 2, 1, 1, 1)
    detector = tj.Detector(20, 2, 1, 1, center=(1.3, 0))
    angles = np.linspace(0, 180 if complete else 90, 32, endpoint=False)
    geometry = tj.ParallelGeometry(grid, detector, angles)
    u = np.arange(detector.nu) - (detector.nu - 1) / 2 + detector.center[0]
    data = np.broadcast_to(np.exp(-(u**2) / 10), (32, 2, 20)).astype(np.float32).copy()
    scan = (
        tj.Frames(np.exp(-data), geometry, white_level=1).corrected()
        if raw
        else tj.Scan(data, geometry)
    )
    expected = tj.reconstruct(scan, "fourier" if complete else "fbp")
    actual = tj.reconstruct(scan)
    np.testing.assert_array_equal(actual.volume, expected.volume)
    assert actual.scan is scan and actual.method == expected.method
    limited = raw or not complete or actual.info.get("backend") == "numpy"
    assert actual.info["automatic"]["status"] == ("limited" if limited else "ok")
    assert bool(actual.info["automatic"]["reasons"]) == limited
    if raw:
        reasons = actual.info["automatic"]["reasons"]
        assert any("stripe correction" in reason for reason in reasons)
        assert any("centre calibration" in reason for reason in reasons)
    tj.save(tmp_path / "reconstruction.nxs", actual)
    loaded = tj.load_reconstruction(tmp_path / "reconstruction.nxs")
    assert loaded.info["automatic"] == actual.info["automatic"]


def test_explicit_method_bypasses_automatic_geometry_policy():
    grid = tj.Grid(4, 4, 2, 1, 1, 1)
    detector = tj.Detector(8, 3, 1, 1)
    geometry = tj.LaminographyGeometry(grid, detector, np.arange(0, 360, 30), tilt_deg=30)
    scan = tj.Scan(np.ones((12, 3, 8), np.float32), geometry)
    result = tj.reconstruct(scan, "fbp", config=FBPConfig())
    assert "automatic" not in result.info and result.scan is scan
    with pytest.raises(ValueError, match="choose method='fista'"):
        tj.reconstruct(scan, grid=replace(grid, nx=3))
