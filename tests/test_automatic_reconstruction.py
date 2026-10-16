"""Automatic reconstruction selects a measured inverse without changing geometry."""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

import tomojax as tj
from tomojax.alignment import api as alignment
from tomojax.io import ProjectionDataset
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
        assert actual.info["automatic"]["center_calibration_reason"] in reasons
    else:
        assert actual.info["automatic"]["center_calibration_reason"] is None
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


@pytest.mark.parametrize(
    "status", [None, "recorded", "provided", "inferred", "not_recorded", "failed"]
)
def test_automatic_center_provenance_failure_and_saved_reuse(status, monkeypatch, tmp_path):
    grid, detector = tj.Grid(8, 8, 2, 1, 1, 1), tj.Detector(12, 2, 1, 1)
    angles = np.linspace(0, 180, 33)
    geometry = tj.ParallelGeometry(grid, detector, angles)
    scan = tj.Frames(np.full((33, 2, 12), 0.5, np.float32), geometry, white_level=1).corrected()
    meta = {} if status is None else {"detector_center_status": status}
    if status == "failed":
        meta["detector_center_status"] = "not_recorded"
    record = ProjectionDataset(scan.projections, angles, geometry_metadata=meta)
    scan = replace(scan, source=record)
    estimate = Mock(return_value=replace(geometry, detector=replace(detector, center=(-2.2, 0))))
    if status == "failed":
        estimate.side_effect = ValueError("no usable half-turn")
    monkeypatch.setattr(alignment, "estimate_center", estimate)
    tj.reconstruct(scan, "fourier")
    assert estimate.call_count == 0
    actual = tj.reconstruct(scan)
    assert estimate.call_count == int(status in {"not_recorded", "failed"})
    assert record.geometry_metadata == meta and scan.detector.center == (0, 0)
    automatic = actual.info["automatic"]
    if status == "not_recorded":
        assert actual.scan.detector.center == (-2.2, 0)
        assert actual.scan.source.geometry_metadata["detector_center_status"] == "inferred"
        assert automatic["center_calibration_reason"] is None
        path = tmp_path / "calibrated.nxs"
        tj.save(path, actual)
        loaded = tj.load_reconstruction(path)
        replay = tj.reconstruct(loaded.scan)
        np.testing.assert_array_equal(actual.volume, replay.volume)
        assert estimate.call_count == 1 and replay.scan is loaded.scan
    else:
        assert actual.scan is scan
        if status == "failed":
            assert automatic["center_estimation"]["status"] == "failed"
            assert automatic["status"] == "limited"
            assert automatic["center_calibration_reason"] in automatic["reasons"]
