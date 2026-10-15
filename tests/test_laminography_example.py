"""The runnable recipe keeps validation separate and preserves conservative fallback."""

from types import SimpleNamespace

from examples import align_laminography as example
import numpy as np
import pytest

import tomojax as tj


def test_uncertain_recipe_keeps_projection_seed(monkeypatch):
    grid = tj.Grid(8, 8, 8, 1, 1, 1)
    detector = tj.Detector(12, 12, 1, 1)
    angles = np.linspace(0, 360, 32, endpoint=False)
    geometry = tj.LaminographyGeometry(grid, detector, angles, tilt_deg=30.9)
    xyz = np.indices((8, 8, 8)).astype(float)
    volume = np.exp(-np.sum((xyz - 3.5) ** 2, axis=0) / 3).astype(np.float32)
    scan = tj.Scan(np.asarray(tj.project(geometry, volume)), geometry)
    monkeypatch.setattr(
        tj,
        "estimate_region",
        lambda s: SimpleNamespace(scan=s, grid=s.grid, info={"status": "broad_uncertain"}),
        raising=False,
    )
    result, info = example.reconstruct_lamino(scan, nominal_tilt_deg=30.9)
    assert np.isfinite(result.volume).all()
    assert result.scan.projections is scan.projections
    assert result.scan.poses.shape == (32, 6)
    assert not info["accepted"] and info["internal"] == []


@pytest.mark.parametrize("improves", [False, True])
def test_recipe_excludes_validation_and_replaces_selected_poses(monkeypatch, improves):
    n = 25
    angles = np.arange(n, dtype=float)
    geometry = tj.LaminographyGeometry(
        tj.Grid(4, 4, 4, 1, 1, 1), tj.Detector(4, 4, 1, 1), angles, tilt_deg=30.9
    )
    data = np.random.default_rng(3).uniform(0.2, 1, (n, 4, 4)).astype(np.float32)
    scan = tj.Scan(data, geometry)
    held = np.arange(n) % 5 == 2
    modes = []

    def align(s, *, mode, **kwargs):
        modes.append(mode)
        np.testing.assert_array_equal(s.angles, angles[~held])
        poses = np.zeros((len(s.angles), 6), np.float32)
        poses[:, 3] = float(mode == "pose")
        return SimpleNamespace(scan=s.with_poses(poses), poses=poses)

    def reconstruct(s, *args, **kwargs):
        return SimpleNamespace(scan=s, volume=np.full((4, 4, 4), s.poses[0, 3]))

    def project(g, volume):
        np.testing.assert_array_equal(g.angles, angles[held])
        error = 0.1 if volume[0, 0, 0] and improves else 0.2
        return data[held] * (1 + error)

    monkeypatch.setattr(tj, "align", align)
    monkeypatch.setattr(
        tj,
        "estimate_region",
        lambda s: SimpleNamespace(scan=s, grid=s.grid, info={"status": "conditional_cv_region"}),
        raising=False,
    )
    monkeypatch.setattr(tj, "reconstruct", reconstruct)
    monkeypatch.setattr(tj, "project", project)
    result, info = example.reconstruct_lamino(scan, nominal_tilt_deg=30.9)
    assert modes == ["full", "pose"]
    assert info["accepted"] is improves
    np.testing.assert_array_equal(result.scan.poses[:, 3], float(improves))
