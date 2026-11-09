"""The migrated watch example scores physical detector rays, including the centre."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import tomojax as tj


@pytest.fixture
def example(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("align_watch")


def arrangement():
    views = 60
    grid = tj.Grid(24, 24, 24, 2.0, 2.0, 2.0)
    detector = tj.Detector(48, 48, 1.0, 1.0)
    poses = np.tile(np.eye(4), (views, 1, 1))
    angles = np.linspace(0, 360, views, endpoint=False)
    poses[:, :3, :3] = Rotation.from_euler("z", angles[:, None], degrees=True).as_matrix()
    return grid, detector, poses


def test_one_common_rigid_gauge_has_zero_detector_error(example):
    grid, detector, truth = arrangement()
    gauge = np.eye(4)
    gauge[:3, :3] = Rotation.from_rotvec([0.01, -0.02, 0.03]).as_matrix()
    gauge[:3, 3] = [0.4, -0.3, 0.2]
    estimated = truth @ np.linalg.inv(gauge)
    adjusted, fitted, error = example.comparison(estimated, truth, detector, detector, grid)
    assert error < 1e-10
    np.testing.assert_allclose(fitted, gauge, atol=2e-10)
    np.testing.assert_allclose(adjusted, estimated, atol=2e-10)


def test_detector_centre_counts_and_matches_its_equivalent_translation(example):
    grid, detector, truth = arrangement()
    true_detector = tj.Detector(48, 48, 1.0, 1.0, center=(2.3, 0.0))
    equivalent = truth.copy()
    equivalent[:, 0, 3] -= 2.3
    _, _, corrected = example.comparison(equivalent, truth, detector, true_detector, grid)
    _, _, uncorrected = example.comparison(truth, truth, detector, true_detector, grid)
    assert corrected < 1e-10
    assert uncorrected == pytest.approx(2.3, abs=1e-10)


def test_truth_registration_uses_the_same_physical_transform(example):
    grid, _, _ = arrangement()
    truth = np.arange(24**3, dtype=float).reshape(24, 24, 24)
    gauge = np.eye(4)
    np.testing.assert_array_equal(example.moved_truth(truth, grid, gauge), truth)
    gauge[0, 3] = grid.vx
    expected = np.zeros_like(truth)
    expected[:, :, 1:] = truth[:, :, :-1]
    np.testing.assert_array_equal(example.moved_truth(truth, grid, gauge), expected)
