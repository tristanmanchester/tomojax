"""Geometry from data consistency alone: orbit heights of multi-orbit cone-beam scans."""

from __future__ import annotations

import numpy as np
import pytest

import tomojax as tj
from tomojax.alignment.api import orbit_heights
from tomojax.geometry import ConeBeam, ConeGeometry, ConeSegments, Detector, Grid
from tomojax.io import build_geometry_from_dataset_metadata

pytestmark = pytest.mark.numerical


def _raised(segment: ConeGeometry, height: float) -> object:
    """``segment`` with its source and detector ``height`` higher (its object lower)."""
    views = len(segment.angles)
    poses = np.zeros((views, 6), np.float32)
    poses[:, 4] = -height
    meta = {
        "detector": segment.detector.to_dict(),
        "grid": segment.grid.to_dict(),
        "angles": np.asarray(segment.angles, np.float32),
        "geometry_type": "cone",
        "cone_beam": segment.beam.to_dict(),
        "align_params": poses,
        "align_gauge": {"pose_translation_frame": "detector"},
    }
    return build_geometry_from_dataset_metadata(meta, poses=True)[2]


def test_orbit_heights_place_a_second_orbit_from_the_data_alone():
    n, vx = 48, 0.25
    grid = Grid(n, n, n, vx, vx, vx)
    detector = Detector(64, 64, 0.5, 0.5)
    angles = np.linspace(0.0, 360.0, 72, endpoint=False)
    first = ConeGeometry(grid, detector, angles, ConeBeam(30.0, 50.0))
    second = ConeGeometry(grid, detector, angles + 2.5, ConeBeam(30.0, 50.0))
    c = (np.arange(n) - (n - 1) / 2) * vx
    x, y, z = np.meshgrid(c, c, c, indexing="ij")
    rng = np.random.default_rng(0)
    volume = np.zeros((n, n, n), np.float32)
    for cx, cy, cz in rng.uniform(-3.0, 3.0, (12, 3)):  # inside the field: no truncation
        volume += np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / 0.8).astype(np.float32)
    true = ConeSegments((first, _raised(second, 0.6)))
    data = np.asarray(tj.project(true, volume))
    pixel_at_axis = detector.dv / first.beam.magnification
    baseline = None
    for intensity in (1.0, 1e-20, 1e20):
        found = orbit_heights(ConeSegments((first, second)), data * intensity)
        assert found is not None
        assert abs(found.heights[1] - 0.6) < 0.5 * pixel_at_axis
        assert found.contrast[1] > 0.5
        if baseline is None:
            baseline = found
        else:
            np.testing.assert_allclose(found.heights, baseline.heights, rtol=2e-6)
            np.testing.assert_allclose(found.contrast, baseline.contrast, rtol=2e-6)
    assert orbit_heights(first, data[:72]) is None  # one orbit: nothing to place


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_unobservable_orbit_keeps_its_height_with_zero_contrast(value):
    grid = Grid(8, 8, 8, 0.25, 0.25, 0.25)
    detector = Detector(8, 8, 0.5, 0.5)
    first = ConeGeometry(grid, detector, np.arange(0.0, 360.0, 90.0), ConeBeam(30.0, 50.0))
    second = ConeGeometry(grid, detector, np.arange(10.0, 360.0, 90.0), ConeBeam(30.0, 50.0))
    geometry = ConeSegments((first, second))
    data = np.full((8, 8, 8), value, np.float32)

    # Blank data and fully truncated constant data have no usable epipolar
    # derivatives. Neither justifies an arbitrary correction at the search edge.
    with np.errstate(divide="raise", invalid="raise"):
        found = orbit_heights(geometry, data, pairs=2)

    assert found is not None
    np.testing.assert_array_equal(found.heights, [0.0, 0.0])
    np.testing.assert_array_equal(found.contrast, [1.0, 0.0])
    assert found.planes == 0
