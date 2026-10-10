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
    found = orbit_heights(ConeSegments((first, second)), data)
    assert found is not None
    pixel_at_axis = detector.dv / first.beam.magnification
    assert abs(found.heights[1] - 0.6) < 0.5 * pixel_at_axis
    assert found.contrast[1] > 0.5
    assert orbit_heights(first, data[:72]) is None  # one orbit: nothing to place
