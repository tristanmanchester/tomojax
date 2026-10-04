"""Validate independent ray and photon-count helpers without installing gVXR."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def generator():
    path = Path(__file__).resolve().parents[1] / "bench" / "gvxr_dataset.py"
    spec = importlib.util.spec_from_file_location("gvxr_dataset_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_box_chords_check_physical_length_parallel_misses_and_direction_scale(generator):
    base = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.1, 0.0, 0.0], [0.0, -10.0, 0.0]])
    direction = np.array([0.0, 2.0, 0.0])
    result = generator.box_chords(base, direction, np.zeros(3), np.array([4.0, 6.0, 8.0]))
    np.testing.assert_allclose(result, [6.0, 6.0, 0.0, 6.0])
    chord = generator.box_chords(
        np.zeros(3), np.array([1.0, 1.0, 0.0]), np.zeros(3), np.array([4.0, 6.0, 8.0])
    )
    np.testing.assert_allclose(chord, 4 * np.sqrt(2))


def test_photon_normalization_and_explicit_zero_floor(generator):
    data = 1e5 * np.exp(-np.array([0.0, 0.2, 2.0]))
    np.testing.assert_allclose(generator.log_attenuation(data, 1e5), [0.0, 0.2, 2.0], atol=1e-14)
    assert generator.log_attenuation(np.array([0.0]), 100.0, floor_photons=0.5)[0] == np.log(200.0)
    for invalid in [np.array([0.0]), np.array([-1.0]), np.array([np.nan])]:
        with pytest.raises(ValueError):
            generator.log_attenuation(invalid, 100.0)
    with pytest.raises(ValueError):
        generator.log_attenuation(data, 0.0)


def test_material_chord_composition_preserves_translation_and_mm_units(generator):
    materials = [
        {"center_mm": [0.0, 0.0, 0.0], "extent_mm": [4.0, 20.0, 8.0], "mu_mm_inverse": [0.02] * 5}
    ]
    camera = np.array([[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    pose = np.eye(4)
    result = generator.reference_for_view(camera, pose, materials)
    np.testing.assert_allclose(result, [[[0.4] * 5, [0.0] * 5]])
    pose[0, 3] = 3.0
    shifted = generator.reference_for_view(camera, pose, materials)
    np.testing.assert_allclose(shifted, [[[0.0] * 5, [0.4] * 5]])


def test_tilted_pose_maps_camera_rays_with_a_proper_rotation(generator):
    r = generator.rotation(37.0, 30.0)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-15)
    np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-15)
    assert abs(r[1, 2]) > 0.1
