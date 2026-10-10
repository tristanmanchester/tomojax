"""Walnut quality comparisons use matching physical voxel centres."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def walnut():
    path = Path(__file__).resolve().parents[1] / "bench" / "walnut.py"
    spec = importlib.util.spec_from_file_location("walnut_benchmark_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _linear_field(shape):
    coordinates = np.meshgrid(*(np.linspace(-25, 25, n) for n in shape), indexing="ij")
    return 2 + 0.01 * coordinates[0] + 0.02 * coordinates[1] - 0.03 * coordinates[2]


@pytest.mark.parametrize("shape", [(5, 5, 5), (6, 6, 6), (7, 5, 3)])
def test_quality_samples_the_same_voxel_centres_at_integer_and_fractional_densities(walnut, shape):
    reference = _linear_field((11, 11, 11))
    actual = _linear_field(shape)
    with np.errstate(divide="ignore"):
        metrics = walnut.compare(actual, reference)
    assert metrics["relative_l2"] < 1e-15
    assert metrics["relative_l2_in_walnut"] < 1e-15


def test_quality_keeps_amplitude_errors_instead_of_renormalizing_coarse_images(walnut):
    reference = np.ones((11, 11, 11))
    actual = np.full((6, 6, 6), 2.0)
    metrics = walnut.compare(actual, reference)
    assert metrics["relative_l2"] == 1.0
    assert metrics["relative_l2_in_walnut"] == 1.0
    assert metrics["psnr_db"] == 0.0


def test_quality_preserves_the_matching_resolution_path_exactly(walnut):
    reference = _linear_field((7, 7, 7))
    actual = reference + 0.1
    expected = np.linalg.norm(actual - reference) / np.linalg.norm(reference)
    metrics = walnut.compare(actual, reference)
    assert metrics["relative_l2"] == expected


@pytest.mark.parametrize("density", [1, 2, 3, 5, 10])
def test_volume_geometries_share_the_published_reference_voxel_centre_extent(walnut, density):
    geometry = walnut.volume_geometry(density)
    for axis, count_name in [("X", "GridColCount"), ("Y", "GridRowCount"), ("Z", "GridSliceCount")]:
        lower = geometry["option"][f"WindowMin{axis}"]
        upper = geometry["option"][f"WindowMax{axis}"]
        spacing = (upper - lower) / geometry[count_name]
        assert lower + 0.5 * spacing == pytest.approx(-25.0)
        assert upper - 0.5 * spacing == pytest.approx(25.0)


def test_matching_projection_cache_reuses_the_decoded_request(walnut, tmp_path, monkeypatch):
    cache = tmp_path / "projections.npz"
    monkeypatch.setattr(
        walnut,
        "load_orbit",
        lambda _path, orbit, every: (np.full((2, 3, 4), orbit), np.full((3, 12), every)),
    )
    expected = walnut.load_orbits(tmp_path, [2], 20, cache=cache)

    def unexpected_read(*_args):
        pytest.fail("matching cache reread the source projections")

    monkeypatch.setattr(walnut, "load_orbit", unexpected_read)
    actual = walnut.load_orbits(tmp_path, [2], 20, cache=cache)
    for found, wanted in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(found, wanted)


@pytest.mark.parametrize("changed", ["walnut", "orbits", "every", "legacy"])
def test_projection_cache_cannot_silently_replace_a_different_requested_scan(
    walnut, tmp_path, monkeypatch, changed
):
    cache = tmp_path / "projections.npz"
    monkeypatch.setattr(
        walnut,
        "load_orbit",
        lambda _path, orbit, every: (np.full((2, 3, 4), orbit), np.full((3, 12), every)),
    )
    walnut.load_orbits(tmp_path, [2], 20, cache=cache)
    if changed == "legacy":
        np.savez(cache, data=np.zeros((2, 3, 4)), vectors=np.zeros((3, 12)))
    source = tmp_path / "another-walnut" if changed == "walnut" else tmp_path
    orbits = [1, 2] if changed == "orbits" else [2]
    every = 10 if changed == "every" else 20
    before = cache.read_bytes()
    with pytest.raises(ValueError, match="projection cache does not match"):
        walnut.load_orbits(source, orbits, every, cache=cache)
    assert cache.read_bytes() == before
