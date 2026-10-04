"""Independent free-voxel data and a verifier that cannot hide pose errors."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "bench"))
    return importlib.import_module("public_alignment_benchmark")


def test_exact_single_voxel_tent_integrals(benchmark):
    integrator = importlib.import_module("voxel_truth").voxel_line_integrals
    field = np.ones((1, 1, 1))
    # A single coefficient is a product of three tent functions, not a box.
    axial = integrator(
        field,
        [[0, 0, 0], [0.25, 0, 0.5], [3, 0, 0]],
        [0, 1, 0],
        spacing=[1, 2, 1],
        origin=[0, 0, 0],
    )
    np.testing.assert_allclose(axial, [2, 0.75, 0], atol=1e-14)
    diagonal = integrator(field, [[0, 0, 0]], [1, 1, 1], spacing=[1, 1, 1], origin=[0, 0, 0])
    np.testing.assert_allclose(diagonal, [np.sqrt(3) / 2], atol=1e-14)
    reverse = integrator(field, [[0, 0, 0]], [-2, -2, -2], spacing=[1, 1, 1], origin=[0, 0, 0])
    np.testing.assert_allclose(reverse, diagonal, atol=1e-14)


def test_independent_integrals_obey_physical_scaling(benchmark):
    integrator = importlib.import_module("voxel_truth").voxel_line_integrals
    field = np.random.default_rng(14).normal(size=(4, 3, 5))
    bases = np.array([[0.1, 0.3, 0.7], [2, 1, 2], [9, 6, 8]])
    direction = np.array([0.3, 0.8, -0.6])
    spacing, origin = np.array([0.8, 1.2, 1.7]), np.array([-1.2, -0.7, -1.8])
    expected = integrator(field, bases, direction, spacing=spacing, origin=origin)
    scaled = integrator(field, bases * 7, direction, spacing=spacing * 7, origin=origin * 7)
    np.testing.assert_allclose(scaled, expected * 7, atol=1e-13)


def test_pose_gauge_is_one_shared_rigid_transform(benchmark):
    nominal = np.tile(np.eye(4), (5, 1, 1))
    angles = np.array([0, 31, 77, 92, 153])[:, None]
    nominal[:, :3, :3] = Rotation.from_euler("z", angles, degrees=True).as_matrix()
    truth = benchmark.physical_poses(nominal, np.zeros((5, 5)))
    gauge = np.eye(4)
    gauge[:3, :3] = Rotation.from_euler("xyz", [0.05, -0.03, 0.07]).as_matrix()
    gauge[:3, 3] = [0.2, -0.3, 0.1]
    estimate = truth @ np.linalg.inv(gauge)
    np.testing.assert_allclose(benchmark.gauge_alignment(estimate, truth), gauge, atol=1e-14)
    corrupted = estimate.copy()
    corrupted[2, :3, :3] = corrupted[2, :3, :3] @ Rotation.from_euler("z", 0.1).as_matrix()
    fitted = corrupted @ benchmark.gauge_alignment(corrupted, truth)
    assert np.linalg.norm(fitted[:, :3, :3] - truth[:, :3, :3]) > 0.1


@pytest.mark.parametrize("kind", ["parallel", "anisotropic", "lamino"])
def test_free_voxel_fixture_and_verifier_keep_errors_visible(
    benchmark, monkeypatch, tmp_path, kind
):
    monkeypatch.setattr(benchmark, "SIZE", 8)
    monkeypatch.setattr(benchmark, "VIEWS", 7)
    path = tmp_path / "case.npz"
    benchmark.generate_fixture(path, kind, False)
    case = benchmark.load_fixture(path)
    result = benchmark.verify(case["truth"], case["truth_params"], case)
    assert result["accepted"]
    assert result["volume_relative_l2"] < 2e-7
    # Neither attenuation fitting nor independent pose registration is allowed.
    assert not benchmark.verify(2 * case["truth"], case["truth_params"], case)["accepted"]
    wrong_pose = case["truth_params"].copy()
    wrong_pose[2, 1] += 0.02
    assert not benchmark.verify(case["truth"], wrong_pose, case)["accepted"]
    bad_volume = case["truth"].copy()
    bad_volume.flat[0] = np.nan
    assert not benchmark.verify(bad_volume, case["truth_params"], case)["accepted"]
