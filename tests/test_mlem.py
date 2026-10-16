"""Matched Poisson updates, angular subsets and reconstruction accuracy."""

from dataclasses import replace

import numpy as np
import pytest

import tomojax as tj
from tomojax.geometry import (
    ConeBeam,
    ConeGeometry,
    ConeSegments,
    Detector,
    Grid,
    LaminographyGeometry,
)
from tomojax.recon import MLEMConfig, mlem


@pytest.fixture(params=["lamino", "cone"])
def problem(request):
    grid = Grid(3, 2, 2, 0.8, 1.1, 1.3)
    detector = Detector(5, 4, 0.7, 1.2, (0.17, -0.23))
    angles = [83, 13, 161, 49, 122]
    geometry = LaminographyGeometry(grid, detector, angles, tilt_deg=23)
    if request.param == "cone":
        geometry = ConeSegments(
            (
                ConeGeometry(grid, detector, angles[:2], ConeBeam(12, 24)),
                ConeGeometry(
                    grid, replace(detector, center=(-0.2, 0.4)), angles[2:], ConeBeam(15, 27)
                ),
            )
        )
    basis = np.eye(12, dtype=np.float32).reshape(12, 3, 2, 2)
    # Assemble only forwards; NumPy's dense transpose supplies an independent
    # reference for all normalizations and multiplicative subset updates.
    matrix = np.stack([np.asarray(tj.project(geometry, x)).ravel() for x in basis], axis=1)
    return grid, detector, geometry, matrix.astype(np.float64)


@pytest.mark.numerical
@pytest.mark.parametrize("subsets", [1, 2, 3])
def test_mlem_dense_subset_updates_with_support(problem, subsets):
    grid, detector, geometry, matrix = problem
    support = np.ones((3, 2, 2), bool)
    support[0] = False
    truth = np.linspace(0.2, 1.1, 12) * support.ravel()
    data = matrix @ truth
    expected = np.full(12, 0.7) * support.ravel()
    order = np.argsort(geometry.angles, kind="stable")
    for _ in range(3):
        for group in range(subsets):
            rows = (order[group::subsets, None] * 20 + np.arange(20)).ravel()
            a = matrix[rows]
            den = a @ expected
            ratio = np.divide(data[rows], den, out=np.zeros_like(den), where=den > 0)
            sensitivity = a.sum(axis=0)
            expected *= np.divide(a.T @ ratio, sensitivity, out=np.ones(12), where=sensitivity > 0)
    volume, info = mlem(
        geometry,
        grid,
        detector,
        data.reshape(5, 4, 5),
        init_x=np.full((3, 2, 2), 0.7, np.float32),
        config=MLEMConfig(iterations=3, subsets=subsets, support=support, views_per_batch=2),
    )
    np.testing.assert_allclose(np.asarray(volume).ravel(), expected, rtol=2e-5, atol=2e-6)
    assert info["subset_updates"] == 3 * subsets


@pytest.mark.numerical
def test_mlem_public_workflow_improves_likelihood_and_image(problem):
    grid, detector, geometry, matrix = problem
    truth = np.linspace(0.2, 1.1, 12)
    data = matrix @ truth
    scan = tj.Scan(data.reshape(5, 4, 5).astype(np.float32), geometry)
    previous = np.ones(12)
    valid = data > 0

    def likelihood(p):
        return np.sum(p[valid] - data[valid] * np.log(p[valid]))

    for iterations in [1, 10, 100]:
        volume = np.asarray(tj.reconstruct(scan, "mlem", iterations=iterations).volume).ravel()
        old, new = matrix @ previous, matrix @ volume
        assert likelihood(new) <= likelihood(old) + 1e-5
        assert np.linalg.norm(volume - truth) < np.linalg.norm(previous - truth)
        previous = volume
    assert np.linalg.norm(previous - truth) / np.linalg.norm(truth) < 0.04
    zero, _ = mlem(
        geometry, grid, detector, np.zeros((5, 4, 5), np.float32), config=MLEMConfig(iterations=2)
    )
    np.testing.assert_array_equal(np.asarray(zero).ravel()[matrix.sum(axis=0) > 0], 0)
    if isinstance(geometry, LaminographyGeometry):
        initial = np.maximum(np.asarray(tj.reconstruct(scan, "fbp").volume), 0)
        expected, _ = mlem(
            geometry, grid, detector, scan.projections, init_x=initial,
            config=MLEMConfig(iterations=1),
        )  # fmt: skip
        warm = tj.reconstruct(scan, "mlem", iterations=1, warm_start=True)
        np.testing.assert_allclose(warm.volume, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.numerical
@pytest.mark.parametrize("subsets", [1, 2])
def test_unobserved_activity_is_zero_not_the_initial_constant(subsets):
    geometry = tj.ParallelGeometry(Grid(3, 1, 3, 1, 1, 1), Detector(1, 1, 1, 1), [0, 90, 0])
    volume, _ = mlem(
        geometry, geometry.grid, geometry.detector, np.array([2, 6, 2], np.float32)[:, None, None],
        init_x=np.full((3, 1, 3), 7, np.float32),
        config=MLEMConfig(iterations=100, subsets=subsets),
    )  # fmt: skip
    np.testing.assert_allclose(np.asarray(volume), [[[0, 2, 0]]] * 3, atol=1e-5)
