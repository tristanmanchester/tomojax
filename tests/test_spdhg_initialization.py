"""A supplied SPDHG start obeys the same constraints as every later iterate."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import tomojax as tj
from tomojax.recon import SPDHGConfig, spdhg_tv

SHAPE = (2, 2, 3)


def _problem():
    grid = tj.Grid(*SHAPE[::-1], 0.8, 1.1, 1.3)
    detector = tj.Detector(5, 4, 0.7, 0.9, (0.17, -0.23))
    geometry = tj.ParallelGeometry(grid, detector, [13.0, 71.0, 127.0])
    initial = np.linspace(-0.4, 0.8, 12).astype(np.float32).reshape(SHAPE)
    support = (np.arange(12) % 3 != 0).reshape(SHAPE).astype(np.float32)
    truth = np.maximum(initial, 0) * support
    return grid, detector, geometry, initial, support, truth


@pytest.mark.numerical
@pytest.mark.parametrize("iterations", [0, 1, 2, 5])
@pytest.mark.parametrize("theta", [0.0, 1.0])
def test_spdhg_initial_feasibility_and_first_step_match_dense(iterations, theta):
    grid, detector, geometry, initial, support, truth = _problem()
    matrix = np.stack(
        [np.asarray(tj.project(geometry, b.reshape(SHAPE))).ravel() for b in np.eye(12)],
        axis=1,
    ).astype(np.float64)
    data = (matrix @ (truth + 0.2 * support).ravel()).astype(np.float32).reshape(3, 4, 5)
    expected = np.maximum(initial.astype(np.float64) * support, 0)
    dual = np.zeros(matrix.shape[0])
    adjoint = np.zeros(12)
    extrapolated = np.zeros(12)
    # Independent primal-first dense recurrence, including its terminal image.
    for _ in range(iterations):
        expected = np.maximum((expected - 0.1 * extrapolated.reshape(SHAPE)) * support, 0)
        dual = (dual + 0.2 * (matrix @ expected.ravel() - data.ravel())) / 1.2
        new_adjoint = matrix.T @ dual
        extrapolated = new_adjoint + theta * (new_adjoint - adjoint)
        adjoint = new_adjoint
    volume, info = spdhg_tv(
        geometry,
        grid,
        detector,
        data,
        init_x=initial,
        config=SPDHGConfig(
            iterations=iterations,
            tau=0.1,
            sigma_data=0.2,
            sigma_tv=0.2,
            theta=theta,
            tv_weight=0,
            views_per_batch=3,
            nonnegative=True,
            support=support,
            projector_model="joseph",
        ),
    )
    np.testing.assert_allclose(volume, expected, rtol=2e-5, atol=2e-6)
    if not iterations:
        assert info["loss"] == []


def test_public_spdhg_warm_start_obeys_support_even_with_no_iterations():
    _, _, geometry, _, support, truth = _problem()
    geometry = replace(geometry, angles=np.linspace(0, 360, 24, endpoint=False))
    scan = tj.Scan(np.asarray(tj.project(geometry, truth)), geometry)
    seed = np.asarray(tj.reconstruct(scan, "fbp").volume)
    result = tj.reconstruct(
        scan,
        "spdhg",
        warm_start=True,
        config=SPDHGConfig(iterations=0, tv_weight=0, nonnegative=True, support=support),
    )
    np.testing.assert_array_equal(result.volume, np.maximum(seed, 0) * support)
    assert result.info["warm_start"] is True
