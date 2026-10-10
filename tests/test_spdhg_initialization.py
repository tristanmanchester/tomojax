"""A supplied SPDHG start obeys the same constraints as every later iterate."""

from __future__ import annotations

import numpy as np
import pytest

import tomojax as tj
from tomojax.recon import SPDHGConfig, spdhg_tv


@pytest.fixture(scope="module", params=["parallel", "lamino", "cone"])
def problem(request):
    grid = tj.Grid(3, 2, 2, 0.8, 1.1, 1.3)
    detector = tj.Detector(5, 4, 0.7, 0.9, (0.17, -0.23))
    angles = [13.0, 71.0, 127.0]
    if request.param == "parallel":
        geometry = tj.ParallelGeometry(grid, detector, angles)
    elif request.param == "lamino":
        geometry = tj.LaminographyGeometry(grid, detector, angles, tilt_deg=25.0)
    else:
        geometry = tj.ConeGeometry(grid, detector, angles, tj.ConeBeam(20.0, 30.0), tilt_deg=25.0)
    shape = (3, 2, 2)
    matrix = np.stack(
        [
            np.asarray(tj.project(geometry, b.reshape(shape))).ravel()
            for b in np.eye(12, dtype=np.float32)
        ],
        axis=1,
    ).astype(np.float64)
    initial = np.linspace(-0.4, 0.8, 12).astype(np.float32).reshape(shape)
    full = (np.arange(12) % 3 != 0).reshape(shape).astype(np.float32)
    broadcast = np.array([True, False])
    # A known image feasible for all three support choices, not a fitted reference.
    truth = np.maximum(initial, 0) * full * broadcast
    data = (matrix @ truth.ravel()).astype(np.float32).reshape(3, 4, 5)
    return grid, detector, geometry, matrix, initial, full, broadcast, truth, data


@pytest.mark.numerical
@pytest.mark.parametrize("iterations", [0, 1])
@pytest.mark.parametrize("nonnegative", [False, True])
@pytest.mark.parametrize("support_kind", ["none", "full", "broadcast"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("backend", ["jax", "auto"])
def test_spdhg_initial_feasibility_and_first_step_match_dense(
    problem, iterations, nonnegative, support_kind, stream, backend
):
    grid, detector, geometry, matrix, initial, full, broadcast, truth, data = problem
    support = {"none": None, "full": full, "broadcast": broadcast}[support_kind]
    expected = initial.astype(np.float64)
    if support is not None:
        expected *= support
    if nonnegative:
        expected = np.maximum(expected, 0)
    if iterations:
        dual = 0.2 * (matrix @ expected.ravel() - data.ravel()) / 1.2
        expected -= 0.1 * (matrix.T @ dual).reshape(initial.shape)
        if support is not None:
            expected *= support
        if nonnegative:
            expected = np.maximum(expected, 0)
    callbacks = []
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
            tv_weight=0,
            views_per_batch=3,
            nonnegative=nonnegative,
            support=support,
            stream_projections=stream,
            projector_model="joseph",
            projector_backend=backend,
            log_every=1,
        ),
        callback=lambda *args: callbacks.append(args),
    )
    volume = np.asarray(volume)
    np.testing.assert_allclose(volume, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        np.linalg.norm(volume - truth), np.linalg.norm(expected - truth), rtol=2e-5, atol=2e-6
    )
    if nonnegative:
        assert volume.min() >= 0
    if support is not None:
        assert np.count_nonzero(volume[np.broadcast_to(support == 0, initial.shape)]) == 0
    if not iterations:
        assert info["loss"] == [] and callbacks == []
    np.testing.assert_array_equal(
        initial, np.linspace(-0.4, 0.8, 12).astype(np.float32).reshape(3, 2, 2)
    )


@pytest.mark.numerical
@pytest.mark.parametrize("regulariser", ["tv", "huber_tv"])
@pytest.mark.parametrize("support_kind", ["none", "full", "broadcast"])
@pytest.mark.parametrize("stream", [False, True])
def test_spdhg_tv_duals_start_from_the_same_feasible_image(
    problem, regulariser, support_kind, stream
):
    grid, detector, geometry, matrix, initial, full, broadcast, truth, data = problem
    support = {"none": None, "full": full, "broadcast": broadcast}[support_kind]
    expected = np.maximum(initial.astype(np.float64) * (1 if support is None else support), 0)
    # Independent forward-difference matrices, including the zero terminal rows.
    indices = np.arange(initial.size).reshape(initial.shape)
    edges = np.zeros((3, initial.size, initial.size))
    for point in np.ndindex(initial.shape):
        for axis in range(3):
            if point[axis] + 1 < initial.shape[axis]:
                neighbour = list(point)
                neighbour[axis] += 1
                edges[axis, indices[point], indices[point]] = -1
                edges[axis, indices[point], indices[tuple(neighbour)]] = 1
    dual_tv = 0.2 * (edges @ expected.ravel())
    if regulariser == "huber_tv":
        dual_tv *= 0.03 / (0.03 + 0.2 * 0.1)
    dual_tv /= np.maximum(1, np.linalg.norm(dual_tv, axis=0) / 0.03)
    dual_data = 0.2 * (matrix @ expected.ravel() - data.ravel()) / 1.2
    gradient = matrix.T @ dual_data + sum(
        edge.T @ dual for edge, dual in zip(edges, dual_tv, strict=True)
    )
    expected -= 0.1 * gradient.reshape(initial.shape)
    expected = np.maximum(expected * (1 if support is None else support), 0)
    volume, _ = spdhg_tv(
        geometry,
        grid,
        detector,
        data,
        init_x=initial,
        config=SPDHGConfig(
            iterations=1,
            tau=0.1,
            sigma_data=0.2,
            sigma_tv=0.2,
            tv_weight=0.03,
            huber_delta=0.1,
            regulariser=regulariser,
            views_per_batch=3,
            nonnegative=True,
            support=support,
            stream_projections=stream,
            projector_model="joseph",
        ),
    )
    np.testing.assert_allclose(volume, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(volume) - truth), np.linalg.norm(expected - truth), rtol=2e-5
    )


def test_public_spdhg_warm_start_obeys_support_even_with_no_iterations(problem):
    grid, detector, geometry, _, _, full, _, truth, _ = problem
    from dataclasses import replace

    geometry = replace(geometry, angles=np.linspace(0, 360, 24, endpoint=False))
    scan = tj.Scan(np.asarray(tj.project(geometry, truth)), geometry)
    seed = np.asarray(tj.reconstruct(scan, "fbp").volume)
    result = tj.reconstruct(
        scan,
        "spdhg",
        warm_start=True,
        config=SPDHGConfig(
            iterations=0,
            tv_weight=0,
            tau=0.1,
            sigma_data=0.2,
            sigma_tv=0.2,
            nonnegative=True,
            support=full,
        ),
    )
    np.testing.assert_array_equal(result.volume, np.maximum(seed, 0) * full)
    assert result.info["warm_start"] is True
