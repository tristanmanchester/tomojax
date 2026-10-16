"""Normalized algebraic updates against a dense matched matrix."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.geometry import ConeSegments
from tomojax.recon import SIRTConfig, sirt


@pytest.mark.numerical
@pytest.mark.parametrize("subsets", [1, 2, 5])
@pytest.mark.parametrize("kind", ["parallel", "axial", "cone", "segments"])
def test_sirt_matches_dense_updates_with_bounds_support_and_tail(subsets, kind):
    grid = tj.Grid(3, 3, 2, 0.8, 1.2, 1.3)
    detector = tj.Detector(5, 4, 0.9, 1.1, (0.1, -0.2))
    if kind == "axial":
        detector = replace(detector, nv=grid.nz, dv=grid.vz, center=(0.1, 0.0))
    angles = [13, 47, 89, 123, 166]
    geometry = (
        tj.ParallelGeometry(grid, detector, angles)
        if kind in {"parallel", "axial"}
        else tj.ConeGeometry(grid, detector, angles, tj.ConeBeam(15, 24))
    )
    if kind == "segments":
        geometry = ConeSegments(
            (
                replace(geometry, angles=angles[:2]),
                replace(
                    geometry,
                    angles=angles[2:],
                    beam=tj.ConeBeam(19, 28),
                    detector=replace(detector, center=(-0.2, 0.4)),
                ),
            )
        )
    shape = (3, 3, 2)
    # Assemble A one column at a time, independent of the solver and transpose.
    basis = np.eye(18, dtype=np.float32).reshape((18, *shape))
    matrix = np.stack([np.asarray(tj.project(geometry, x)).ravel() for x in basis], axis=1)
    rng = np.random.default_rng(12)
    truth = rng.uniform(0.1, 0.8, size=18)
    support = np.ones(18, np.float32)
    support[[2, 7, 13]] = 0
    truth *= support
    data = matrix @ truth
    expected = np.full(18, 0.2) * support
    for _ in range(4):
        for index in range(subsets):
            ids = np.arange(5)[index::subsets]
            a = matrix.reshape(5, -1, 18)[ids].reshape(-1, 18).astype(np.float64)
            b = data.reshape(5, -1)[ids].ravel()
            rows, columns = a @ support, a.sum(axis=0)
            residual = np.divide(b - a @ expected, rows, out=np.zeros_like(b), where=rows > 0)
            update = np.divide(a.T @ residual, columns, out=np.zeros(18), where=columns > 0)
            expected = np.clip(expected + 1.1 * update, 0.05, 0.7) * support
    volume, info = sirt(
        geometry,
        grid,
        detector,
        data.reshape(5, detector.nv, detector.nu),
        init_x=np.full(shape, 0.2),
        config=SIRTConfig(
            iterations=4,
            subsets=subsets,
            relaxation=1.1,
            lower_bound=0.05,
            upper_bound=0.7,
            support=support.reshape(shape),
            views_per_batch=2,
        ),
    )
    np.testing.assert_allclose(np.asarray(volume).ravel(), expected, atol=3e-6, rtol=3e-5)
    assert info["subset_updates"] == 4 * subsets


@pytest.mark.numerical
@pytest.mark.parametrize("subsets", [1, 3])
def test_public_sirt_converges_on_consistent_data(subsets):
    grid = tj.Grid(5, 5, 1, 1, 1, 1)
    geometry = tj.ParallelGeometry(grid, tj.Detector(9, 1, 1, 1), np.arange(0, 180, 10))
    truth = jnp.zeros((5, 5, 1)).at[1:4, 1:4, :].set(1)
    data = np.asarray(tj.project(geometry, truth))
    result = tj.reconstruct(
        tj.Scan(data, geometry),
        "sirt",
        config=SIRTConfig(subsets=subsets, lower_bound=0, iterations=100),
    )
    jax.block_until_ready(result.volume)
    assert np.linalg.norm(np.asarray(result.volume) - truth) / np.linalg.norm(truth) < 0.025


@pytest.mark.numerical
@pytest.mark.parametrize("subsets", [1, 3])
def test_axial_memory_slabs_preserve_the_full_constrained_solve(monkeypatch, subsets):
    import tomojax.backends

    grid = tj.Grid(5, 4, 3, 0.8, 1.2, 1.3, vol_center=(0.2, -0.3, 0.7))
    detector = tj.Detector(7, 3, 0.9, 1.3, (0.1, 0.7))
    geometry = tj.ParallelGeometry(grid, detector, np.arange(0, 180, 20))
    rng = np.random.default_rng(42)
    truth = rng.uniform(size=(5, 4, 3)).astype(np.float32)
    data = np.asarray(tj.project(geometry, truth))
    cfg = SIRTConfig(
        iterations=5, subsets=subsets, lower_bound=0, upper_bound=0.9, support=truth > 0.3
    )
    initial = truth * 0.1
    full, _ = sirt(geometry, grid, detector, data, init_x=initial, config=cfg)
    monkeypatch.setattr(tomojax.backends, "device_free_memory_bytes", lambda: 1)
    slabs, info = sirt(geometry, grid, detector, data, init_x=initial, config=cfg)
    assert info["host_slabs"] and isinstance(slabs, np.ndarray)
    np.testing.assert_allclose(slabs, full, atol=3e-6, rtol=3e-5)
