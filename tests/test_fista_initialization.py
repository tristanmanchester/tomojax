"""Ordered-subset initialization against an explicit optimizer oracle."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.forward import project_joseph
from tomojax.geometry import Detector, Grid, LaminographyGeometry
from tomojax.recon import FistaConfig, fista_tv


@pytest.mark.numerical
@pytest.mark.parametrize("shuffled", [False, True])
def test_ordered_start_tail_and_full_refinement_match_dense_quadratic(shuffled):
    grid, detector = Grid(5, 4, 2, 0.8, 1.1, 1.2), Detector(7, 3, 0.7, 1.1, (0.2, -0.1))
    angles = np.linspace(0, 180, 19)
    if shuffled:
        angles = angles[np.random.default_rng(24).permutation(19)]
    geometry = LaminographyGeometry(grid, detector, angles, tilt_deg=23)
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(19)])

    def forward(x):
        return project_joseph(x.reshape(5, 4, 2), poses, grid, detector, backend="jax")

    # The independently written optimizer, not this matrix, is the oracle here.
    matrix = np.asarray(jax.jit(jax.jacfwd(forward))(jnp.zeros(40))).reshape(-1, 40)
    rng = np.random.default_rng(13)
    data = (matrix @ rng.uniform(0, 1, 40) + rng.normal(0, 0.01, matrix.shape[0])).reshape(19, 3, 7)
    initial = rng.uniform(0, 0.1, (5, 4, 2)).astype(np.float32)
    blocks = [
        matrix.reshape(19, 21, 40)[np.argsort(angles)][i::4].reshape(-1, 40).astype(float)
        for i in range(4)
    ]
    targets = [data[np.argsort(angles)][i::4].ravel() for i in range(4)]
    bounds = [np.max(a.T @ (a @ np.ones(40))) * (1 + 8 * np.finfo(np.float32).eps) for a in blocks]
    x, z, t = initial.ravel().copy(), initial.ravel().copy(), 1.0
    for k in range(12):
        i = (0, 2, 1, 3)[k % 4]
        new = np.maximum(z - blocks[i].T @ (blocks[i] @ z - targets[i]) / bounds[i], 0)
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        x, z, t = new, new + (t - 1) / tn * (new - x), tn
    z, t = x.copy(), 1.0
    for _ in range(5):
        new = np.maximum(z - matrix.T @ (matrix @ z - data.ravel()) / sum(bounds), 0)
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        x, z, t = new, np.maximum(new + (t - 1) / tn * (new - x), 0), tn
    actual, info = fista_tv(
        geometry,
        grid,
        detector,
        data.astype(np.float32),
        init_x=jnp.asarray(initial),
        config=FistaConfig(
            iterations=5, tv_weight=0, nonnegative=True, initial_epochs=3, initial_subsets=4
        ),
    )
    np.testing.assert_allclose(actual.ravel(), x, rtol=6e-5, atol=6e-6)
    np.testing.assert_allclose(info["lipschitz"], sum(bounds), rtol=3e-6)
    assert info["lipschitz"] >= np.linalg.norm(matrix, 2) ** 2 * (1 - 3e-6)
    assert info["initial_updates"] == 12 and info["effective_iterations"] == 5
    assert len(info["loss"]) == 5


@pytest.mark.parametrize(
    "changes",
    [
        {"initial_epochs": -1},
        {"initial_subsets": 0},
        {"initial_subsets": 3},
        {"nonnegative": False},
        {"tv_weight": 0.1},
        {"iterations": 0},
        {"stream_projections": True},
        {"support": jnp.ones((2, 2, 1))},
        {"projector_model": "ray"},
    ],
)
def test_ordered_start_rejects_unsupported_objective_or_storage(changes):
    grid, detector = Grid(2, 2, 1, 1, 1, 1), Detector(2, 1, 1, 1)
    geometry = LaminographyGeometry(grid, detector, [0, 90], tilt_deg=23)
    config = FistaConfig(
        iterations=2, tv_weight=0, nonnegative=True, initial_epochs=1, initial_subsets=2
    )
    with pytest.raises(ValueError, match="initial_"):
        fista_tv(
            geometry,
            grid,
            detector,
            np.ones((2, 1, 2), np.float32),
            config=replace(config, **changes),
        )


@pytest.mark.numerical
@pytest.mark.parametrize("early", [False, True])
def test_volume_larger_than_data_preserves_owned_start_and_final_state(early):
    import importlib

    module = importlib.import_module("tomojax.recon.fista_tv")
    grid, detector = Grid(16, 16, 8, 1, 1, 1), Detector(8, 6, 1, 1)
    geometry = LaminographyGeometry(grid, detector, np.linspace(180, 0, 31), tilt_deg=27)
    rng = np.random.default_rng(614)
    data = jnp.asarray(rng.uniform(size=(31, 6, 8)).astype(np.float32))
    initial = jnp.asarray(rng.uniform(0, 0.1, (16, 16, 8)).astype(np.float32))
    before = np.asarray(initial).copy()
    config = FistaConfig(
        iterations=5,
        tv_weight=0,
        nonnegative=True,
        initial_epochs=2,
        initial_subsets=4,
        recon_rel_tol=1e5 if early else None,
        recon_patience=1,
        lipschitz=1000 if early else None,
    )
    runtime = module._prepare_fista_runtime(
        geometry, grid, detector, data, init_x=initial, config=config, det_grid=None
    )
    expected, limit = module._run_fista_scan(grid, detector, data, runtime)
    callbacks = []
    actual, info = fista_tv(
        geometry,
        grid,
        detector,
        data,
        init_x=initial,
        config=config,
        callback=lambda step, loss: callbacks.append((step, loss)),
    )
    np.testing.assert_allclose(actual, expected.x, rtol=3e-5, atol=2e-7)
    np.testing.assert_allclose(info["loss"], expected.loss, rtol=3e-6)
    np.testing.assert_allclose(info["lipschitz"], limit, rtol=1e-6)
    np.testing.assert_array_equal(initial, before)
    assert np.isfinite(np.asarray(data)).all()
    assert info["effective_iterations"] == int(expected.iters_done)
    assert info["early_stop"] == bool(expected.done)
    assert callbacks[-1][0] == info["effective_iterations"] - 1
