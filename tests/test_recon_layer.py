"""Differentiable reconstruction must solve once and propagate measured data."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.alignment._geometry.geometry_applier import BaseGeometryArrays, apply_alignment_state

# check-public-imports: allow-private
from tomojax.alignment._model.state import AlignmentState

# check-public-imports: allow-private
from tomojax.alignment._objectives import fold_recon, recon_layer

# check-public-imports: allow-private
from tomojax.core.projector import forward_project_view_T
from tomojax.geometry import Detector, Grid, ParallelGeometry


def _problem():
    grid = Grid(2, 2, 2, 0.8, 1.1, 1.3)
    detector = Detector(4, 3, 0.7, 0.9, (0.17, 0.23))
    geometry = ParallelGeometry(grid, detector, np.array([13.0, 71.0, 127.0], dtype=np.float32))
    state = AlignmentState.zeros(n_views=3)
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    layer = recon_layer.ReconLayer(
        base,
        grid,
        detector,
        recon_layer.ReconLayerConfig(
            iterations=8,
            tv_weight=0.0,
            lipschitz=10.0,
            differentiation_mode="implicit",
            implicit_damping=0.1,
            implicit_cg_iterations=32,
            implicit_cg_tol=1e-6,
        ),
    )
    data = jnp.asarray(np.random.default_rng(71).normal(size=(3, 3, 4)), dtype=jnp.float32)
    return layer, state, data


@pytest.mark.numerical
def test_implicit_layer_solves_once_and_preserves_primal_diagnostics(monkeypatch):
    layer, state, data = _problem()
    reference = replace(
        layer, config=replace(layer.config, differentiation_mode="unrolled")
    ).reconstruct(state=state, projections=data)
    real_solve = recon_layer.fista_tv_core_arrays
    count = 0

    def counted_solve(**kwargs):
        nonlocal count
        count += 1
        return real_solve(**kwargs)

    monkeypatch.setattr(recon_layer, "fista_tv_core_arrays", counted_solve)
    result = layer.reconstruct(state=state, projections=data)
    jax.block_until_ready(result.x)
    assert count == 1
    np.testing.assert_allclose(result.x, reference.x, rtol=1e-6, atol=1e-6)
    for key in ["loss", "data_loss", "regulariser_value", "effective_iterations"]:
        np.testing.assert_allclose(result.info[key], reference.info[key], rtol=1e-6, atol=1e-6)


@pytest.mark.numerical
def test_implicit_data_gradient_matches_dense_damped_normal_equations():
    layer, state, data = _problem()
    effective = apply_alignment_state(layer.base, state)
    shape = (2, 2, 2)

    def project(flat):
        return jax.vmap(
            lambda t: forward_project_view_T(
                t, layer.grid, layer.detector, flat.reshape(shape), det_grid=effective.det_grid
            )
        )(effective.pose_stack).ravel()

    matrix = np.asarray(jax.jacfwd(project)(jnp.zeros(8))).astype(np.float64)
    cotangent = jnp.arange(1, 9, dtype=jnp.float32).reshape(shape) / 8
    expected = matrix @ np.linalg.solve(
        matrix.T @ matrix + 0.1 * np.eye(8), np.asarray(cotangent).ravel()
    )

    def value(y, initial):
        return jnp.vdot(layer.reconstruct(state=state, projections=y, init_x=initial).x, cotangent)

    grad_data, grad_init = jax.jit(jax.grad(value, argnums=(0, 1)))(data, jnp.zeros(shape))
    np.testing.assert_allclose(grad_data.ravel(), expected, rtol=2e-4, atol=2e-5)
    np.testing.assert_array_equal(grad_init, np.zeros(shape))


def _identity_layer(grid, detector, n_views, **config):
    geometry = ParallelGeometry(grid, detector, [0.0] * n_views)
    layer = recon_layer.ReconLayer(
        BaseGeometryArrays.from_geometry(geometry, detector),
        grid,
        detector,
        recon_layer.ReconLayerConfig(
            tv_weight=0, ray_integrator="joseph", differentiation_mode="implicit", **config
        ),
    )
    return layer, AlignmentState.zeros(n_views=n_views)


@pytest.mark.numerical
def test_implicit_nonnegative_gradient_vanishes_on_pinned_voxels():
    grid, detector = Grid(2, 1, 1, 1.0, 1.0, 1.0), Detector(2, 1, 1.0, 1.0)
    layer, state = _identity_layer(
        grid,
        detector,
        1,
        iterations=1,
        lipschitz=1,
        nonnegative=True,
        implicit_damping=0.1,
        implicit_cg_tol=1e-7,
    )
    # A=I: the constrained solution is max(y, 0), and only the free voxel moves with y.
    data = jnp.asarray([-1.0, 2.0]).reshape(1, 1, 2)
    solution = layer.reconstruct(state=state, projections=data).x
    np.testing.assert_allclose(solution.ravel(), [0.0, 2.0], atol=1e-6)
    derivative = jax.grad(lambda y: layer.reconstruct(state=state, projections=y).x.sum())(data)
    np.testing.assert_allclose(derivative.ravel(), [0.0, 1 / 1.1], rtol=1e-6)


@pytest.mark.numerical
def test_implicit_view_weight_derivative_matches_weighted_mean():
    grid, detector = Grid(1, 1, 1, 1.0, 1.0, 1.0), Detector(1, 1, 1.0, 1.0)
    layer, state = _identity_layer(
        grid, detector, 2, iterations=60, lipschitz=3, implicit_damping=0, implicit_cg_tol=1e-7
    )
    weights = jnp.asarray([0.0, 2.0])
    data = jnp.asarray([1.0, 3.0]).reshape(2, 1, 1)

    def reconstruct(w):
        return layer.reconstruct(state=state, projections=data, view_weights=w).x.sum()

    # Two rays through one voxel give x = sum(w y) / sum(w), here with one zero weight.
    np.testing.assert_allclose(reconstruct(weights), 3.0, rtol=1e-6)
    expected = (np.asarray([1.0, 3.0]) - 3.0) / 2.0
    np.testing.assert_allclose(jax.grad(reconstruct)(weights), expected, rtol=2e-6, atol=2e-7)


@pytest.mark.numerical
def test_train_fold_keeps_the_requested_ray_integrator():
    grid, detector = Grid(2, 1, 1, 0.8, 1.1, 1.3), Detector(4, 3, 0.7, 0.9)
    geometry = ParallelGeometry(grid, detector, [13.0, 71.0, 127.0])
    state = AlignmentState.zeros(n_views=3)
    effective = apply_alignment_state(BaseGeometryArrays.from_geometry(geometry, detector), state)

    def project(flat):
        return jax.vmap(
            lambda t: forward_project_view_T(
                t, grid, detector, flat.reshape(2, 1, 1), ray_integrator="exact"
            )
        )(effective.pose_stack)

    matrix = np.asarray(jax.jacfwd(project)(jnp.zeros(2)), np.float64)
    truth = np.asarray([0.2, 1.0])
    result, _ = fold_recon.reconstruct_train_fold_nograd(
        geometry=geometry,
        grid=grid,
        detector=detector,
        projections=jnp.asarray(np.einsum("...i,i->...", matrix, truth), jnp.float32),
        state=state,
        train_idx=jnp.asarray([0, 1, 2]),
        train_mask=jnp.asarray([1, 1, 1]),
        init_x=None,
        level_factor=1,
        cfg=fold_recon.FoldReconstructionConfig(
            iterations=200,
            tv_weight=0,
            regulariser="huber_tv",
            huber_delta=0.01,
            tv_prox_iterations=1,
            nonnegative=False,
            lipschitz=float(np.linalg.norm(matrix.reshape(-1, 2), 2) ** 2 * 1.01),
            ray_integrator="exact",
        ),
    )
    np.testing.assert_allclose(np.asarray(result).ravel(), truth, rtol=2e-4, atol=2e-5)
