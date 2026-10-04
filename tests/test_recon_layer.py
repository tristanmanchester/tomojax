"""Differentiable reconstruction must solve once and propagate measured data."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.align._geometry.geometry_applier import BaseGeometryArrays, apply_alignment_state

# check-public-imports: allow-private
from tomojax.align._model.state import AlignmentState

# check-public-imports: allow-private
from tomojax.align._objectives import recon_layer

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
            iters=8,
            lambda_tv=0.0,
            L=10.0,
            differentiation_mode="implicit",
            implicit_damping=0.1,
            implicit_cg_iters=32,
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
    for key in ["loss", "data_loss", "regulariser_value", "effective_iters"]:
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
