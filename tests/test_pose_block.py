"""Independent dense references for view-coupled pose elimination."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.alignment._pose._pose_block import (
    pose_block_solver,
    trajectory_block_solver,
)


@pytest.mark.numerical
@pytest.mark.parametrize("views", [1, 2, 3, 9, 61])
@pytest.mark.parametrize("smooth", [False, True])
def test_pose_factorization_matches_dense_second_difference_system(views, smooth):
    rng = np.random.default_rng(782)
    active = np.array([1, 0, 1, 1, 0])
    columns = rng.normal(size=(views, 11, 5)) * active
    damping = 0.003
    diagonal = columns.transpose(0, 2, 1) @ columns + damping * np.eye(5)
    weights = np.array([0.8, 1.2, 0.3, 1.7, 0.7]) * active if smooth else np.zeros(5)
    d2 = np.diff(np.eye(views), n=2, axis=0)
    design = np.kron(d2, np.diag(weights))
    dense = 2 * design.T @ design
    for i in range(views):
        dense[i * 5 : (i + 1) * 5, i * 5 : (i + 1) * 5] += diagonal[i]
    rhs = rng.normal(size=(views, 5)) * active
    expected = np.linalg.solve(dense, rhs.ravel()).reshape(views, 5)

    @jax.jit
    def solve(blocks, rhs):
        return pose_block_solver(blocks, jnp.asarray(weights, jnp.float32), has_smoothness=smooth)(
            rhs
        )

    result = solve(jnp.asarray(diagonal, jnp.float32), jnp.asarray(rhs, jnp.float32))
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(result)[:, active == 0], 0)
    np.testing.assert_allclose(
        dense @ np.asarray(result).ravel(), rhs.ravel(), rtol=1e-5, atol=2e-6
    )


@pytest.mark.numerical
def test_pose_smoothness_preserves_fp32_when_x64_is_enabled():
    with jax.enable_x64():
        diagonal = jnp.tile(jnp.eye(5, dtype=jnp.float32), (4, 1, 1))
        weights = jnp.ones(5, dtype=jnp.float32)
        rhs = jnp.ones((4, 5), dtype=jnp.float32)
        result = jax.jit(lambda d, r: pose_block_solver(d, weights, has_smoothness=True)(r))(
            diagonal, rhs
        )
        assert result.dtype == jnp.float32
        # A constant sequence has zero second difference.
        np.testing.assert_allclose(result, rhs, rtol=1e-6, atol=1e-6)


@pytest.mark.numerical
def test_trajectory_block_matches_dense_restricted_system():
    rng = np.random.default_rng(82)
    columns = rng.normal(size=(9, 7, 3))
    diagonal = columns.transpose(0, 2, 1) @ columns + 0.2 * np.eye(3)
    basis = np.linalg.qr(np.column_stack((np.ones(9), np.arange(9), np.arange(9) ** 2)))[0]
    weights = np.array([0.0, 0.2, 0.5])
    second = np.kron(np.diff(np.eye(9), n=2, axis=0), np.diag(weights))
    from scipy.linalg import block_diag

    dense = block_diag(*diagonal) + 2 * second.T @ second
    expansion = np.kron(basis, np.eye(3))
    rhs = rng.normal(size=(9, 3))
    expected = expansion @ np.linalg.solve(
        expansion.T @ dense @ expansion, expansion.T @ rhs.ravel()
    )
    actual = trajectory_block_solver(
        *(jnp.asarray(v, jnp.float32) for v in (diagonal, weights, basis))
    )(jnp.asarray(rhs, jnp.float32))
    np.testing.assert_allclose(np.asarray(actual).ravel(), expected, rtol=2e-5, atol=2e-6)
