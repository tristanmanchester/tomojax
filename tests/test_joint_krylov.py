"""Numerical contracts for the preconditioned image Schur solve."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import cg

# check-public-imports: allow-private
from tomojax.alignment._joint._krylov import solve


def test_custom_symmetric_positive_inverse_solves_the_dense_system():
    rng = np.random.default_rng(43)
    n = 13
    a, b = (rng.normal(size=(n, n)) for _ in range(2))
    matrix, base = a.T @ a + np.eye(n), b.T @ b + np.eye(n)
    with jax.enable_x64(True):
        operator, inverse = jnp.asarray(matrix), jnp.asarray(base)
        rhs = jnp.asarray(rng.normal(size=n))
        result = solve(
            lambda p: operator @ p,
            rhs,
            jnp.ones(n),
            30,
            precondition=lambda v: inverse @ v,
            rtol=1e-12,
        )
    np.testing.assert_allclose(
        result.solution, np.linalg.solve(matrix, rhs), rtol=1e-10, atol=1e-11
    )
    assert float(result.relative_residual) < 1e-10


def test_nonpositive_operator_is_reported_as_failure():
    with jax.enable_x64(True):
        matrix = jnp.diag(jnp.array([-1.0, 2, 3]))
        result = solve(lambda p: matrix @ p, jnp.ones(3), jnp.ones(3), 5)
    assert np.isinf(float(result[2]))


def test_budget_exhaustion_keeps_the_cg_iterate_without_a_final_diagnostic_matvec():
    rng = np.random.default_rng(52)
    n, budget = 37, 8
    vectors, _ = np.linalg.qr(rng.normal(size=(n, n)))
    matrix = (vectors * np.geomspace(1.0, 100.0, n)) @ vectors.T
    rhs = rng.normal(size=n)
    inverse = 1 / np.diag(matrix)
    expected, status = cg(matrix, rhs, M=np.diag(inverse), maxiter=budget, rtol=1e-12)
    assert status == budget
    calls = []
    with jax.enable_x64(True):
        operator = jnp.asarray(matrix)

        def matvec(p):
            jax.debug.callback(lambda: calls.append(1), ordered=True)
            return operator @ p

        result = jax.jit(
            lambda: solve(matvec, jnp.asarray(rhs), jnp.asarray(inverse), budget, rtol=1e-12)
        )()
        jax.block_until_ready(result)
        jax.effects_barrier()
    np.testing.assert_allclose(result.solution, expected, rtol=2e-12, atol=2e-12)
    assert int(result.iterations) == int(result.matvecs) == len(calls) == budget
    assert not bool(result.residual_is_explicit)
    assert float(result.relative_residual) > 1e-12


def test_convergence_is_explicitly_verified_and_zero_rhs_needs_no_matvec():
    calls = []
    with jax.enable_x64(True):
        diagonal = jnp.arange(1.0, 18.0)
        rhs = jnp.sin(diagonal)

        def matvec(p):
            jax.debug.callback(lambda: calls.append(1), ordered=True)
            return diagonal * p

        result = solve(matvec, rhs, 1 / diagonal, 8, rtol=1e-10)
        jax.block_until_ready(result)
        jax.effects_barrier()
        assert int(result.iterations) == 1
        assert int(result.matvecs) == len(calls) == 2
        assert bool(result.residual_is_explicit)
        explicit = np.linalg.norm(np.asarray(rhs - diagonal * result.solution)) / np.linalg.norm(
            rhs
        )
        np.testing.assert_allclose(result.relative_residual, explicit, atol=1e-15)
        calls.clear()
        zero = solve(matvec, jnp.zeros_like(rhs), 1 / diagonal, 8)
        jax.block_until_ready(zero)
        jax.effects_barrier()
    assert int(zero.iterations) == int(zero.matvecs) == len(calls) == 0
    assert bool(zero.residual_is_explicit) and float(zero.relative_residual) == 0


def test_float32_false_convergence_restarts_within_the_remaining_budget():
    rng = np.random.default_rng(53)
    vectors, _ = np.linalg.qr(rng.normal(size=(13, 13)))
    matrix = ((vectors * np.geomspace(1.0, 1e4, 13)) @ vectors.T).astype(np.float32)
    rhs = rng.normal(size=13).astype(np.float32)
    budget, tolerance = 80, 1e-8
    with jax.enable_x64(True):
        operator = jnp.asarray(matrix)
        result = solve(
            lambda p: jnp.matmul(operator, p, precision=jax.lax.Precision.HIGHEST),
            jnp.asarray(rhs),
            1 / jnp.diag(operator),
            budget,
            rtol=tolerance,
        )
    # This tolerance is below the attainable residual for this FP32 system.
    # Failed verifications spend matvecs, restart the direction, and never
    # masquerade as successful tolerance convergence.
    assert int(result.matvecs) > int(result.iterations)
    assert int(result.iterations) == budget
    assert not (bool(result.residual_is_explicit) and float(result.relative_residual) <= tolerance)
