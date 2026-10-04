"""Bounded-storage Cholesky solve for the per-view pose normal block."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


def _matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def pose_block_solver(
    diagonal: jax.Array,
    smoothness_weights: jax.Array,
    *,
    has_smoothness: bool,
) -> Callable[[jax.Array], jax.Array]:
    """Factor J.T J + damping I + Hessian(||weights * D2 p||²).

    ``diagonal`` contains the per-view data Gram matrices plus positive pose
    damping. Frozen DOFs must have zero data columns and smoothness weights.
    With no smoothness, each view is independent. Otherwise D2 couples at most
    two adjacent views: block Cholesky and substitution use O(views * DOFs²)
    storage, never a dense (views * DOFs)-squared matrix. No extra damping,
    jitter, or diagonal approximation changes the requested system.
    """
    n, width, _ = diagonal.shape
    if not has_smoothness or n < 3:
        factors = jnp.linalg.cholesky(diagonal)

        def solve(rhs):
            def one(factor, value):
                y = solve_triangular(factor, value, lower=True)
                return solve_triangular(factor.T, y, lower=False)

            return jax.vmap(one)(factors, rhs)

        return solve

    # Diagonals of D2.T D2, with only the n-2 actual second differences.
    main = jnp.zeros(n, diagonal.dtype).at[:-2].add(1).at[1:-1].add(4).at[2:].add(1)
    first = jnp.zeros(n, diagonal.dtype).at[1:-1].add(-2).at[2:].add(-2)
    second = jnp.zeros(n, diagonal.dtype).at[2:].set(1)
    smooth = jnp.diag(2 * smoothness_weights**2)
    diagonal = diagonal + main[:, None, None] * smooth
    below_one = first[:, None, None] * smooth
    below_two = second[:, None, None] * smooth
    identity = jnp.eye(width, dtype=diagonal.dtype)
    zero = jnp.zeros_like(identity)

    def factor_step(previous, blocks):
        prev_two, prev_one, prev_sub = previous
        block, one, two = blocks
        l_two = solve_triangular(prev_two, two.T, lower=True).T
        l_one = solve_triangular(prev_one, (one - _matmul(l_two, prev_sub.T)).T, lower=True).T
        l_diagonal = jnp.linalg.cholesky(block - _matmul(l_one, l_one.T) - _matmul(l_two, l_two.T))
        return (prev_one, l_diagonal, l_one), (l_diagonal, l_one, l_two)

    _, (factors, first_factors, second_factors) = jax.lax.scan(
        factor_step, (identity, identity, zero), (diagonal, below_one, below_two)
    )

    def solve(rhs):
        initial = (jnp.zeros_like(rhs[0]), jnp.zeros_like(rhs[0]))

        def forward(previous, blocks):
            prev_two, prev_one = previous
            d, one, two, value = blocks
            y = solve_triangular(
                d, value - _matmul(one, prev_one) - _matmul(two, prev_two), lower=True
            )
            return (prev_one, y), y

        _, y = jax.lax.scan(forward, initial, (factors, first_factors, second_factors, rhs))
        # For row i of L.T use L[i+1,i] and L[i+2,i].
        next_one = jnp.concatenate((first_factors[1:], zero[None]))
        next_two = jnp.concatenate((second_factors[2:], jnp.stack((zero, zero))))

        def backward(following, blocks):
            next_two_value, next_one_value = following
            d, one, two, value = blocks
            x = solve_triangular(
                d.T,
                value - _matmul(one.T, next_one_value) - _matmul(two.T, next_two_value),
                lower=False,
            )
            return (next_one_value, x), x

        return jax.lax.scan(backward, initial, (factors, next_one, next_two, y), reverse=True)[1]

    return solve
