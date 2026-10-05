"""Bounded-storage Cholesky solve for the per-view pose normal block."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

# Relative to each scaled diagonal entry: keeps FP32 block factorisations stable.
_RELATIVE_DAMPING = 1e-5


def _cholesky(blocks: jax.Array) -> jax.Array:
    """Factor unit-diagonal blocks, damping only those whose FP32 factor fails."""
    factors = jnp.linalg.cholesky(blocks)
    eye = jnp.eye(blocks.shape[-1], dtype=blocks.dtype)
    damped = jnp.linalg.cholesky(blocks + _RELATIVE_DAMPING * eye)
    failed = ~jnp.all(jnp.isfinite(factors), axis=(-2, -1), keepdims=True)
    return jnp.where(failed, damped, factors)


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
    storage, never a dense (views * DOFs)-squared matrix.

    The system is solved in symmetrically scaled variables with a unit
    diagonal, which changes only the conditioning. Rotation and translation
    columns differ in size by orders of magnitude, and a view can determine
    some pose combinations only weakly. Where FP32 Cholesky of a block still
    fails, that block alone is factored with ``_RELATIVE_DAMPING`` added to its
    unit diagonal (Marquardt damping); every other block is solved exactly.
    """
    n, width, _ = diagonal.shape
    scale = jax.lax.rsqrt(
        jnp.maximum(jnp.diagonal(diagonal, axis1=1, axis2=2), jnp.finfo(diagonal.dtype).tiny)
    )
    if not has_smoothness or n < 3:
        factors = _cholesky(scale[:, :, None] * diagonal * scale[:, None, :])

        def solve(rhs):
            def one(factor, value):
                y = solve_triangular(factor, value, lower=True)
                return solve_triangular(factor.T, y, lower=False)

            return scale * jax.vmap(one)(factors, scale * rhs)

        return solve

    # Diagonals of D2.T D2, with only the n-2 actual second differences.
    main = jnp.zeros(n, diagonal.dtype).at[:-2].add(1).at[1:-1].add(4).at[2:].add(1)
    first = jnp.zeros(n, diagonal.dtype).at[1:-1].add(-2).at[2:].add(-2)
    second = jnp.zeros(n, diagonal.dtype).at[2:].set(1)
    smooth = jnp.diag(2 * smoothness_weights**2)
    diagonal = diagonal + main[:, None, None] * smooth
    scale = jax.lax.rsqrt(
        jnp.maximum(jnp.diagonal(diagonal, axis1=1, axis2=2), jnp.finfo(diagonal.dtype).tiny)
    )
    diagonal = scale[:, :, None] * diagonal * scale[:, None, :]
    # Row i couples views i-1 and i-2; scale rows by view i, columns by the other.
    lag_one = jnp.concatenate((scale[:1], scale[:-1]))
    lag_two = jnp.concatenate((scale[:2], scale[:-2]))
    below_one = first[:, None, None] * smooth * scale[:, :, None] * lag_one[:, None, :]
    below_two = second[:, None, None] * smooth * scale[:, :, None] * lag_two[:, None, :]
    identity = jnp.eye(width, dtype=diagonal.dtype)
    zero = jnp.zeros_like(identity)

    def factor_step(previous, blocks):
        prev_two, prev_one, prev_sub = previous
        block, one, two = blocks
        l_two = solve_triangular(prev_two, two.T, lower=True).T
        l_one = solve_triangular(prev_one, (one - _matmul(l_two, prev_sub.T)).T, lower=True).T
        l_diagonal = _cholesky(block - _matmul(l_one, l_one.T) - _matmul(l_two, l_two.T))
        return (prev_one, l_diagonal, l_one), (l_diagonal, l_one, l_two)

    _, (factors, first_factors, second_factors) = jax.lax.scan(
        factor_step, (identity, identity, zero), (diagonal, below_one, below_two)
    )

    def solve(unscaled_rhs):
        rhs = scale * unscaled_rhs
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

        return (
            scale
            * jax.lax.scan(backward, initial, (factors, next_one, next_two, y), reverse=True)[1]
        )

    return solve
