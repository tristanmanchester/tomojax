"""Matrix-free normal-equation solves for coupled volume and pose increments."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

type BlockVector = tuple[jax.Array, jax.Array]


class CoupledLinearResult(NamedTuple):
    increment: BlockVector
    iterations: jax.Array
    relative_residual: jax.Array
    finite: jax.Array


def _dot(left: BlockVector, right: BlockVector) -> jax.Array:
    return sum(
        jnp.vdot(a, b, precision=jax.lax.Precision.HIGHEST).real
        for a, b in zip(left, right, strict=True)
    )


def solve_coupled_normal(
    normal: Callable[[BlockVector], BlockVector],
    rhs: BlockVector,
    inverse_diagonal: BlockVector,
    *,
    max_iters: int,
    rtol: float | jax.Array,
    dot: Callable[[BlockVector, BlockVector], jax.Array] = _dot,
) -> CoupledLinearResult:
    """Solve a positive (semi)definite block normal system by preconditioned CG.

    The two arrays may have different shapes: no dense volume matrix, joint
    Jacobian, or iteration history is materialised. ``inverse_diagonal`` must
    be positive on the active subspace. Inactive coordinates have zero RHS and
    zero operator coupling. A final *explicit* normal residual is returned;
    exhausting the budget never implies convergence. The nonlinear caller must
    independently check its actual constrained objective before accepting a step.
    ``dot`` is the inner product of block vectors (one summing across devices,
    say, when each holds part of a block).
    """
    zero = tuple(jnp.zeros_like(a) for a in rhs)

    def precondition(r):
        return tuple(a * d for a, d in zip(r, inverse_diagonal, strict=True))

    z0 = precondition(rhs)
    gamma0 = dot(rhs, z0)
    rhs_squared = dot(rhs, rhs)
    threshold = jnp.square(jnp.asarray(rtol, rhs_squared.dtype)) * rhs_squared

    def condition(state):
        iteration, _, residual, _, gamma, failed = state
        return (
            (iteration < max_iters) & (dot(residual, residual) > threshold) & (gamma > 0) & ~failed
        )

    def body(state):
        iteration, x, residual, direction, gamma, failed = state
        normal_direction = normal(direction)
        denominator = dot(direction, normal_direction)
        valid = jnp.isfinite(denominator) & (denominator > 0) & jnp.isfinite(gamma)
        alpha = jnp.where(valid, gamma / jnp.where(valid, denominator, 1), 0)
        updated = tuple(a + alpha * b for a, b in zip(x, direction, strict=True))
        next_residual = tuple(
            a - alpha * b for a, b in zip(residual, normal_direction, strict=True)
        )
        z = precondition(next_residual)
        next_gamma = dot(next_residual, z)
        beta = next_gamma / jnp.where(gamma > 0, gamma, 1)
        next_direction = tuple(a + beta * b for a, b in zip(z, direction, strict=True))
        failed = failed | ~valid | ~jnp.isfinite(next_gamma)
        return iteration + 1, updated, next_residual, next_direction, next_gamma, failed

    count, solution, _, _, _, failed = jax.lax.while_loop(
        condition, body, (jnp.int32(0), zero, rhs, z0, gamma0, jnp.bool_(False))
    )
    true_residual = tuple(a - b for a, b in zip(rhs, normal(solution), strict=True))
    relative = jnp.sqrt(dot(true_residual, true_residual) / jnp.maximum(rhs_squared, 1e-30))
    finite = ~failed & jnp.isfinite(relative)
    return CoupledLinearResult(solution, count, relative, finite)
