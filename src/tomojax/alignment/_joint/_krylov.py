"""Preconditioned CG with explicit convergence verification for the image Schur solve."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


def _dot(a, b):
    return jnp.sum(a.astype(jnp.float64) * b.astype(jnp.float64), dtype=jnp.float64)


class KrylovResult(NamedTuple):
    """Linear step and its convergence evidence, including actual matvec count."""

    solution: jax.Array
    iterations: jax.Array
    relative_residual: jax.Array
    residual_is_explicit: jax.Array
    matvecs: jax.Array


def solve(matvec, rhs, invdiag, maxiter, *, rtol=0.03, precondition=None):
    """Solve K x = rhs from zero with verified tolerance convergence.

    The caller supplies the same symmetric positive operator for every use.
    ``precondition`` can replace Jacobi with another symmetric positive inverse;
    it must stay fixed throughout this CG solve. Image vectors keep
    the RHS precision, and scalar products use FP64. A recursive residual below
    ``rtol`` is verified by a fresh matvec; a false convergence restarts CG from
    that explicit residual within the remaining iteration budget. A solve that
    exhausts its budget above tolerance reports its recursive residual as such,
    without spending another matvec on a diagnostic that cannot change the step.
    The zero initial step has the exact residual ``rhs`` without a matvec.
    A non-positive operator or preconditioner reports an infinite residual.
    """

    def apply_inverse(residual):
        return residual * invdiag if precondition is None else precondition(residual)

    squared = _dot(rhs, rhs)
    z = apply_inverse(rhs)
    gamma = _dot(rhs, z)
    valid = jnp.isfinite(gamma) & ((gamma > 0) | (squared == 0))
    initial = (
        jnp.int32(0),
        jnp.zeros_like(rhs),
        rhs,
        z,
        gamma,
        valid,
        squared,
        jnp.asarray(True),
        jnp.int32(0),
    )
    threshold = rtol**2 * squared

    def condition(state):
        k, _, _, _, _, ok, residual_squared, explicit, _ = state
        advance = (k < maxiter) & (residual_squared > threshold)
        verify = (residual_squared <= threshold) & ~explicit
        return ok & (advance | verify)

    def advance(state):
        k, x, r, p, rz, ok, _, _, products = state
        ap = matvec(p)
        denominator = _dot(p, ap)
        safe = jnp.isfinite(denominator) & (denominator > 0)
        alpha = jnp.where(safe, rz / jnp.where(safe, denominator, 1), 0).astype(rhs.dtype)
        x, r = x + alpha * p, r - alpha * ap
        z = apply_inverse(r)
        next_rz = _dot(r, z)
        safe = safe & jnp.isfinite(next_rz) & (next_rz >= 0)
        beta = jnp.where(safe & (rz > 0), next_rz / jnp.where(rz > 0, rz, 1), 0)
        return (
            k + 1,
            x,
            r,
            z + beta.astype(rhs.dtype) * p,
            next_rz,
            ok & safe,
            _dot(r, r),
            jnp.asarray(False),
            products + 1,
        )

    def verify(state):
        k, x, _, _, _, ok, _, _, products = state
        r = rhs - matvec(x)
        z = apply_inverse(r)
        gamma, residual_squared = _dot(r, z), _dot(r, r)
        ok = ok & jnp.isfinite(gamma) & jnp.isfinite(residual_squared)
        ok = ok & ((gamma > 0) | (residual_squared == 0))
        return k, x, r, z, gamma, ok, residual_squared, jnp.asarray(True), products + 1

    def step(state):
        return jax.lax.cond(state[6] <= threshold, verify, advance, state)

    k, x, _, _, _, valid, residual_squared, explicit, products = jax.lax.while_loop(
        condition, step, initial
    )
    relative = jnp.sqrt(residual_squared / jnp.maximum(squared, 1e-300))
    return KrylovResult(x, k, jnp.where(valid, relative, jnp.inf), explicit, products)
