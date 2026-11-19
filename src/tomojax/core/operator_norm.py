"""Power-iteration bounds on a normal operator's norm, shared by the solvers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable


def unit(vector: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``vector`` scaled to unit length, and its length; zero stays zero (no floor)."""
    length = jnp.linalg.norm(vector.ravel())
    return vector / jnp.where(length == 0, 1.0, length), length


# Power iteration approaches ||N|| from below; on a nonnegative N the Collatz
# ratio max(N q / q) of the same positive iterate bounds it from above. A step
# from the upper bound always converges; iterating until the two agree within
# BOUND_GAP keeps that step close to the largest one.
BOUND_GAP = 0.02
# Passes, unit iterate, power estimate, Collatz bound.
_PowerState = tuple[jax.Array, jax.Array, jax.Array, jax.Array]


def collatz_ratio(normal: jax.Array, vector: jax.Array) -> jax.Array:
    """``max(normal / vector)`` over the positive entries of ``vector``."""
    ratio = normal / jnp.where(vector > 0, vector, 1.0)
    return jnp.max(jnp.where(vector > 0, ratio, 0.0))


def bounds_settled(k: int, iterations: int, power: float, upper: float) -> bool:
    """Whether power iteration may stop at ``k`` passes: enough, and the bounds agree."""
    cap = max(4 * iterations, 20)
    return k >= cap or (k >= iterations and upper <= (1 + BOUND_GAP) * power)


def normal_upper_bound(
    apply_normal: Callable[[jax.Array], jax.Array],
    start: jax.Array,
    *,
    iterations: int,
    nonnegative: bool = True,
) -> jax.Array:
    """A step-safe ``||N||`` for symmetric positive semidefinite ``N``, from a positive start.

    For a nonnegative ``N`` this is the Collatz upper bound, after at least
    ``iterations`` passes and until it is within ``BOUND_GAP`` of the power
    estimate. Otherwise (signed cubic weights) it is the power estimate.
    """
    iterations = max(1, iterations)
    cap = max(4 * iterations, 20)

    def proceed(state: _PowerState) -> jax.Array:
        k, _, power, upper = state
        return (k < iterations) | ((k < cap) & (upper > (1 + BOUND_GAP) * power))

    def step(state: _PowerState) -> _PowerState:
        k, vector, _, _ = state
        normal = apply_normal(vector)
        upper = collatz_ratio(normal, vector) if nonnegative else jnp.inf
        new, power = unit(normal)
        # Fixed carry types, also under global x64.
        return k + 1, new.astype(vector.dtype), power.astype(jnp.float32), jnp.float32(upper)

    vector = unit(start)[0]
    state = (jnp.int32(0), vector, jnp.float32(0), jnp.float32(jnp.inf))
    if not nonnegative:
        state = jax.lax.fori_loop(0, iterations, lambda _, s: step(s), state)
        return state[2]
    _, _, _, upper = jax.lax.while_loop(proceed, step, state)
    # A small FP32 margin covers rounding in the passes.
    return upper * (1.0 + 8 * jnp.finfo(jnp.float32).eps)
