"""Portable loop unrolling for the Pallas Triton lowering."""

from __future__ import annotations

from collections.abc import Callable
import operator
from typing import Any

import jax
import jax.numpy as jnp


def paired_fori_loop(
    upper: jnp.ndarray,
    body: Callable[[jnp.ndarray, Any], Any],
    initial: Any,
) -> Any:
    """Expose two ray samples per iteration without changing their recurrence.

    The dynamic bound still clips work to the longest active ray in a tile.
    A separate zero-or-one-step tail respects odd bounds and explicit sample
    limits; starting the next sample via a coordinate multiply would drift from
    the reference's repeated float32 additions.
    """
    pairs = upper // 2

    def pair_body(pair, state):
        state = body(pair * 2, state)
        return body(pair * 2 + 1, state)

    result = jax.lax.fori_loop(0, pairs, pair_body, initial)
    return jax.lax.fori_loop(pairs * 2, upper, body, result)


def static_fori_loop(
    n_steps: int,
    body: Callable[[jnp.ndarray, Any], Any],
    initial: Any,
    *,
    unroll: int | None,
) -> Any:
    """Unroll the loop body explicitly; Triton cannot lower scan(unroll > 1)."""
    if unroll is None or unroll is False:
        factor = 1
    elif unroll is True:
        factor = max(1, n_steps)
    else:
        factor = operator.index(unroll)
        if factor < 0:
            raise ValueError("unroll must be non-negative")
        factor = min(factor, n_steps) if factor else n_steps
        factor = max(1, factor)
    blocks, tail = divmod(n_steps, factor)

    def block_body(block, state):
        for offset in range(factor):
            state = body(block * factor + offset, state)
        return state

    result = jax.lax.fori_loop(0, blocks, block_body, initial)
    for offset in range(tail):
        result = body(jnp.int32(blocks * factor + offset), result)
    return result
