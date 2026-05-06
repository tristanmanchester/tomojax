"""Shared numerical helpers for alignment LM/GN solvers."""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable


def finite_difference_jacobian(
    function: Callable[[jax.Array], jax.Array],
    params: jax.Array,
    *,
    step_size: float,
) -> jax.Array:
    eye = jnp.eye(params.shape[0], dtype=jnp.float32)

    def one_column(direction: jax.Array) -> jax.Array:
        step = jnp.asarray(step_size, dtype=jnp.float32) * direction
        return (function(params + step) - function(params - step)) / (
            2.0 * jnp.asarray(step_size, dtype=jnp.float32)
        )

    return jax.vmap(one_column)(eye).T
