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
    params = jnp.asarray(params, dtype=jnp.float32)
    step_scale = jnp.asarray(step_size, dtype=jnp.float32)
    columns = []
    for index in range(int(params.shape[0])):
        direction = jnp.zeros_like(params).at[index].set(1.0)
        step = step_scale * direction
        columns.append((function(params + step) - function(params - step)) / (2.0 * step_scale))
    return jnp.stack(columns, axis=1)
