"""FISTA's first gradient and the start of its operator-norm estimate."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from tomojax.core.operator_norm import normal_upper_bound

from ._projection import (
    least_squares_operators,
    normal_equation_operators,
    projection_operators,
)

if TYPE_CHECKING:
    from tomojax.geometry import Detector, Grid

    from .fista_tv import FistaRuntime


def power_start(backprojection: jax.Array) -> jax.Array:
    """Where power iteration for ``||A^T A||`` starts: ``|A^T y|``, plus a little constant.

    The backprojected data overlap the leading eigenvector (a smooth, positive
    volume) far more than a constant does: on the FIPS walnut three iterations from
    it come closer than five from a constant. The constant keeps every voxel in play
    and covers zero data.
    """
    size = jnp.abs(backprojection)
    peak = jnp.max(size)
    return size + jnp.where(peak > 0, 1e-3 * peak, 1.0)


def first_gradient(
    grid: Grid, detector: Detector, projections: jnp.ndarray, runtime: FistaRuntime
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """The Lipschitz constant, and the data term and gradient at zero when FISTA starts there.

    The gradient at zero, minus the backprojected data, needs no projection of
    its own, and it starts the power iteration (see :func:`power_start`).
    """
    cfg = runtime.config
    model, backend, batch = runtime.projector
    mask = None if runtime.volume_mask is None else jnp.asarray(runtime.volume_mask, jnp.float32)
    split = runtime.split
    value, gradient = least_squares_operators(
        runtime.poses,
        grid,
        detector,
        backend,
        batch,
        model,
        stream=runtime.stream,
        split=split,
        data_term=cfg.data_term,
        det_grid=runtime.detector_grid,
    )(None, projections)
    if split is None:
        normal, _ = normal_equation_operators(
            runtime.poses, grid, detector, backend, batch, model, det_grid=runtime.detector_grid
        )

        def apply_normal(x: jax.Array) -> jax.Array:
            return normal(x)[1]
    else:
        forward, adjoint = projection_operators(
            runtime.poses, grid, detector, runtime.detector_grid, backend, batch, model, split=split
        )

        def apply_normal(x: jax.Array) -> jax.Array:
            return adjoint(forward(x))

    def masked_normal(x: jax.Array) -> jax.Array:
        y = apply_normal(x if mask is None else x * mask)
        return y if mask is None else y * mask

    start = power_start(gradient)
    lipschitz = normal_upper_bound(
        masked_normal, start if mask is None else start * mask,
        iterations=int(cfg.power_iterations),
    )  # fmt: skip
    c = runtime.constraints
    zero_start = runtime.x0 is None and (c.lower_bound is None or c.lower_bound <= 0.0)
    zero_start = zero_start and (c.upper_bound is None or c.upper_bound >= 0.0)
    if not zero_start:
        return lipschitz, value, None
    return lipschitz, value, gradient if mask is None else gradient * mask
