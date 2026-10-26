"""FISTA starting images and data-term norm work, including opt-in ordered subsets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from ._projection import (
    least_squares_operators,
    normal_operator_norm,
    positive_normal_bound,
    projection_operators,
)

if TYPE_CHECKING:
    from tomojax.geometry import Detector, Grid

    from .fista_tv import FistaConfig, FistaRuntime


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
    assert runtime.projector is not None
    model, backend, batch = runtime.projector
    mask = None if runtime.volume_mask is None else jnp.asarray(runtime.volume_mask, jnp.float32)
    split = runtime.split
    value, gradient = least_squares_operators(
        runtime.poses, grid, detector, backend, batch, model, stream=runtime.stream, split=split
    )(None, projections)
    forward, adjoint = projection_operators(
        runtime.poses, grid, detector, None, backend, batch, model, split=split
    )
    lipschitz = normal_operator_norm(
        forward, adjoint, (grid.nz, grid.ny, grid.nx), iterations=int(cfg.power_iterations),
        mask=mask, start=power_start(gradient),
    )  # fmt: skip
    c = runtime.constraints
    zero_start = runtime.x0 is None and (c.lower_bound is None or c.lower_bound <= 0.0)
    zero_start = zero_start and (c.upper_bound is None or c.upper_bound >= 0.0)
    if not zero_start:
        return lipschitz, value, None
    return lipschitz, value, gradient if mask is None else gradient * mask


def initial_view_order(cfg: FistaConfig, geometry: object) -> jax.Array | None:
    """Balance subsets in nominal angle order, not an incidental file ordering."""
    angles = getattr(geometry, "angles", None)
    if not cfg.initial_epochs or angles is None:
        return None
    order = np.argsort(angles, kind="stable")
    return None if np.array_equal(order, np.arange(len(order))) else jnp.asarray(order)


def validate_initialization(
    cfg: FistaConfig,
    projector: tuple[object, str, int] | None,
    n_views: int,
    *,
    stream: bool,
    split: object,
) -> None:
    """Keep the optional initializer's supported objective and storage explicit."""
    if not cfg.initial_epochs:
        return
    if (
        not cfg.nonnegative
        or cfg.tv_weight
        or cfg.support is not None
        or cfg.lower_bound not in (None, 0)
        or cfg.upper_bound is not None
        or cfg.iterations < 1
    ):
        raise ValueError(
            "initial_epochs requires unmasked nonnegative least squares and final iterations"
        )
    if stream or split is not None or projector is None or projector[0] != "joseph":
        raise ValueError(
            "initial_epochs requires resident, single-device parallel Joseph operators"
        )
    if cfg.initial_subsets > n_views:
        raise ValueError("initial_subsets must not exceed the number of views")


def ordered_data(data: jax.Array, runtime: FistaRuntime) -> tuple[jax.Array, jax.Array]:
    """Nominal-angle-ordered poses and measurements, without a padded image stack."""
    if runtime.initial_order is None:
        return runtime.poses, data
    return runtime.poses[runtime.initial_order], data[runtime.initial_order]


def subset_order(subsets: int) -> jax.Array:
    """Visit interleaved angular groups in bit-reversal order."""
    bits = (subsets - 1).bit_length()
    return jnp.asarray(sorted(range(subsets), key=lambda i: int(f"{i:0{bits}b}"[::-1], 2)))


def ordered_start(
    grid: Grid, detector: Detector, data: jax.Array, runtime: FistaRuntime
) -> tuple[jax.Array, jax.Array]:
    """Initialize with positive subset steps; return a full-operator norm bound.

    Every view occurs once per epoch, including an uneven last group. The sum
    of the positive per-group norm bounds bounds the full normal operator.
    Subset acceleration alone can cycle; the caller must refine with all data.
    """
    cfg = runtime.config
    assert runtime.projector is not None
    _, backend, _ = runtime.projector
    n, subsets = runtime.poses.shape[0], cfg.initial_subsets
    nominal, data = ordered_data(data, runtime)
    count = (n + subsets - 1) // subsets
    shape = (grid.nz, grid.ny, grid.nx)
    indices = jnp.arange(subsets)[:, None] + subsets * jnp.arange(count)[None, :]
    valid = (indices < n)[:, :, None, None]
    if n % subsets:
        poses = nominal[jnp.minimum(indices, n - 1)]
        images = data[jnp.minimum(indices, n - 1)]
    else:
        poses = nominal.reshape(count, subsets, 4, 4).swapaxes(0, 1)
        images = data.reshape(count, subsets, detector.nv, detector.nu).swapaxes(0, 1)
    order = subset_order(subsets)

    def operators(p, mask):
        forward, adjoint = projection_operators(p, grid, detector, None, backend, count, "joseph")
        if n % subsets:
            return lambda x: forward(x) * mask, lambda y: adjoint(y * mask)
        return forward, adjoint

    def bound(group):
        forward, adjoint = operators(*group)
        return positive_normal_bound(forward, adjoint, shape, iterations=1)

    limits = jax.lax.map(bound, (poses, valid))

    def step(state, k):
        x, z, t = state
        i = order[k % subsets]
        if n % subsets:
            forward, adjoint = operators(poses[i], valid[i])
            gradient = adjoint(forward(z) - images[i])
        else:
            _, gradient = least_squares_operators(
                poses[i], grid, detector, backend, count, "joseph"
            )(z, images[i])
        new = jnp.maximum(z - gradient / jnp.where(limits[i] > 0, limits[i], 1), 0)
        tn = (1 + jnp.sqrt(1 + 4 * t * t)) / 2
        return (new, new + (t - 1) / tn * (new - x), tn), None

    initial = jnp.zeros(shape, jnp.float32) if runtime.x0 is None else jnp.maximum(runtime.x0, 0)
    state, _ = jax.lax.scan(
        step, (initial, initial, jnp.float32(1)), jnp.arange(cfg.initial_epochs * subsets)
    )
    total = jnp.sum(limits)
    return state[0], jnp.where(total > 0, total, 1)
