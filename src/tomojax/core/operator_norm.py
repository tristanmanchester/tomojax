"""Array-only ray-operator norm estimates and bounds shared by solver families."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from tomojax.core.backend_policy import cuda_backend
from tomojax.core.projector import forward_project_view_T, sum_backproject_views_T

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Grid


def unit(vector: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``vector`` scaled to unit length, and its length; zero stays zero (no floor)."""
    length = jnp.linalg.norm(vector.ravel())
    return vector / jnp.where(length == 0, 1.0, length), length


def _joseph_normal(
    poses: jax.Array,
    volume: jax.Array,
    valid: jax.Array,
    grid: Grid,
    detector: Detector,
    det_grid: tuple[jax.Array, jax.Array] | None,
    *,
    interpolation: str,
    absolute_weights: bool,
    accumulate: jax.Array,
    unit_volume: bool,
) -> jax.Array:
    """Apply the Joseph normal or its componentwise absolute-weight bound."""
    from .joseph import (
        forward_project_planes,
        plane_coefficients,
        project_ones_native,
        sum_backproject_planes,
    )

    coefficients = plane_coefficients(poses, grid, detector, det_grid)
    # Keep both sides on the bounded-workspace pair, including unit-volume
    # forwards used by the row-sum bound.
    backend = "cuda" if cuda_backend() else "jax"
    projected = None
    if unit_volume and backend == "cuda" and interpolation == "linear":
        projected = project_ones_native(coefficients, grid, detector)
    if projected is None:
        projected = forward_project_planes(
            coefficients, volume, grid, detector,
            backend=backend, interpolation=interpolation, absolute_weights=absolute_weights,
        )  # fmt: skip
    return sum_backproject_planes(
        coefficients, projected * valid, grid, detector,
        backend=backend, interpolation=interpolation, absolute_weights=absolute_weights,
        accumulate=accumulate,
    )  # fmt: skip


@partial(
    jax.jit,
    static_argnames=(
        "grid",
        "detector",
        "batch_size",
        "iterations",
        "unroll",
        "checkpoint",
        "gather_dtype",
        "upper_bound",
        "ray_integrator",
    ),
)
def estimate_normal_norm(
    poses: jnp.ndarray,
    initial: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    support: jnp.ndarray | None,
    *,
    grid: Grid,
    detector: Detector,
    batch_size: int,
    iterations: int,
    unroll: int,
    checkpoint: bool,
    gather_dtype: str,
    upper_bound: bool = False,
    ray_integrator: str = "sampled",
) -> jnp.ndarray:
    """Estimate the largest eigenvalue of ``(A M)* (A M)`` by power iteration.

    Geometry and masks are dynamic inputs. Repeated reconstructions therefore
    reuse compiled code without capturing old measurements or geometry values.

    With ``upper_bound=True``, use the maximum row sum of the nonnegative
    normal operator instead of power iteration. One forward/adjoint pair gives
    an upper bound for nonnegative ray models. Cubic Joseph interpolation
    instead uses ``|A|.T |A|`` to bound its negative lobes. Absolute support
    weights also bound signed masks. This mode ignores ``initial`` values and
    ``iterations``; its physical units are squared length, just like ``A.T A``.
    """
    if upper_bound and support is not None:
        support = jnp.abs(support)
    n = int(poses.shape[0])
    b = max(1, min(batch_size, n))
    chunks = (n + b - 1) // b

    def apply_normal(volume: jnp.ndarray) -> jnp.ndarray:
        masked = volume if support is None else volume * support

        def body(acc: jnp.ndarray, chunk: jnp.ndarray) -> tuple[jnp.ndarray, None]:
            start = chunk * b
            valid = jnp.minimum(b, n - start)
            shifted = jnp.maximum(0, start - (b - valid))
            batch = jax.lax.dynamic_slice_in_dim(poses, shifted, b)
            mask = (jnp.arange(b) >= b - valid)[:, None, None]
            if ray_integrator in {"joseph", "joseph_cubic"}:
                return _joseph_normal(
                    batch,
                    masked,
                    mask,
                    grid,
                    detector,
                    det_grid,
                    interpolation="cubic" if ray_integrator == "joseph_cubic" else "linear",
                    absolute_weights=upper_bound and ray_integrator == "joseph_cubic",
                    accumulate=acc,
                    unit_volume=upper_bound and support is None,
                ), None
            projected = jax.vmap(
                lambda t: forward_project_view_T(
                    t,
                    grid,
                    detector,
                    masked,
                    det_grid=det_grid,
                    ray_integrator=ray_integrator,
                    unroll=unroll,
                    use_checkpoint=checkpoint,
                    gather_dtype=gather_dtype,
                )
            )(batch)
            adjoint = sum_backproject_views_T(
                batch,
                grid,
                detector,
                projected * mask,
                det_grid=det_grid,
                ray_integrator=ray_integrator,
                unroll=unroll,
                gather_dtype=gather_dtype,
            )
            return acc + adjoint, None

        result, _ = jax.lax.scan(body, jnp.zeros_like(volume), jnp.arange(chunks))
        return result if support is None else result * support

    def normalize(volume: jnp.ndarray) -> jnp.ndarray:
        return unit(volume)[0]

    if upper_bound:
        return jnp.max(apply_normal(jnp.ones_like(initial)))

    vector = jax.lax.fori_loop(
        0, max(1, iterations), lambda _, v: normalize(apply_normal(v)), normalize(initial)
    )
    return jnp.vdot(vector, apply_normal(vector)).real
