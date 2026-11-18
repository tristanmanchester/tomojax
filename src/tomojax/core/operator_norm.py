"""Array-only ray-operator norm estimates and bounds shared by solver families."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from tomojax.core.backend_policy import cuda_backend
from tomojax.core.projector import (
    project_views,
    sum_backproject_views_T,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomojax.core.geometry.base import Detector, Grid


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
            projected = project_views(
                batch,
                grid,
                detector,
                masked,
                det_grid=det_grid,
                ray_integrator=ray_integrator,
                unroll=unroll,
                use_checkpoint=checkpoint,
                gather_dtype=gather_dtype,
            )
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

    if upper_bound:
        return jnp.max(apply_normal(jnp.ones_like(initial)))
    return normal_upper_bound(
        apply_normal, initial, iterations=iterations, nonnegative=ray_integrator != "joseph_cubic"
    )
