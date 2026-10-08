"""Array-only ray-operator norm estimates and bounds shared by solver families."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from tomojax.core.projector import forward_project_view_T, sum_backproject_views_T

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Grid


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
    an upper bound for the nonnegative trilinear ray model. Absolute support
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
            batch = jax.lax.dynamic_slice(poses, (shifted, 0, 0), (b, 4, 4))
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
            mask = (jnp.arange(b) >= b - valid)[:, None, None]
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
        return volume / (jnp.linalg.norm(volume.ravel()) + 1e-12)

    if upper_bound:
        return jnp.maximum(jnp.max(apply_normal(jnp.ones_like(initial))), jnp.float32(1e-6))

    vector = jax.lax.fori_loop(
        0, max(1, iterations), lambda _, v: normalize(apply_normal(v)), normalize(initial)
    )
    return jnp.maximum(jnp.vdot(vector, apply_normal(vector)).real, jnp.float32(1e-6))
