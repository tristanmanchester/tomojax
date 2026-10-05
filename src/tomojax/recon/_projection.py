"""Batched projection operators shared by the iterative reconstruction solvers.

``projection_operators`` returns a matched forward projector and transpose over
all views, processed in fixed-size view batches inside one compiled loop.
``"joseph"`` samples voxel-centre planes along each ray's dominant axis and is
the fastest model on CUDA; ``"ray"`` is the trilinear ray marcher. Both use
physical units and agree with analytic line integrals to the same accuracy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp

from tomojax.core.projector import forward_project_view_T, sum_backproject_views_T

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomojax.geometry import Detector, Grid

type ProjectorModel = Literal["auto", "joseph", "ray"]
type ProjectorBackend = Literal["auto", "jax", "pallas"]


def cuda_available() -> bool:
    """Return whether JAX's default device is a CUDA GPU."""
    return (
        jax.default_backend() == "gpu"
        and "cuda" in jax.devices()[0].client.platform_version.lower()
    )


def resolve_projector(
    model: str,
    backend: str,
    *,
    det_grid: object = None,
    context: str,
) -> tuple[str, str]:
    """Resolve ``auto`` choices; explicit choices must be supported, never replaced.

    ``auto`` selects Joseph plane sampling with Pallas kernels on CUDA. Explicit
    detector grids need the ray model, and Pallas needs CUDA and the canonical
    detector grid.
    """
    if model not in {"auto", "joseph", "ray"}:
        raise ValueError(f"{context}: projector_model must be 'auto', 'joseph' or 'ray'")
    if backend not in {"auto", "jax", "pallas"}:
        raise ValueError(f"{context}: projector_backend must be 'auto', 'jax' or 'pallas'")
    if model == "auto":
        model = "ray" if det_grid is not None else "joseph"
    if model == "joseph" and det_grid is not None:
        raise ValueError(f"{context}: Joseph sampling requires the canonical detector grid")
    cuda = cuda_available()
    if backend == "auto":
        backend = "pallas" if cuda and det_grid is None else "jax"
    if backend == "pallas" and (not cuda or det_grid is not None):
        raise ValueError(f"{context}: Pallas requires CUDA and the canonical detector grid")
    return model, backend


def projection_operators(
    poses: jax.Array,
    grid: Grid,
    detector: Detector,
    det_grid: tuple[jax.Array, jax.Array] | None,
    backend: str,
    batch_size: int,
    model: str = "ray",
    joseph_interpolation: str = "linear",
    absolute_weights: bool = False,
) -> tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:
    """Return matched ``(forward, adjoint)`` over all views in fixed-size batches.

    ``absolute_weights`` returns ``abs(A).T`` instead of the adjoint, for
    cancellation bounds with negative interpolation lobes.
    """
    n = int(poses.shape[0])
    batch_size = min(batch_size, n)
    chunks = (n + batch_size - 1) // batch_size
    if model == "joseph":
        from tomojax.core.joseph import (
            forward_project_planes,
            plane_coefficients,
            sum_backproject_planes,
        )

        coefficients = plane_coefficients(poses, grid, detector)

    def select(chunk: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        start = jnp.minimum(chunk * batch_size, n - batch_size)
        batch = jax.lax.dynamic_slice(poses, (start, 0, 0), (batch_size, 4, 4))
        # The final shifted batch overlaps earlier views. Its adjoint must count
        # only new views; forward writes the same overlapping values again.
        valid = start + jnp.arange(batch_size) >= chunk * batch_size
        return start, batch, valid

    def forward(volume: jax.Array) -> jax.Array:
        def body(chunk: jax.Array, output: jax.Array) -> jax.Array:
            start, batch, _ = select(chunk)
            if model == "joseph":
                coeff = jax.lax.dynamic_slice(coefficients, (start, 0), (batch_size, 14))
                projected = forward_project_planes(
                    coeff,
                    volume,
                    grid,
                    detector,
                    backend=backend,
                    interpolation=joseph_interpolation,
                )
            elif backend == "pallas":
                from tomojax.core.pallas.api import (
                    PallasProjectorOptions,
                    forward_project_views_T_pallas,
                )

                projected = forward_project_views_T_pallas(
                    batch,
                    grid,
                    detector,
                    volume,
                    options=PallasProjectorOptions(tile_shape=(16, 4), num_warps=1),
                )
            else:
                projected = jax.vmap(
                    lambda t: forward_project_view_T(t, grid, detector, volume, det_grid=det_grid)
                )(batch)
            return jax.lax.dynamic_update_slice(output, projected, (start, 0, 0))

        return jax.lax.fori_loop(
            0, chunks, body, jnp.zeros((n, detector.nv, detector.nu), dtype=jnp.float32)
        )

    def adjoint(images: jax.Array) -> jax.Array:
        def body(chunk: jax.Array, output: jax.Array) -> jax.Array:
            start, batch, valid = select(chunk)
            data = jax.lax.dynamic_slice(
                images, (start, 0, 0), (batch_size, detector.nv, detector.nu)
            )
            data = jnp.where(valid[:, None, None], data, 0.0)
            if model == "joseph":
                coeff = jax.lax.dynamic_slice(coefficients, (start, 0), (batch_size, 14))
                update = sum_backproject_planes(
                    coeff,
                    data,
                    grid,
                    detector,
                    backend=backend,
                    interpolation=joseph_interpolation,
                    absolute_weights=absolute_weights,
                )
            elif backend == "pallas":
                from tomojax.core.pallas.api import sum_backproject_views_T_pallas

                update = sum_backproject_views_T_pallas(
                    batch, grid, detector, data, tile_shape=(16, 4), num_warps=1
                )
            else:
                update = sum_backproject_views_T(batch, grid, detector, data, det_grid=det_grid)
            return output + update

        return jax.lax.fori_loop(
            0, chunks, body, jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
        )

    return forward, adjoint


def normal_operator_norm(
    forward: Callable[[jax.Array], jax.Array],
    adjoint: Callable[[jax.Array], jax.Array],
    shape: tuple[int, int, int],
    *,
    iters: int,
    mask: jax.Array | None = None,
) -> jax.Array:
    """Estimate ``||M A^T A M||`` by power iteration from a constant volume.

    Power iteration approaches the largest eigenvalue from below; callers
    needing a safe step size should add a margin.
    """

    def normal(x: jax.Array) -> jax.Array:
        x = x if mask is None else x * mask
        y = adjoint(forward(x))
        return y if mask is None else y * mask

    def step(_: jax.Array, carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        x, _ = carry
        y = normal(x)
        norm = jnp.sqrt(jnp.sum(y * y))
        return y / jnp.maximum(norm, jnp.finfo(jnp.float32).tiny), norm

    x0 = jnp.ones(shape, jnp.float32)
    x0 = x0 / jnp.sqrt(jnp.sum((x0 if mask is None else x0 * mask) ** 2) + 1e-30)
    _, norm = jax.lax.fori_loop(0, max(1, iters), step, (x0, jnp.float32(0)))
    return norm
