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
from tomojax.recon._host_stream import read_views

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


class _Batches:
    """Fixed-size view batches over one pose stack, with matched batch operators."""

    def __init__(
        self,
        poses: jax.Array,
        grid: Grid,
        detector: Detector,
        det_grid: tuple[jax.Array, jax.Array] | None,
        backend: str,
        batch_size: int,
        model: str,
        joseph_interpolation: str,
        absolute_weights: bool,
    ) -> None:
        self.n = int(poses.shape[0])
        self.size = min(batch_size, self.n)
        self.count = (self.n + self.size - 1) // self.size
        self.poses, self.grid, self.detector, self.det_grid = poses, grid, detector, det_grid
        self.backend, self.model = backend, model
        self.interpolation, self.absolute_weights = joseph_interpolation, absolute_weights
        if model == "joseph":
            from tomojax.core.joseph import plane_coefficients

            self.coefficients = plane_coefficients(poses, grid, detector)

    def select(self, chunk: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        start = jnp.minimum(chunk * self.size, self.n - self.size)
        batch = jax.lax.dynamic_slice(self.poses, (start, 0, 0), (self.size, 4, 4))
        # The final shifted batch overlaps earlier views. Its adjoint must count
        # only new views; forward writes the same overlapping values again.
        valid = start + jnp.arange(self.size) >= chunk * self.size
        return start, batch, valid

    def images(self, stack: jax.Array, start: jax.Array) -> jax.Array:
        shape = (self.size, self.detector.nv, self.detector.nu)
        return jax.lax.dynamic_slice(stack, (start, 0, 0), shape)

    def _coefficients(self, start: jax.Array) -> jax.Array:
        return jax.lax.dynamic_slice(self.coefficients, (start, 0), (self.size, 14))

    def project(self, volume: jax.Array, start: jax.Array, batch: jax.Array) -> jax.Array:
        grid, detector = self.grid, self.detector
        if self.model == "joseph":
            from tomojax.core.joseph import forward_project_planes

            return forward_project_planes(
                self._coefficients(start),
                volume,
                grid,
                detector,
                backend=self.backend,
                interpolation=self.interpolation,
            )
        if self.backend == "pallas":
            from tomojax.core.pallas.api import (
                PallasProjectorOptions,
                forward_project_views_T_pallas,
            )

            return forward_project_views_T_pallas(
                batch,
                grid,
                detector,
                volume,
                options=PallasProjectorOptions(tile_shape=(16, 4), num_warps=1),
            )
        return jax.vmap(
            lambda t: forward_project_view_T(t, grid, detector, volume, det_grid=self.det_grid)
        )(batch)

    def backproject(
        self, images: jax.Array, start: jax.Array, batch: jax.Array, output: jax.Array
    ) -> jax.Array:
        """Return ``output`` plus the batch transpose; Joseph on CUDA adds in place."""
        grid, detector = self.grid, self.detector
        if self.model == "joseph":
            from tomojax.core.joseph import sum_backproject_planes

            return sum_backproject_planes(
                self._coefficients(start),
                images,
                grid,
                detector,
                backend=self.backend,
                interpolation=self.interpolation,
                absolute_weights=self.absolute_weights,
                accumulate=output,
            )
        if self.backend == "pallas":
            from tomojax.core.pallas.api import sum_backproject_views_T_pallas

            update = sum_backproject_views_T_pallas(
                batch, grid, detector, images, tile_shape=(16, 4), num_warps=1
            )
        else:
            update = sum_backproject_views_T(batch, grid, detector, images, det_grid=self.det_grid)
        return output + update

    def zeros_volume(self) -> jax.Array:
        return jnp.zeros((self.grid.nx, self.grid.ny, self.grid.nz), dtype=jnp.float32)


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
) -> tuple[Callable[[jax.Array], jax.Array], Callable[..., jax.Array]]:
    """Return matched ``(forward, adjoint)`` over all views in fixed-size batches.

    ``absolute_weights`` returns ``abs(A).T`` instead of the adjoint, for
    cancellation bounds with negative interpolation lobes. ``adjoint(stacks,
    combine)`` applies ``combine`` to batches of several stacks, so a derived
    sinogram such as ``abs(a) + abs(b)`` is never stored whole; ``accumulate``
    adds the result to an existing volume, in place for Joseph on CUDA.
    """
    ops = _Batches(
        poses,
        grid,
        detector,
        det_grid,
        backend,
        batch_size,
        model,
        joseph_interpolation,
        absolute_weights,
    )

    def forward(volume: jax.Array) -> jax.Array:
        def body(chunk: jax.Array, output: jax.Array) -> jax.Array:
            start, batch, _ = ops.select(chunk)
            projected = ops.project(volume, start, batch)
            return jax.lax.dynamic_update_slice(output, projected, (start, 0, 0))

        empty = jnp.zeros((ops.n, detector.nv, detector.nu), dtype=jnp.float32)
        return jax.lax.fori_loop(0, ops.count, body, empty)

    def adjoint(
        images: jax.Array | tuple[jax.Array, ...],
        combine: Callable[..., jax.Array] | None = None,
        accumulate: jax.Array | None = None,
    ) -> jax.Array:
        stacks = (images,) if combine is None else tuple(images)
        initial = ops.zeros_volume() if accumulate is None else accumulate

        def body(chunk: jax.Array, output: jax.Array) -> jax.Array:
            start, batch, valid = ops.select(chunk)
            slices = [ops.images(a, start) for a in stacks]
            data = slices[0] if combine is None else combine(*slices)
            data = jnp.where(valid[:, None, None], data, 0.0)
            return ops.backproject(data, start, batch, output)

        return jax.lax.fori_loop(0, ops.count, body, initial)

    return forward, adjoint


def least_squares_operators(
    poses: jax.Array,
    grid: Grid,
    detector: Detector,
    backend: str,
    batch_size: int,
    model: str = "ray",
    joseph_interpolation: str = "linear",
    *,
    stream: bool = False,
) -> tuple[
    Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
    Callable[[jax.Array, jax.Array], jax.Array],
]:
    """Return ``(value_and_gradient, value)`` of ``0.5 ||A x - y||^2``.

    Each view batch is projected, compared with the data and transposed before
    the next, so no sinogram-sized intermediate is ever stored. With
    ``stream``, ``y`` is the key of a host array registered with
    :func:`host_source`, read one batch at a time.
    """
    ops = _Batches(
        poses, grid, detector, None, backend, batch_size, model, joseph_interpolation, False
    )

    def batch_residual(volume: jax.Array, data: jax.Array, chunk: jax.Array) -> tuple:
        start, batch, valid = ops.select(chunk)
        if stream:
            measured = read_views(data, start, (ops.size, detector.nv, detector.nu))
        else:
            measured = ops.images(data, start)
        residual = ops.project(volume, start, batch) - measured
        return start, batch, jnp.where(valid[:, None, None], residual, 0.0)

    def squared(residual: jax.Array) -> jax.Array:
        return 0.5 * jnp.vdot(residual, residual).real

    def value_and_gradient(volume: jax.Array, data: jax.Array) -> tuple[jax.Array, jax.Array]:
        def body(chunk: jax.Array, carry: tuple) -> tuple:
            value, gradient = carry
            start, batch, residual = batch_residual(volume, data, chunk)
            return value + squared(residual), ops.backproject(residual, start, batch, gradient)

        return jax.lax.fori_loop(0, ops.count, body, (jnp.float32(0), ops.zeros_volume()))

    def value(volume: jax.Array, data: jax.Array) -> jax.Array:
        def body(chunk: jax.Array, total: jax.Array) -> jax.Array:
            return total + squared(batch_residual(volume, data, chunk)[2])

        return jax.lax.fori_loop(0, ops.count, body, jnp.float32(0))

    return value_and_gradient, value


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


def normal_equation_operators(
    poses: jax.Array,
    grid: Grid,
    detector: Detector,
    backend: str,
    batch_size: int,
    model: str = "ray",
    joseph_interpolation: str = "linear",
) -> tuple[
    Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
]:
    """Return ``(normal, gradient)`` streaming data from a :func:`host_source` key.

    ``normal(d)`` returns ``(||A d||^2, A^T A d)`` in one pass over view batches,
    touching no measured data. ``gradient(x, key)`` returns ``A^T (y - A x)`` and
    the cancellation scale ``|A|^T (|y| + |A x|)``, reading each batch of ``y``
    from the host. Only volume-sized arrays and one batch are ever on the device.
    """
    ops = _Batches(
        poses, grid, detector, None, backend, batch_size, model, joseph_interpolation, False
    )
    magnitude = (
        _Batches(
            poses, grid, detector, None, backend, batch_size, model, joseph_interpolation, True
        )
        if model == "joseph" and joseph_interpolation == "cubic"
        else ops
    )
    shape = (ops.size, detector.nv, detector.nu)

    def normal(direction: jax.Array) -> tuple[jax.Array, jax.Array]:
        def body(
            chunk: jax.Array, carry: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            energy, total = carry
            start, batch, valid = ops.select(chunk)
            projected = jnp.where(valid[:, None, None], ops.project(direction, start, batch), 0.0)
            energy = energy + jnp.vdot(projected, projected).real
            return energy, ops.backproject(projected, start, batch, total)

        return jax.lax.fori_loop(0, ops.count, body, (jnp.float32(0), ops.zeros_volume()))

    def gradient(volume: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        def body(
            chunk: jax.Array, carry: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            total, scale = carry
            start, batch, valid = ops.select(chunk)
            measured = read_views(key, start, shape)
            predicted = ops.project(volume, start, batch)
            mask = valid[:, None, None]
            residual = jnp.where(mask, measured - predicted, 0.0)
            size = jnp.where(mask, jnp.abs(measured) + jnp.abs(predicted), 0.0)
            return (
                ops.backproject(residual, start, batch, total),
                magnitude.backproject(size, start, batch, scale),
            )

        zero = ops.zeros_volume()
        return jax.lax.fori_loop(0, ops.count, body, (zero, zero))

    return normal, gradient
