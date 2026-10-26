"""One pose stack's matched batch calls and volume-scoped forward preparation.

Solver loops own iteration and reduction; this module owns view selection,
shifted-tail counting, geometry coefficients and prepared-volume lifetimes.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from tomojax.core.cone import (
    ConeModel,
    cone_project,
    cone_project_ones,
)
from tomojax.core.projector import forward_project_view_T, sum_backproject_views_T

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomojax.geometry import Detector, Grid


class ViewBatches:
    """Fixed-size view batches over one pose stack, with matched batch operators."""

    def __init__(
        self,
        poses: jax.Array,
        grid: Grid,
        detector: Detector,
        det_grid: tuple[jax.Array, jax.Array] | None,
        backend: str,
        batch_size: int,
        model: str | ConeModel,
        joseph_interpolation: str,
        *,
        absolute_weights: bool,
    ) -> None:
        self.batch_size = batch_size
        self.lay_out(int(poses.shape[0]))
        self.poses, self.grid, self.detector, self.det_grid = poses, grid, detector, det_grid
        self.counts: jax.Array | None = None  # on a device's share: which views count
        self.coefficients: jax.Array | None = None
        self.backend, self.model = backend, model
        self.interpolation, self.absolute_weights = joseph_interpolation, absolute_weights
        if isinstance(model, ConeModel):
            # Cone weights are non-negative, so |A|^T is the transpose itself.
            self.coefficients = model.coefficients(poses, grid)
            self.cone_backend = backend
        elif model == "joseph":
            from tomojax.core.joseph import plane_coefficients

            self.coefficients = plane_coefficients(poses, grid, detector)

    def lay_out(self, n: int) -> None:
        """Batch ``n`` views."""
        self.n = n
        self.size = min(self.batch_size, n)
        self.count = (n + self.size - 1) // self.size

    def per_view(self) -> tuple[jax.Array, ...]:
        """The per-view arrays a device's share of the views is cut from."""
        return (self.poses,) if self.coefficients is None else (self.poses, self.coefficients)

    def share(self, per_view: tuple[jax.Array, ...], counts: jax.Array) -> ViewBatches:
        """These operators over one device's ``per_view`` arrays, of which ``counts`` count."""
        share = copy.copy(self)
        share.poses, *coefficients = per_view
        share.coefficients = coefficients[0] if coefficients else None
        share.lay_out(int(share.poses.shape[0]))
        share.counts = counts
        return share

    def select(self, chunk: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        start = jnp.minimum(chunk * self.size, self.n - self.size)
        batch = jax.lax.dynamic_slice(self.poses, (start, 0, 0), (self.size, 4, 4))
        # The final shifted batch overlaps earlier views. Its adjoint must count
        # only new views; forward writes the same overlapping values again.
        valid = start + jnp.arange(self.size) >= chunk * self.size
        if self.counts is not None:
            valid = valid & self.counted(start)
        return start, batch, valid

    def counted(self, start: jax.Array) -> jax.Array:
        """Which views of the batch at ``start`` count, on a device's share."""
        assert self.counts is not None
        return jax.lax.dynamic_slice(self.counts, (start,), (self.size,))

    def images(self, stack: jax.Array, start: jax.Array) -> jax.Array:
        shape = (self.size, self.detector.nv, self.detector.nu)
        return jax.lax.dynamic_slice(stack, (start, 0, 0), shape)

    def _coefficients(self, start: jax.Array) -> jax.Array:
        assert self.coefficients is not None
        return jax.lax.dynamic_slice_in_dim(self.coefficients, start, self.size)

    def project_ones(self, start: jax.Array, batch: jax.Array) -> jax.Array:
        """Project uniform unit support, retaining the ordinary weights at its boundary."""
        if isinstance(self.model, ConeModel):
            return cone_project_ones(
                self._coefficients(start), self.grid, self.detector, backend=self.cone_backend
            )
        if self.model == "joseph" and self.backend == "cuda":
            from tomojax.core.joseph import project_ones_native

            native = project_ones_native(self._coefficients(start), self.grid, self.detector)
            if native is not None:
                return native
        return self.project(jnp.ones_like(self.zeros_volume()), start, batch)

    def project(self, volume: jax.Array, start: jax.Array, batch: jax.Array) -> jax.Array:
        grid, detector = self.grid, self.detector
        if isinstance(self.model, ConeModel):
            return cone_project(
                volume, self._coefficients(start), grid, detector, backend=self.cone_backend
            )
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
        if self.backend == "cuda":
            from tomojax.core.cuda_projector.api import (
                CudaProjectorOptions,
                forward_project_views_T_cuda,
            )

            return forward_project_views_T_cuda(
                batch,
                grid,
                detector,
                volume,
                options=CudaProjectorOptions(tile_shape=(16, 4), num_warps=1),
            )
        return jax.vmap(
            lambda t: forward_project_view_T(t, grid, detector, volume, det_grid=self.det_grid)
        )(batch)

    def projector(self, volume: jax.Array) -> Callable[[jax.Array, jax.Array], jax.Array]:
        """Bind a logical projection's volume, sharing preparation across its batches."""
        return lambda start, batch: self.project(volume, start, batch)

    def backproject(
        self, images: jax.Array, start: jax.Array, batch: jax.Array, output: jax.Array
    ) -> jax.Array:
        """Return ``output`` plus the batch transpose; Joseph on CUDA adds in place."""
        grid, detector = self.grid, self.detector
        if isinstance(self.model, ConeModel):
            return self.model.backproject(
                images,
                self._coefficients(start),
                grid=grid,
                detector=detector,
                backend=self.cone_backend,
                accumulate=output,
            )
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
        if self.backend == "cuda":
            from tomojax.core.cuda_projector.api import sum_backproject_views_T_cuda

            update = sum_backproject_views_T_cuda(
                batch, grid, detector, images, tile_shape=(16, 4), num_warps=1
            )
        else:
            update = sum_backproject_views_T(batch, grid, detector, images, det_grid=self.det_grid)
        return output + update

    def backproject_pair(self, images, start, batch, outputs):
        """Two exact transposes with common batch geometry and separate accumulators."""
        if isinstance(self.model, ConeModel):
            return self.model.backproject_pair(
                images,
                self._coefficients(start),
                grid=self.grid,
                detector=self.detector,
                backend=self.cone_backend,
                accumulate=outputs,
            )
        return tuple(
            self.backproject(image, start, batch, out)
            for image, out in zip(images, outputs, strict=True)
        )

    def zeros_volume(self) -> jax.Array:
        return jnp.zeros((self.grid.nx, self.grid.ny, self.grid.nz), dtype=jnp.float32)
