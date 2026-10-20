"""Matched Joseph submatrices for bounded volume and detector arrays."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from ._plane_interpolation import cubic_weights_and_derivatives, validate_interpolation
from .joseph import select_axis

if TYPE_CHECKING:
    from .geometry.base import Detector, Grid


@dataclass(frozen=True, kw_only=True)
class PlaneRegion:
    """A submatrix of the full-grid plane projector, with integer storage origins.

    Coefficients always describe ``grid`` and ``detector``, not the local arrays.
    ``volume_origin`` is an (x, y, z) index; detector origins and shapes are (v, u).
    Forward values and pose derivatives use global FP32 coordinates before
    subtracting these integer origins to address a bounded array.

    Accumulators belong to the caller. CUDA can reuse their buffers when they
    are donated by an enclosing compiled call. No volume is stored by this object.
    """

    grid: Grid
    detector: Detector
    backend: str
    volume_origin: tuple[int, int, int] = (0, 0, 0)
    detector_origin: tuple[int, int] = (0, 0)
    detector_shape: tuple[int, int] | None = None
    interpolation: str = "linear"
    absolute_weights: bool = False

    def __post_init__(self) -> None:
        validate_interpolation(self.interpolation)
        if self.backend not in {"jax", "cuda"}:
            raise ValueError("Plane projection backend must be 'jax' or 'cuda'")

    @property
    def image_shape(self) -> tuple[int, int]:
        """The local detector window's (v, u) shape."""
        return self.detector_shape or (self.detector.nv, self.detector.nu)

    @partial(jax.jit, static_argnums=(0,))
    def project(
        self, coefficients: jax.Array, volume: jax.Array, *, accumulate: jax.Array | None = None
    ) -> jax.Array:
        """Project a local volume and optionally add to a detector-window accumulator."""
        if self.backend == "cuda":
            from .cuda_projector._joseph_region import columns

            tangent = jnp.zeros((coefficients.shape[0], 0, 14), jnp.float32)
            return columns(
                coefficients,
                volume,
                tangent,
                region=self,
                accumulate=None if accumulate is None else accumulate[:, None],
            )[:, 0]
        value = _forward(coefficients, volume, self)
        return value if accumulate is None else value + accumulate

    @partial(jax.jit, static_argnums=(0,))
    def transpose(
        self, coefficients: jax.Array, images: jax.Array, *, accumulate: jax.Array
    ) -> jax.Array:
        """Add the matched transpose to a local volume accumulator."""
        if self.backend == "cuda":
            from .cuda_projector._joseph_region import adjoint

            return adjoint(coefficients, images, accumulate, region=self)
        zero = jnp.zeros_like(accumulate)
        value = jax.linear_transpose(lambda x: _forward(coefficients, x, self), zero)(images)[0]
        return value + accumulate

    @partial(jax.jit, static_argnums=(0,))
    def columns(
        self,
        coefficients: jax.Array,
        volume: jax.Array,
        *,
        tangent: jax.Array,
        accumulate: jax.Array | None = None,
    ) -> jax.Array:
        """Return prediction and directional columns as (view, 1+direction, v, u).

        ``tangent`` contains (view, direction, 14) coefficient derivatives.
        """
        if self.backend == "cuda":
            from .cuda_projector._joseph_region import columns

            return columns(coefficients, volume, tangent, region=self, accumulate=accumulate)

        def project(c: jax.Array) -> jax.Array:
            return _forward(c, volume, self)

        derivative = jax.vmap(
            lambda t: jax.jvp(project, (coefficients,), (t,))[1],
            in_axes=1,
            out_axes=1,
        )(tangent)
        value = jnp.concatenate((project(coefficients)[:, None], derivative), axis=1)
        return value if accumulate is None else value + accumulate


def _forward(coefficients: jax.Array, volume: jax.Array, region: PlaneRegion) -> jax.Array:
    shape = volume.shape
    u, v = jnp.meshgrid(
        jnp.arange(region.image_shape[1]) + region.detector_origin[1],
        jnp.arange(region.image_shape[0]) + region.detector_origin[0],
    )

    def view(cf: jax.Array) -> jax.Array:
        axis = cf[0].astype(jnp.int32)
        start = select_axis(axis, *region.volume_origin)

        def plane(local: jax.Array, total: jax.Array) -> jax.Array:
            k = local + start
            qb = (cf[1] * u + cf[2] * v) + (cf[3] * k + cf[4])
            qc = (cf[5] * u + cf[6] * v) + (cf[7] * k + cf[8])
            ib, ic = jnp.floor(qb).astype(jnp.int32), jnp.floor(qc).astype(jnp.int32)
            wb, wc = qb - ib, qc - ic

            def sample(b: jax.Array, c: jax.Array) -> jax.Array:
                ix = select_axis(axis, k, c, b) - region.volume_origin[0]
                iy = select_axis(axis, b, k, c) - region.volume_origin[1]
                iz = select_axis(axis, c, b, k) - region.volume_origin[2]
                valid = (
                    (ix >= 0)
                    & (ix < shape[0])
                    & (iy >= 0)
                    & (iy < shape[1])
                    & (iz >= 0)
                    & (iz < shape[2])
                )
                value = volume[
                    jnp.clip(ix, 0, shape[0] - 1),
                    jnp.clip(iy, 0, shape[1] - 1),
                    jnp.clip(iz, 0, shape[2] - 1),
                ]
                return jnp.where(valid, value, 0.0)

            if region.interpolation == "linear":
                value = (sample(ib, ic) * (1 - wc) + sample(ib, ic + 1) * wc) * (1 - wb) + (
                    sample(ib + 1, ic) * (1 - wc) + sample(ib + 1, ic + 1) * wc
                ) * wb
            else:
                weights_b, _ = cubic_weights_and_derivatives(wb)
                weights_c, _ = cubic_weights_and_derivatives(wc)
                if region.absolute_weights:
                    weights_b = tuple(jnp.abs(w) for w in weights_b)
                    weights_c = tuple(jnp.abs(w) for w in weights_c)
                value = jnp.zeros_like(total)
                for row in range(4):
                    value_row = jnp.zeros_like(total)
                    for col in range(4):
                        value_row += sample(ib + row - 1, ic + col - 1) * weights_c[col]
                    value += value_row * weights_b[row]
            return total + value * cf[9]

        return jax.lax.fori_loop(0, max(shape), plane, jnp.zeros(region.image_shape, jnp.float32))

    return jax.vmap(view)(coefficients)


__all__ = ["PlaneRegion"]
