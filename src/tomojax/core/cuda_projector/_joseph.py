"""Exact CUDA plane sampling and bounded-workspace matched transposes."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
from jax.custom_batching import sequential_vmap
import jax.numpy as jnp

from tomojax.core._cuda_axial import (
    axial_coefficients,
    forward_dispatch_cuda,
    transpose_axial_cuda,
    transpose_axial_vu_cuda,
)
from tomojax.core._cuda_joseph import gather_transpose_cuda, use_cuda_gather
from tomojax.core._plane_interpolation import validate_interpolation

from . import _joseph_generic as generic

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Grid


@partial(jax.jit, static_argnames=("grid", "detector", "interpolation", "absolute_weights"))
def forward_cuda(
    coeff: jax.Array,
    volume: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    interpolation: str = "linear",
    absolute_weights: bool = False,
) -> jax.Array:
    """Project voxel-centre planes, optionally using absolute interpolation weights."""
    validate_interpolation(interpolation)
    if interpolation == "linear" and use_cuda_gather(grid, axial=True):
        threads = 32 if detector.nv == 1 else 128
        tile_v = 1
        text = generic.source(
            "forward",
            INDEX_TYPE="int" if grid.nx * grid.ny * grid.nz < 2**31 else "long",
            TILE_V=tile_v,
            BLOCK=threads,
            CUBIC=0,
            ABSOLUTE=0,
        )

        @sequential_vmap
        def dispatch(c, x):
            return forward_dispatch_cuda(c, x, grid, detector, generic_source=text)

        return dispatch(coeff, volume)
    return generic.forward(
        coeff,
        volume,
        grid,
        detector,
        interpolation=interpolation,
        absolute_weights=absolute_weights,
    )


@partial(jax.jit, static_argnames=("grid", "detector", "interpolation", "absolute_weights"))
def adjoint_cuda(
    coeff: jax.Array,
    images: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    interpolation: str = "linear",
    absolute_weights: bool = False,
    accumulate: jax.Array | None = None,
) -> jax.Array:
    """Add the matched transpose to an initializer without scattered atomics."""
    validate_interpolation(interpolation)
    shape = (grid.nz, grid.ny, grid.nx)
    initial = jnp.zeros(shape, jnp.float32) if accumulate is None else accumulate

    def fallback(args):
        c, y, x = args
        if interpolation == "linear" and not absolute_weights and use_cuda_gather(grid):
            return gather_transpose_cuda(c, generic.swap_detector_axes(y), grid, detector, x)
        return generic.adjoint(
            c, y, x, grid, detector, interpolation=interpolation, absolute_weights=absolute_weights
        )

    if not absolute_weights and use_cuda_gather(grid, axial=True):

        @sequential_vmap
        def dispatch(c, y, x):
            def native(args):
                c, y, x = args
                if interpolation == "cubic":
                    return transpose_axial_cuda(
                        c, generic.swap_detector_axes(y), x, grid=grid, interpolation=interpolation
                    ).reshape(shape)
                return transpose_axial_vu_cuda(c, y, x, grid=grid, interpolation=interpolation)

            # Batch the complete decision, not only its callbacks: a batched
            # cond would execute both generic and axial projection kernels.
            return jax.lax.cond(axial_coefficients(c), native, fallback, (c, y, x))

        return dispatch(coeff, images, initial)
    if interpolation == "linear" and not absolute_weights and use_cuda_gather(grid):

        @sequential_vmap
        def dispatch(c, y, x):
            return fallback((c, y, x))

        return dispatch(coeff, images, initial)
    return fallback((coeff, images, initial))
