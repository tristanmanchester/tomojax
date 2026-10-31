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


_STATIC = ("grid", "detector", "interpolation", "absolute_weights", "z_fast")


@partial(jax.jit, static_argnames=_STATIC)
def forward_cuda(
    coeff: jax.Array,
    volume: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    interpolation: str = "linear",
    absolute_weights: bool = False,
    z_fast: bool = False,
) -> jax.Array:
    """Project voxel-centre planes, optionally using absolute interpolation weights.

    The volume is (nz, ny, nx), or solver-private (nx, ny, nz) with ``z_fast``.
    """
    validate_interpolation(interpolation)
    if interpolation == "linear" and use_cuda_gather(grid, axial=True):
        threads = 32 if detector.nv == 1 else 128
        tile_v = generic.storage_tile(detector, z_fast=z_fast)
        text = generic.source(
            "forward",
            INDEX_TYPE="int" if grid.nx * grid.ny * grid.nz < 2**31 else "long",
            TILE_V=tile_v,
            BLOCK=threads,
            CUBIC=0,
            ABSOLUTE=0,
            Z_FAST=int(z_fast),
        )

        @sequential_vmap
        def dispatch(c, x):
            return forward_dispatch_cuda(
                c, x, grid, detector, generic_source=text, tile_v=tile_v, z_fast=z_fast
            )

        return dispatch(coeff, volume)
    return generic.forward(
        coeff,
        volume,
        grid,
        detector,
        interpolation=interpolation,
        absolute_weights=absolute_weights,
        z_fast=z_fast,
    )


@partial(jax.jit, static_argnames=_STATIC)
def adjoint_cuda(
    coeff: jax.Array,
    images: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    interpolation: str = "linear",
    absolute_weights: bool = False,
    accumulate: jax.Array | None = None,
    z_fast: bool = False,
) -> jax.Array:
    """Add the matched transpose to an initializer without scattered atomics.

    The volume is (nz, ny, nx), or solver-private (nx, ny, nz) with ``z_fast``.
    """
    validate_interpolation(interpolation)
    shape = (grid.nx, grid.ny, grid.nz) if z_fast else (grid.nz, grid.ny, grid.nx)
    initial = jnp.zeros(shape, jnp.float32) if accumulate is None else accumulate

    def fallback(args):
        c, y, x = args
        if interpolation == "linear" and not absolute_weights and use_cuda_gather(grid):
            y = generic.swap_detector_axes(y)
            return gather_transpose_cuda(c, y, grid, detector, x, z_fast=z_fast)
        return generic.adjoint(
            c,
            y,
            x,
            grid,
            detector,
            interpolation=interpolation,
            absolute_weights=absolute_weights,
            z_fast=z_fast,
        )

    if not absolute_weights and use_cuda_gather(grid, axial=True):

        @sequential_vmap
        def dispatch(c, y, x):
            def native(args):
                c, y, x = args
                if interpolation == "cubic":
                    y = generic.swap_detector_axes(y)
                    return transpose_axial_cuda(
                        c, y, x, grid=grid, interpolation=interpolation, z_fast=z_fast
                    ).reshape(shape)
                return transpose_axial_vu_cuda(
                    c, y, x, grid=grid, interpolation=interpolation, z_fast=z_fast
                )

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
