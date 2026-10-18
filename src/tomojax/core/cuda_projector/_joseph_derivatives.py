"""First-order Joseph derivatives with constant per-ray CUDA workspace."""

from __future__ import annotations

from functools import cache
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np

from tomojax.core.cuda import broadcast_offset, launch_kernel

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Grid


@cache
def _source(
    shape: tuple[int, int, int],
    detector: tuple[int, int],
    mode: int,
    parameters: int,
    interpolation: str,
    batch_shape: tuple[int, ...],
    input_shapes: tuple[tuple[int, ...], ...],
    *,
    has_coeff: bool,
    has_volume: bool,
) -> str:
    width = 1 << (parameters + parameters * (parameters + 1) // 2).bit_length() if mode == 3 else 16
    constants: dict[str, int | str] = dict(
        zip(("NX", "NY", "NZ", "NV", "NU"), (*shape, *detector), strict=True)
    )
    constants.update(
        MODE=mode,
        PARAMETERS=parameters,
        WIDTH=width,
        CUBIC=int(interpolation == "cubic"),
        HAS_COEFF=int(has_coeff),
        HAS_VOLUME=int(has_volume),
    )
    ranks = (2, 3, 2 if mode == 2 else 3, 3 if mode == 3 else 2, 3)
    for name, full, rank in zip(
        ("COEFF", "VOLUME", "DATA", "DIRECTIONS", "DVOLUME"), input_shapes, ranks, strict=True
    ):
        constants[name + "_OFFSET"] = broadcast_offset(
            batch_shape, full[:-rank], item_size=math.prod(full[-rank:])
        )
    source = Path(__file__).with_name("joseph_derivatives.cu").read_text()
    header = Path(__file__).parents[1].joinpath("joseph_sampling.cuh").read_text()
    return (
        "\n".join(f"#define {key} {value}" for key, value in constants.items())
        + "\n"
        + source.replace('#include "joseph_sampling.cuh"', header)
    )


def _run(
    coeff: jax.Array,
    volume: jax.Array,
    data: jax.Array,
    directions: jax.Array,
    dvolume: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    mode: int,
    interpolation: str,
    has_coeff: bool = True,
    has_volume: bool = False,
) -> tuple[jax.Array, jax.Array]:
    shape = (grid.nx, grid.ny, grid.nz)
    det_shape = (detector.nv, detector.nu)
    parameters = directions.shape[-1] if mode == 3 else 0
    width = 1 << (parameters + parameters * (parameters + 1) // 2).bit_length() if mode == 3 else 16
    views = coeff.shape[0]
    tiles = (detector.nu * detector.nv + 127) // 128

    def launch(context: Any, outputs: Any, cf: Any, x: Any, y: Any, ds: Any, dx: Any) -> None:
        # Unused pointer arguments alias existing inputs/outputs, not allocated
        # zero arrays. Compile-time mode flags ensure they are never read.
        partials, residual = outputs if mode in (1, 3) else (outputs, outputs)
        batch_shape = tuple(partials.shape[:-3])
        batches = math.prod(batch_shape)
        if not batches:
            return
        input_shapes = tuple(tuple(a.shape) for a in (cf, x, y, ds, dx))
        launch_kernel(
            _source(
                shape,
                det_shape,
                mode,
                parameters,
                interpolation,
                batch_shape,
                input_shapes,
                has_coeff=has_coeff,
                has_volume=has_volume,
            ),
            "derivatives",
            options=("--fmad=false",),
            stream=context.stream,
            grid=(batches * views * tiles,),
            block=(128,),
            arguments=(cf, x, y, ds, dx, partials, residual, np.int32(views), np.int32(tiles)),
        )

    partial_shape = jax.ShapeDtypeStruct((views, tiles, width), jnp.float32)
    residual_shape = jax.ShapeDtypeStruct((views, *det_shape), jnp.float32)
    result_shape = (
        (partial_shape, residual_shape)
        if mode in (1, 3)
        else partial_shape
        if mode == 0
        else residual_shape
    )
    result = buffer_callback(
        launch, result_shape, vmap_method="expand_dims", command_buffer_compatible=True
    )(coeff, volume, data, directions, dvolume)
    return result if mode in (1, 3) else (result, result)


def coefficient_vjp(
    coeff: jax.Array,
    volume: jax.Array,
    cotangent: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    interpolation: str = "linear",
) -> jax.Array:
    """Pull a projection cotangent back to the 14 coefficients per view."""
    partials, _ = _run(
        coeff, volume, cotangent, coeff, volume, grid, detector, mode=0, interpolation=interpolation
    )
    return jnp.sum(partials, axis=1)[:, :14]


def projection_jvp(
    coeff: jax.Array,
    volume: jax.Array,
    dcoeff: jax.Array | None,
    dvolume: jax.Array | None,
    grid: Grid,
    detector: Detector,
    *,
    interpolation: str = "linear",
) -> jax.Array:
    """Project coefficient and/or volume tangents without a sample tape."""
    _, output = _run(
        coeff,
        volume,
        coeff if dcoeff is None else dcoeff,
        coeff,
        volume if dvolume is None else dvolume,
        grid,
        detector,
        mode=2,
        interpolation=interpolation,
        has_coeff=dcoeff is not None,
        has_volume=dvolume is not None,
    )
    return output


def l2_loss_and_grads(
    coeff: jax.Array,
    volume: jax.Array,
    target: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    interpolation: str = "linear",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return half squared error, coefficient gradient and matched volume gradient."""
    from tomojax.core.cuda_projector._joseph import adjoint_cuda

    partials, residual = _run(
        coeff, volume, target, coeff, volume, grid, detector, mode=1, interpolation=interpolation
    )
    sums = jnp.sum(partials, axis=1)
    return (
        sums[:, 15].sum(),
        sums[:, :14],
        adjoint_cuda(coeff, residual, grid, detector, interpolation=interpolation),
    )


def pose_normal_equations(
    coeff: jax.Array,
    volume: jax.Array,
    target: jax.Array,
    directions: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    interpolation: str,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return per-view loss, directional gradient, Gauss-Newton matrix and residual."""
    partials, residual = _run(
        coeff,
        volume,
        target,
        directions,
        volume,
        grid,
        detector,
        mode=3,
        interpolation=interpolation,
    )
    sums = jnp.sum(partials, axis=1)
    views, parameters = coeff.shape[0], directions.shape[-1]
    triangle = parameters * (parameters + 1) // 2
    rows, cols = jnp.triu_indices(parameters)
    upper = sums[:, 1 + parameters : 1 + parameters + triangle]
    matrix = jnp.zeros((views, parameters, parameters), jnp.float32)
    matrix = matrix.at[:, rows, cols].set(upper).at[:, cols, rows].set(upper)
    return sums[:, 0], sums[:, 1 : 1 + parameters], matrix, residual
