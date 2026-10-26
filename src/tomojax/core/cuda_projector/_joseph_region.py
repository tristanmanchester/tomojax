"""Joseph products on bounded arrays with unchanged global sample coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np

from tomojax.core.cuda import launch_kernel

from ._joseph_generic import source

if TYPE_CHECKING:
    from tomojax.core.joseph_regions import PlaneRegion


def columns(
    coefficients: jax.Array,
    volume: jax.Array,
    tangent: jax.Array,
    *,
    region: PlaneRegion,
    accumulate: jax.Array | None,
) -> jax.Array:
    views, directions = tangent.shape[:2]
    nv, nu = region.image_shape
    v0, u0 = region.detector_origin
    grid = region.grid
    text = source(
        "region",
        CUBIC=int(region.interpolation == "cubic"),
        ABSOLUTE=int(region.absolute_weights),
        DIRECTIONS=directions,
        RUN=1,
        ACCUMULATE=int(accumulate is not None),
    )

    def launch(context: Any, out: Any, cf: Any, x: Any, t: Any, initial: Any) -> None:
        launch_kernel(
            text,
            "region_columns",
            options=("--fmad=false",),
            stream=context.stream,
            grid=((-(-nu // 8)) * (-(-nv // 16)) * views,),
            block=(128,),
            arguments=(
                cf,
                t,
                x,
                initial,
                out,
                *map(
                    np.int32,
                    (
                        grid.nx,
                        grid.ny,
                        grid.nz,
                        *volume.shape[::-1],
                        *region.volume_origin,
                        nu,
                        nv,
                        u0,
                        v0,
                        views,
                    ),
                ),
            ),
        )

    return buffer_callback(
        launch,
        jax.ShapeDtypeStruct((views, directions + 1, nv, nu), jnp.float32),
        input_output_aliases={3: 0} if accumulate is not None else {},
        command_buffer_compatible=True,
        vmap_method="sequential",
    )(coefficients, volume, tangent, volume if accumulate is None else accumulate)


def adjoint(
    coefficients: jax.Array,
    images: jax.Array,
    initial: jax.Array,
    *,
    region: PlaneRegion,
) -> jax.Array:
    views, nv, nu = images.shape
    v0, u0 = region.detector_origin
    shape = initial.shape
    run = min(shape[0], 4)
    runs = shape[2] * shape[1] * (-(-shape[0] // run))
    text = source(
        "region",
        CUBIC=int(region.interpolation == "cubic"),
        ABSOLUTE=int(region.absolute_weights),
        DIRECTIONS=0,
        RUN=run,
        ACCUMULATE=0,
    )

    def launch(context: Any, out: Any, cf: Any, y: Any, x: Any) -> None:
        launch_kernel(
            text,
            "region_adjoint",
            options=("--fmad=false",),
            stream=context.stream,
            grid=((-(-runs // 128)),),
            block=(128,),
            arguments=(
                cf,
                y,
                x,
                out,
                *map(
                    np.int32,
                    (
                        views,
                        *shape[::-1],
                        *region.volume_origin,
                        region.detector.nu,
                        region.detector.nv,
                        nu,
                        nv,
                        u0,
                        v0,
                    ),
                ),
            ),
        )

    return buffer_callback(
        launch,
        jax.ShapeDtypeStruct(shape, jnp.float32),
        input_output_aliases={2: 0},
        command_buffer_compatible=True,
        vmap_method="sequential",
    )(coefficients, images, initial)
