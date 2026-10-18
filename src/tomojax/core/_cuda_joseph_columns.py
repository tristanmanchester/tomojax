"""Fused CUDA value and directional derivatives of the Joseph ray model."""

from __future__ import annotations

from functools import cache, partial
from pathlib import Path
from typing import TYPE_CHECKING

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np

from ._cuda_joseph import xla_stream
from ._plane_interpolation import validate_interpolation

if TYPE_CHECKING:
    from .geometry.base import Detector, Grid


@cache
def _kernel(directions: int, interpolation: str):
    import cupy as cp

    root = Path(__file__).parent
    source = (root / "joseph_sampling.cuh").read_text() + (
        root / "joseph_pose_columns.cu"
    ).read_text()
    source = source.replace("DIRECTIONS", str(directions)).replace(
        "CUBIC", str(int(interpolation == "cubic"))
    )
    return cp.RawModule(code=source, options=("--fmad=false",)).get_function("pose_columns")


def _launch(context, out, coeff, tangent, volume, *, grid, detector, interpolation):
    import cupy as cp

    views, directions = tangent.shape[:2]
    with xla_stream(context, out):
        _kernel(directions, interpolation)(
            (-(-detector.nu // 8) * -(-detector.nv // 16) * views,),
            (128,),
            (
                cp.asarray(coeff),
                cp.asarray(tangent),
                cp.asarray(volume),
                cp.asarray(out),
                np.int32(grid.nx),
                np.int32(grid.ny),
                np.int32(grid.nz),
                np.int32(detector.nu),
                np.int32(detector.nv),
                np.int32(views),
            ),
        )


def pose_columns_cuda(
    coefficients: jax.Array,
    volume: jax.Array,
    tangent: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    interpolation: str = "linear",
) -> jax.Array:
    """Return ``(views, 1 + directions, nv, nu)`` values and coefficient tangents.

    Tangents have shape ``(views, directions, 14)``. The discrete dominant
    axis is held fixed, and derivatives use the current interpolation cell.
    No ray-by-plane tape or repeated volume is materialised.
    """
    validate_interpolation(interpolation)
    call = buffer_callback(
        partial(_launch, grid=grid, detector=detector, interpolation=interpolation),
        jax.ShapeDtypeStruct(
            (coefficients.shape[0], 1 + tangent.shape[1], detector.nv, detector.nu), jnp.float32
        ),
        vmap_method="sequential",
    )
    return call(
        coefficients.astype(jnp.float32), tangent.astype(jnp.float32), volume.astype(jnp.float32)
    )
