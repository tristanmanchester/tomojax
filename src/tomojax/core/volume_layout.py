"""Bounded CUDA volume permutation at resident-solver boundaries."""

from __future__ import annotations

from typing import Any

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np


def _launch(context: Any, out: Any, volume: Any) -> None:
    import cupy as cp

    from tomojax.core._cone_runtime import cuda_module
    from tomojax.core._cuda_joseph import xla_stream

    nz, ny, nx = volume.shape
    with xla_stream(context, out):
        cuda_module(z_fast=True).get_function("transpose_xz")(
            (-(-nx // 32), -(-nz // 32), ny),
            (32, 8),
            (cp.asarray(volume), cp.asarray(out), np.int32(nx), np.int32(ny), np.int32(nz)),
        )


@jax.custom_jvp
def transpose_volume(volume: jax.Array) -> jax.Array:
    """Reverse the volume axes using one output buffer and a bounded shared tile.

    This explicit permutation avoids compiler autotuning with volume-sized
    trial buffers at the boundary of a memory-limited solve.
    """
    volume = jnp.asarray(volume, jnp.float32)
    call = buffer_callback(
        _launch,
        jax.ShapeDtypeStruct(volume.shape[::-1], jnp.float32),
        vmap_method="sequential",
    )
    return call(volume)


@transpose_volume.defjvp
def _transpose_jvp(
    primals: tuple[jax.Array], tangents: tuple[jax.Array]
) -> tuple[jax.Array, jax.Array]:
    # Keep the linear tangent in JAX, including its exact reverse-mode rule.
    return transpose_volume(primals[0]), jnp.transpose(
        jnp.asarray(tangents[0], jnp.float32), (2, 1, 0)
    )
