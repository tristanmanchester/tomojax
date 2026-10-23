"""Eager trilinear volume resampling with zero-extended grid boundaries.

Large CUDA volumes use CuPy on XLA's stream, retaining double-precision affine
coordinates without changing JAX's precision setting. Small volumes and CPU
runs use SciPy. The result is a host float32 array on both paths.
"""

from __future__ import annotations

import os
from typing import Any

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np
from scipy import ndimage

from ._cuda_joseph import cuda_gather_available, xla_stream

# Below this size a one-off callback/kernel compilation costs more than the
# CPU interpolation. This is a dispatch threshold, not a change in resolution.
_MIN_CUDA_VOXELS = 2**22


def _launch(context: Any, out: Any, volume: Any, words: Any) -> None:
    import cupy as cp
    from cupyx.scipy.ndimage import affine_transform

    with xla_stream(context, out):
        matrix = cp.asarray(words).view(cp.float64).reshape(4, 4)
        affine_transform(
            cp.asarray(volume),
            matrix,
            output=cp.asarray(out),
            order=1,
            mode="grid-constant",
            float64_coords=True,
        )


@jax.jit
def _resample_cuda(volume: jax.Array, words: jax.Array) -> jax.Array:
    call = buffer_callback(_launch, jax.ShapeDtypeStruct(volume.shape, jnp.float32))
    return call(volume, words)


def affine_resample(volume: np.ndarray | jax.Array, matrix: np.ndarray) -> np.ndarray:
    """Resample a zyx volume at ``matrix @ [iz, iy, ix, 1]``.

    ``matrix`` is the homogeneous 4-by-4 output-index-to-input-index map.
    Trilinear interpolation blends zero-valued samples outside the input grid,
    including at its faces. Inputs and outputs have the same shape. This eager
    reporting operation is not a differentiable projector.

    ``TOMOJAX_CUDA_KERNELS=0`` disables CUDA; ``1`` also uses it for small inputs.
    """
    setting = os.environ.get("TOMOJAX_CUDA_KERNELS", "auto")
    large = setting == "1" or volume.size >= _MIN_CUDA_VOXELS
    if setting != "0" and large and cuda_gather_available():
        source = jnp.asarray(volume, jnp.float32)
        (device,) = source.devices()
        if device.platform == "gpu":
            # FP64 coefficients travel as raw words, so they survive
            # jax_enable_x64=False; CuPy reads them back on the device.
            words = np.ascontiguousarray(matrix, np.float64).view(np.uint32)
            encoded = jax.device_put(words.reshape(4, 4, 2), device)
            return np.asarray(_resample_cuda(source, encoded))
    return ndimage.affine_transform(
        np.asarray(volume, np.float32),
        np.asarray(matrix, np.float64),
        order=1,
        mode="grid-constant",
    )


__all__ = ["affine_resample"]
