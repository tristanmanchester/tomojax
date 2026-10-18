"""Synchronous copies between host storage and one explicitly selected device."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np

if TYPE_CHECKING:
    from tomojax._typed_arrays import Device


def upload_host(array: np.ndarray, device: Device) -> jax.Array:
    """Upload without array-sized pinned staging; never donate the caller's input.

    Strided inputs are packed once. CUDA copies complete on the current CuPy
    stream before returning, so the caller may immediately reuse host storage.
    """
    if device.platform != "gpu":
        return jax.device_put(array, device)
    import cupy as cp

    source = np.ascontiguousarray(array)
    with cp.cuda.Device(device.local_hardware_id):
        target = cp.empty(source.shape, source.dtype)
        stream = cp.cuda.get_current_stream()
        cp.cuda.runtime.memcpyAsync(
            target.data.ptr,
            source.ctypes.data,
            source.nbytes,
            cp.cuda.runtime.memcpyHostToDevice,
            stream.ptr,
        )
        stream.synchronize()
        return jax.dlpack.from_dlpack(target)


def copy_to_host(array: jax.Array, target: np.ndarray) -> None:
    """Copy into owned storage, avoiding temporary host arrays when contiguous.

    The copy is complete on return. Neither the source device array nor storage
    outside the destination view is modified; strided destinations use NumPy.
    """
    device = next(iter(array.devices()))
    if device.platform != "gpu" or not (target.flags.c_contiguous or target.flags.f_contiguous):
        np.copyto(target, np.asarray(array), casting="no")
        return
    import cupy as cp

    with cp.cuda.Device(device.local_hardware_id):
        cp.from_dlpack(array).get(out=target, blocking=True)
