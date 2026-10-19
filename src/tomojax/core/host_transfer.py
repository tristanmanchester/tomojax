"""Synchronous copies between host storage and one explicitly selected device."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np

if TYPE_CHECKING:
    from tomojax._typed_arrays import Device


def _pitched_rows(array: np.ndarray) -> tuple[int, int, int] | None:
    """Byte width, pitch and count when storage is a regular sequence of rows."""
    width, pitch, rows = array.dtype.itemsize, 0, 1
    for length, stride in zip(reversed(array.shape), reversed(array.strides), strict=True):
        if length == 1:
            continue
        if not pitch:
            if stride == width:
                width *= length
                continue
            if stride < width:
                return None
            pitch, rows = stride, length
        else:
            if stride != rows * pitch:
                return None
            rows *= length
    return width, pitch or width, rows


def upload_host(array: np.ndarray, device: Device) -> jax.Array:
    """Upload without array-sized pinned staging; never donate the caller's input.

    Regular row windows copy directly; other strided inputs are packed once.
    CUDA copies complete on the current CuPy stream before returning, so the
    caller may immediately reuse host storage.
    """
    if device.platform != "gpu":
        return jax.device_put(array, device)
    import cupy as cp

    layout = _pitched_rows(array)
    source = np.ascontiguousarray(array) if layout is None else array
    with cp.cuda.Device(device.local_hardware_id):
        target = cp.empty(source.shape, source.dtype)
        stream = cp.cuda.get_current_stream()
        if source.flags.c_contiguous:
            cp.cuda.runtime.memcpyAsync(
                target.data.ptr, source.ctypes.data, source.nbytes,
                cp.cuda.runtime.memcpyHostToDevice, stream.ptr,
            )  # fmt: skip
        else:
            assert layout is not None
            width, pitch, rows = layout
            cp.cuda.runtime.memcpy2DAsync(
                target.data.ptr, width, source.ctypes.data, pitch, width, rows,
                cp.cuda.runtime.memcpyHostToDevice, stream.ptr,
            )  # fmt: skip
        stream.synchronize()
        return jax.dlpack.from_dlpack(target)


def copy_to_host(array: jax.Array, target: np.ndarray) -> None:
    """Copy into owned storage without host staging for contiguous or row windows.

    The copy is complete on return. Neither the source device array nor storage
    outside the destination view is modified; other strided views use NumPy.
    """
    device = next(iter(array.devices()))
    layout = _pitched_rows(target)
    contiguous = target.flags.c_contiguous or target.flags.f_contiguous
    if (
        device.platform != "gpu"
        or (layout is None and not contiguous)
        or target.shape != array.shape
        or target.dtype != array.dtype
    ):
        np.copyto(target, np.asarray(array), casting="no")
        return
    import cupy as cp

    with cp.cuda.Device(device.local_hardware_id):
        source = cp.from_dlpack(array)
        if contiguous:
            source.get(out=target, blocking=True)
        else:
            assert layout is not None
            width, pitch, rows = layout
            source = cp.ascontiguousarray(source)
            stream = cp.cuda.get_current_stream()
            cp.cuda.runtime.memcpy2DAsync(
                target.ctypes.data, pitch, source.data.ptr, width, width, rows,
                cp.cuda.runtime.memcpyDeviceToHost, stream.ptr,
            )  # fmt: skip
            stream.synchronize()


def upload_host_views(array: np.ndarray, indices: np.ndarray, device: Device) -> jax.Array:
    """Upload selected FP32 views without an advanced-indexed host batch.

    Regular per-view rows copy directly from the caller's storage. Other
    layouts and dtype conversions retain the bounded packed-batch path.
    """
    layout = _pitched_rows(array[0])
    if device.platform != "gpu" or array.dtype != np.float32 or layout is None:
        return upload_host(np.asarray(array[indices], np.float32), device)
    import cupy as cp

    with cp.cuda.Device(device.local_hardware_id):
        target = cp.empty((len(indices), *array.shape[1:]), np.float32)
        stream = cp.cuda.get_current_stream()
        width, pitch, rows = layout
        for position, index in enumerate(indices):
            source = array[int(index)]
            cp.cuda.runtime.memcpy2DAsync(
                target.data.ptr + position * source.nbytes, width,
                source.ctypes.data, pitch, width, rows,
                cp.cuda.runtime.memcpyHostToDevice, stream.ptr,
            )  # fmt: skip
        stream.synchronize()
        return jax.dlpack.from_dlpack(target)
