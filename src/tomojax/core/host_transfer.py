"""Synchronous copies between host storage and one explicitly selected device."""

from __future__ import annotations

from itertools import product
import math
from typing import TYPE_CHECKING, Literal

import jax
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tomojax._typed_arrays import Device


def _copy_windows(shape: tuple[int, ...], itemsize: int) -> Iterator[tuple[slice, ...]]:
    """Partition logical indices into at most 64 MiB contiguous host tiles."""
    if any(length == 0 for length in shape):
        return
    block = list(shape)
    for axis, length in enumerate(shape):
        tail = math.prod(block[axis + 1 :])
        block[axis] = min(length, max(1, (64 << 20) // (itemsize * tail)))
        if math.prod(block) * itemsize <= 64 << 20:
            break
    for start in product(*(range(0, n, b) for n, b in zip(shape, block, strict=True))):
        yield tuple(slice(s, min(n, s + b)) for s, n, b in zip(start, shape, block, strict=True))


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


def copy_to_host(
    array: jax.Array,
    target: np.ndarray,
    *,
    casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "no",
) -> None:
    """Copy into owned storage without host staging for contiguous or row windows.

    The copy is complete on return. Neither the source device array nor storage
    outside the destination view is modified. Other layouts and dtype conversions
    use at most 64 MiB of host and device scratch. ``casting`` follows NumPy's
    copy rules; by default the source and destination dtypes must match.
    """
    device = next(iter(array.devices()))
    if device.platform != "gpu":
        np.copyto(target, np.asarray(array), casting=casting)
        return
    if not np.can_cast(array.dtype, target.dtype, casting=casting):
        raise TypeError(f"Cannot cast {array.dtype} to {target.dtype} with casting={casting!r}")
    import cupy as cp

    with cp.cuda.Device(device.local_hardware_id):
        source = cp.broadcast_to(cp.from_dlpack(array), target.shape)
        stream = cp.cuda.get_current_stream()
        same_dtype = target.dtype == source.dtype
        matching = (source.flags.c_contiguous and target.flags.c_contiguous) or (
            source.flags.f_contiguous and target.flags.f_contiguous
        )
        layout = _pitched_rows(target)
        if same_dtype and matching:
            cp.cuda.runtime.memcpyAsync(
                target.ctypes.data, source.data.ptr, target.nbytes,
                cp.cuda.runtime.memcpyDeviceToHost, stream.ptr,
            )  # fmt: skip
        elif same_dtype and source.flags.c_contiguous and layout is not None:
            width, pitch, rows = layout
            cp.cuda.runtime.memcpy2DAsync(
                target.ctypes.data, pitch, source.data.ptr, width, width, rows,
                cp.cuda.runtime.memcpyDeviceToHost, stream.ptr,
            )  # fmt: skip
        else:
            # Any packing done by get() belongs to this copy, not CuPy's shared pool.
            with cp.cuda.using_allocator(None):
                for window in _copy_windows(target.shape, source.dtype.itemsize):
                    part = source[window]
                    temporary = np.empty(part.shape, source.dtype)
                    part.get(out=temporary, blocking=True)
                    np.copyto(target[window] if window else target, temporary, casting=casting)
                    del temporary
            return
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
