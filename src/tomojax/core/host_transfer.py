"""Synchronous copies between host storage and one explicitly selected device."""

from __future__ import annotations

from functools import partial
from itertools import product
import math
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
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


@partial(jax.jit, static_argnames=("shape",))
def _allocate(value: jax.Array, *, shape: tuple[int, ...]) -> jax.Array:
    # A dynamic parameter makes this a fresh output, not a reusable constant.
    return jnp.broadcast_to(value, shape)


def allocate_device(
    shape: tuple[int, ...], *, dtype: np.dtype[np.generic], device: Device
) -> jax.Array:
    """Allocate private storage in JAX's pool, ready for a synchronous host copy."""
    value = jax.device_put(np.zeros((), dtype=dtype), device=device)
    result = _allocate(value, shape=shape)
    result.block_until_ready()
    return result


def upload_host(array: np.ndarray, device: Device) -> jax.Array:
    """Upload without array-sized pinned staging; never donate the caller's input.

    Regular row windows copy directly; other strided inputs are packed once.
    CPU uploads own a private host copy. CUDA storage belongs to JAX's allocator;
    copies complete on the current CuPy stream before returning. On either
    platform the caller may immediately reuse host storage.
    """
    if device.platform != "gpu":
        return jax.device_put(np.array(array, copy=True), device)
    import cupy as cp

    dtype = jax.dtypes.canonicalize_dtype(array.dtype)
    layout = _pitched_rows(array)
    source = (
        np.ascontiguousarray(array, dtype=dtype)
        if layout is None or array.dtype != dtype
        else array
    )
    with cp.cuda.Device(device.local_hardware_id):
        result = allocate_device(source.shape, dtype=source.dtype, device=device)
        target = cp.from_dlpack(result)
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
        return result


def upload_host_slabs(
    shape: tuple[int, int, int], slabs: Iterator[tuple[int, np.ndarray]], device: Device
) -> jax.Array:
    """Assemble a private FP32 device volume from disjoint C-order z slabs.

    Each pair gives the starting plane and its values. The slabs cover ``shape``
    in order. Only the current host slab is retained, and its copy completes
    before requesting the next one.
    """
    if device.platform != "gpu":
        output = np.empty(shape, np.float32)
        for first, values in slabs:
            output[:, :, first : first + values.shape[2]] = values
            del values
        return jax.device_put(output, device)
    import cupy as cp

    with cp.cuda.Device(device.local_hardware_id):
        result = allocate_device(shape, dtype=np.dtype(np.float32), device=device)
        target = cp.from_dlpack(result)
        stream = cp.cuda.get_current_stream()
        for first, values in slabs:
            width = values.shape[2] * 4
            cp.cuda.runtime.memcpy2DAsync(
                target.data.ptr + first * 4, shape[2] * 4,
                values.ctypes.data, width, width, shape[0] * shape[1],
                cp.cuda.runtime.memcpyHostToDevice, stream.ptr,
            )  # fmt: skip
            stream.synchronize()
            del values
    return result


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
