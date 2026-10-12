"""Download corrected batches into their final host storage, without a second copy."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Protocol

    import jax

    class _DeviceProducer(Protocol):
        def __dlpack_device__(self) -> tuple[int, int]: ...

        def delete(self) -> None: ...


# For small batches the optional import and DLPack hand-off cost more than
# copying JAX's cached host buffer. Large batches avoid that temporary buffer.
_DIRECT_BYTES = 8 * 2**20


def cuda_device_id(source: jax.Array) -> int | None:
    devices = source.devices()
    if len(devices) != 1:
        return None
    (device,) = devices
    if device.platform != "gpu":
        return None
    kind, ordinal = cast("_DeviceProducer", source).__dlpack_device__()
    return ordinal if kind == 2 else None


def stage_download(source: jax.Array, destination: np.ndarray) -> Callable[[], None]:
    """Prepare a batch download; the returned function completes its host writes."""
    direct = (
        source.nbytes >= _DIRECT_BYTES
        and destination.flags.c_contiguous
        and destination.shape == source.shape
        and destination.dtype == source.dtype
    )
    if direct and (device := cuda_device_id(source)) is not None:
        try:
            import cupy
        except ImportError:
            pass
        else:
            # Use the producer's device, not whichever CUDA device the caller
            # happens to have current. DLPack orders the producer and consumer
            # streams; get is blocking, so destination is ready on return.
            def collect_direct() -> None:
                with cupy.cuda.Device(device):
                    cupy.from_dlpack(source).get(out=destination)

            return collect_direct
    source.copy_to_host_async()

    def collect_jax() -> None:
        destination[...] = np.asarray(source)

    return collect_jax


def release_device(source: jax.Array) -> None:
    """Release an owned CUDA buffer after all its downloads and device uses finish.

    CPU buffers can still back NumPy statistics, so their normal shared
    ownership is retained. CUDA downloads have independent host storage.
    """
    if cuda_device_id(source) is not None:
        cast("_DeviceProducer", source).delete()
