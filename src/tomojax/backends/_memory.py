from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import TYPE_CHECKING, cast

from tomojax.core.host_volume import host_available_memory_bytes

if TYPE_CHECKING:
    from tomojax._typed_arrays import Device

LOG = logging.getLogger(__name__)


def is_oom_error(exc: Exception) -> bool:
    """Whether ``exc`` is a device running out of memory, which smaller batches may avoid."""
    msg = str(exc).lower()
    # cuFFT reports exhausted scratch memory as an INTERNAL plan-creation error.
    cufft_scratch = "fft_plan != nullptr" in msg and "cufft batched plan with scratch" in msg
    return "resource_exhausted" in msg or "out of memory" in msg or cufft_scratch


def _free_bytes_from_memory_info(info: object) -> int | None:
    """Extract free bytes from JAX/XLA memory-info shapes."""
    if isinstance(info, Mapping):
        mapping = cast("Mapping[str, object]", info)
        for key in ("free", "free_bytes", "bytes_free", "available", "bytes_available"):
            if key in mapping:
                return _byte_count(mapping[key])
        if "bytes_limit" in mapping and "bytes_in_use" in mapping:
            limit, used = _byte_count(mapping["bytes_limit"]), _byte_count(mapping["bytes_in_use"])
            return None if limit is None or used is None else max(0, limit - used)
        return None
    if isinstance(info, tuple | list) and len(info) >= 1:
        return _byte_count(cast("object", info[0]))
    return None


def _byte_count(value: object) -> int | None:
    try:
        return int(cast("int", value))
    except (TypeError, ValueError):
        return None


def _cuda_free_memory_bytes(device: Device) -> int | None:
    """Query the requested CUDA device when its allocator has no statistics."""
    try:
        import cupy

        # JAX ids can be global across processes; CUDA ordinals are local.
        ordinal = int(device.local_hardware_id)
        with cupy.cuda.Device(ordinal):
            free, _total = cupy.cuda.runtime.memGetInfo()
        return int(free)
    except Exception as exc:
        LOG.debug("CUDA device memory probe unavailable: %s", exc)
        return None


def device_free_memory_bytes(*, device: Device | None = None) -> int | None:
    """Best-effort query of free memory (bytes) on ``device``: an upper bound on what fits.

    Under XLA's pool this is the pool's limit less the bytes in use, the total room. It
    cannot see fragmentation: an array needs one contiguous block, and once use is spread
    over the pool's regions the largest block can be far smaller (K11 67208's alignment
    failed a 1.29 GiB allocation with 2.95 GiB "free" and a 1.0 GiB largest block). The pool
    never returns a free region to the driver: an array fits only in its largest free block
    or in what it may still grow by under its limit (``memory_stats``).

    - ``device`` defaults to ``jax.default_device``, else the first GPU.
    - Prefer JAX's device API when available: `jax.device_get_memory_info` (>=0.4.14).
    - Also support current `Device.memory_stats()` reports when exposed.
    - Query CUDA directly when a GPU allocator does not expose statistics.
    - Use host available memory only for a CPU device or no GPU.
    - Return None for an accelerator whose memory cannot be queried.
    """
    try:  # JAX GPU/TPU or CPU
        import jax  # type: ignore

        if device is None:
            device = jax.config.jax_default_device
            if device is None or isinstance(device, str):
                device = next(iter(jax.devices(device or "gpu")), None)
        if device is None or device.platform == "cpu":
            return host_available_memory_bytes()
        memory_info = getattr(jax, "device_get_memory_info", None)
        if callable(memory_info):
            free = _free_bytes_from_memory_info(memory_info(device))
            if free is not None:
                return free
        memory_stats = getattr(device, "memory_stats", None)
        if callable(memory_stats):
            free = _free_bytes_from_memory_info(memory_stats())
            if free is not None:
                return free
    except Exception as exc:
        LOG.debug("Device memory probe failed: %s", exc)
    # Host RAM is not a capacity estimate for a known accelerator.
    if device is not None and not isinstance(device, str) and device.platform != "cpu":
        return _cuda_free_memory_bytes(device) if device.platform == "gpu" else None
    return host_available_memory_bytes()
