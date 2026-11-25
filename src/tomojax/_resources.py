"""Estimate logical working arrays before dispatch and sample the selected allocator."""

from __future__ import annotations

import math
from time import time
from typing import TYPE_CHECKING

from tomojax._typed_arrays import device_id
from tomojax.wire.resources import Placement, Snapshot

if TYPE_CHECKING:
    from tomojax._planning import ResolvedReconstruction
    from tomojax._scan import Scan
    from tomojax._smooth_alignment import CentreSelection, JointSelection


def sample(
    plan: CentreSelection | JointSelection | ResolvedReconstruction | None = None,
) -> tuple[Snapshot, ...]:
    """Sample only on the execution thread. CPU availability is host/cgroup room."""
    from tomojax._planning import ResolvedReconstruction
    from tomojax._smooth_alignment import CentreSelection
    from tomojax.backends import device_free_memory_bytes
    from tomojax.core.devices import current_device
    from tomojax.core.host_volume import host_available_memory_bytes

    clock = time()
    if isinstance(plan, CentreSelection):
        return (Snapshot(clock, "cpu:0", "host", None, None, host_available_memory_bytes()),)
    devices = None
    if isinstance(plan, ResolvedReconstruction):
        devices = getattr(plan.config, "devices", None)
        if plan.method == "fourier":
            from tomojax.recon.fourier import resolve_fourier_backend

            backend = resolve_fourier_backend(plan.config.backend)
            if backend == "numpy":
                return (
                    Snapshot(clock, "cpu:0", "host", None, None, host_available_memory_bytes()),
                )
            import cupy

            ordinal = cupy.cuda.runtime.getDevice()
            free, _ = cupy.cuda.runtime.memGetInfo()
            return (Snapshot(clock, f"gpu:{ordinal}", "cuda_driver", None, None, int(free)),)
    result = []
    for device in devices or (current_device(),):
        stats = getattr(device, "memory_stats", dict)() or {}
        result.append(
            Snapshot(
                clock,
                device_id(device),
                "host" if device.platform == "cpu" else "jax",
                stats.get("bytes_in_use"),
                stats.get("peak_bytes_in_use"),
                device_free_memory_bytes(device=device),
            )
        )
        if (
            isinstance(plan, ResolvedReconstruction)
            and device.platform == "gpu"
            and plan.method in {"cgls", "sirt", "mlem"}
            and plan.config.backend != "jax"
        ):
            from importlib.util import find_spec

            from tomojax.core.backend_policy import cuda_backend, cupy_available_bytes

            if cuda_backend(device.platform) and find_spec("cupy") is not None:
                import cupy

                with cupy.cuda.Device(device.local_hardware_id):
                    if cupy.cuda.is_available():
                        result.append(
                            Snapshot(
                                clock, device_id(device), "cupy", None, None, cupy_available_bytes()
                            )
                        )
    return tuple(result)


def estimate(
    scan: Scan,
    plan: CentreSelection | JointSelection | ResolvedReconstruction,
    snapshot: tuple[Snapshot, ...],
) -> Placement:
    """Project logical storage from the resolved work against a fixed sample.

    Count FP32 image and projection arrays. Batch and slab sizes limit projected
    transfers. Runtime compilation, allocator fragmentation and external allocations
    can change the eventual strategy and peak.
    """
    from tomojax._smooth_alignment import CentreSelection, JointSelection

    views, rows, cols = scan.projections.shape
    hosts = views * rows * cols * scan.projections.dtype.itemsize
    devices = tuple(item.device for item in snapshot)
    available = min((s.available for s in snapshot if s.available is not None), default=None)
    if isinstance(plan, CentreSelection):
        # Sample up to two interpolation neighbours at either end of every pair, in FP64.
        work = 8 * min(views, len(plan.samples) * 4) * min(rows, 64) * cols
        return Placement(devices, "resident", hosts + work, 0, snapshot)
    if isinstance(plan, JointSelection):
        # Levels execute in sequence and keep their host observations for the solve.
        bins = [level.binning for level in plan.schedule.levels]
        data = [4 * views * (rows // b) * (cols // b) for b in bins]
        hosts += 2 * sum(data)
        # Image state and the observation, precision, prediction and derivative stacks.
        state = max(
            4 * math.prod(level.grid.shape) * 6 + pixels * 8
            for level, pixels in zip(plan.schedule.levels, data, strict=True)
        )
        slabbed = available is not None and state > available
        return Placement(
            devices,
            "slabbed" if slabbed else "resident",
            hosts + (max(data) if slabbed else 0),
            state,
            snapshot,
        )

    from tomojax.recon.planning import inversion_storage

    return inversion_storage(
        plan.config,
        plan.scan.geometry,
        scan.projections,
        snapshot,
        warm_start=plan.warm_start,
    )
