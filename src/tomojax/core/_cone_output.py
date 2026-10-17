"""Bounded contiguous output storage for the exact cone transpose.

The callback owns this scratch for all view chunks of one physical z slab.
The caller selects the device and stream; no complete extra volume is retained.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Grid


def slab_capacity(
    grid: Grid,
    detector: Detector,
    nviews: int,
    hint: tuple[tuple[float, float, float], float] | None,
    l2_bytes: int,
    free_bytes: int | None,
    *,
    rhs: int = 1,
) -> int:
    """Bound axis-2 output packing when its strided working set exceeds L2.

    The host hint contains voxel spacing and the fraction of axis-2 central
    rays. It selects storage only; every ray keeps its exact axis and weights.
    Unknown metadata or capacity leaves the original transpose unchanged.
    """
    if hint is None or l2_bytes <= 0 or free_bytes is None or grid.nz <= 1:
        return 0
    spacing, fraction = hint
    if spacing != (grid.vx, grid.vy, grid.vz) or fraction <= 0:
        return 0
    budget = 64 << 20
    plane_bytes = 4 * rhs * grid.nx * grid.ny
    if plane_bytes > budget:
        return 0
    # A strided plane touches up to one cache line per voxel. Its footprint
    # competes with the input images; contiguous scratch reduces that pressure.
    # One channel alone must exceed L2; additional RHSs reduce slab capacity
    # without widening this conservative admission.
    scatter_bytes = grid.nx * grid.ny * min(4 * grid.nz, 128)
    image_bytes = 4 * min(nviews, 32) * fraction * detector.nu * detector.nv
    if scatter_bytes + image_bytes <= l2_bytes:
        return 0
    capacity = min(grid.nz, budget // plane_bytes)
    return capacity if capacity * plane_bytes <= free_bytes // 8 else 0


def device_capacity(
    grid: Grid,
    detector: Detector,
    nviews: int,
    hint: tuple[tuple[float, float, float], float] | None,
    *,
    rhs: int,
) -> int:
    """Plan scratch on the caller's current device, after its other allocations."""
    if hint is None:
        return 0
    import cupy as cp

    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    free, _ = cp.cuda.runtime.memGetInfo()
    return slab_capacity(grid, detector, nviews, hint, properties["l2CacheSize"], free, rhs=rhs)


def accumulate(
    module: Any,
    coeff: Any,
    *,
    boxes: Any,
    images: tuple[Any, ...],
    targets: tuple[Any, ...],
    grid: Grid,
    detector: Detector,
    capacity: int,
    view_chunk: int,
) -> None:
    """Add all views through bounded z slabs, for one or two independent RHSs."""
    import cupy as cp

    views, rhs = int(coeff.shape[0]), len(images)
    packed = cp.empty((rhs, grid.nx, capacity, grid.ny), cp.float32)
    channels = tuple(packed[channel] for channel in range(rhs))
    kernel = module.get_function(
        "pair_plane_adjoint_packed" if rhs == 2 else "plane_adjoint_packed"
    )
    dims = tuple(np.int32(x) for x in (grid.nx, grid.ny, grid.nz, detector.nu, detector.nv))
    for z_start in range(0, grid.nz, capacity):
        height = min(capacity, grid.nz - z_start)
        packed.fill(0)
        for start in range(0, views, view_chunk):
            count = min(view_chunk, views - start)
            kernel(
                (-(-grid.ny // 32), -(-grid.nx // 32), height),
                (128,),
                (
                    coeff[start : start + count], boxes[start : start + count],
                    np.int32(count), np.int32(2),
                    *(image[start : start + count] for image in images), *channels, *dims,
                    np.int32(z_start), np.int32(capacity),
                ),
            )  # fmt: skip
        for channel, target in zip(channels, targets, strict=True):
            output = target[:, :, z_start : z_start + height]
            cp.add(output, channel[:, :height, :].transpose(0, 2, 1), out=output)
