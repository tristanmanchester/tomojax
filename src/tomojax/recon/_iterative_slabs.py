"""Bounded raw-row consumption for independent iterative reconstruction planes."""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import replace
from typing import TYPE_CHECKING

import jax
import numpy as np

from tomojax.backends.api import device_free_memory_bytes, is_oom_error
from tomojax.geometry import Detector, Grid, ParallelGeometry, grid_volume_origin

from ._axial_slabs import independent_planes
from .api import reconstruct_arrays
from .types import ReconstructionSlab

if TYPE_CHECKING:
    from .mlem import MLEMConfig
    from .sirt import SIRTConfig


def iterative_slabs[T](
    geometry: ParallelGeometry,
    grid: Grid,
    detector: Detector,
    read: Callable[[int, int], tuple[np.ndarray | jax.Array, T]],
    *,
    method: str,
    config: SIRTConfig | MLEMConfig,
    warm_start: bool = False,
) -> Generator[ReconstructionSlab[T], None, None]:
    """Yield owned host volumes without ever allocating the whole output.

    Exactly matching detector and voxel planes make SIRT/OSEM updates separable.
    Correction and result buffers remain owned by each yielded slab; callers may
    retain them, or discard each to keep total storage bounded. No TV/CGLS solve
    is split: their regularizer or global step lengths couple the planes.
    """
    if (
        method not in {"sirt", "mlem"}
        or config.projector_model == "ray"
        or not independent_planes(geometry, grid, detector)
    ):
        raise ValueError(
            "iterative slabs require independent axial Joseph planes "
            "(matching grid and detector rows)"
        )
    origin = grid_volume_origin(grid)
    free = device_free_memory_bytes()
    per_row = 4 * (
        grid.nx * grid.ny * (6 + 2 * getattr(config, "acceleration", False))
        + len(geometry.angles) * detector.nu * (3 + 2 / config.subsets)
    )
    budget = 512 * 2**20 if free is None else free // 2
    fitting = max(1, int(budget / per_row))
    # Bound CPU output too: the full requested volume may exceed host RAM.
    depth = min(8, 2 ** (fitting.bit_length() - 1))
    start = 0
    while start < grid.nz:
        stop = min(start + depth, grid.nz)
        count = stop - start
        data = context = volume = result = None
        try:
            data, context = read(start, stop)
            # Translate both together, reusing one compiled geometry per depth.
            local_grid = replace(
                grid,
                nz=count,
                vol_center=None,
                vol_origin=(*origin[:2], -(count - 1) * grid.vz / 2),
            )
            local_detector = replace(detector, nv=count, center=(detector.center[0], 0.0))
            local_geometry = ParallelGeometry(local_grid, local_detector, geometry.angles)
            support = config.support
            if support is not None and np.ndim(support) > 0 and np.shape(support)[-1] != 1:
                support = support[..., start:stop]
            result, info = reconstruct_arrays(
                method,
                local_geometry,
                local_grid,
                local_detector,
                data,
                config=replace(config, support=support),
                warm_start=warm_start,
            )
            volume = np.asarray(result)
            del result
        except Exception as error:
            data = context = volume = result = None
            if depth == 1 or not is_oom_error(error):
                raise
            depth = max(1, depth // 2)
            continue
        yield ReconstructionSlab(start, stop, slice(start, stop), data, volume, context, info)
        data = context = volume = result = None
        start = stop
