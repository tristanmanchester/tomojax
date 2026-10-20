"""FBP/FDK initialization retaining all calibrated per-view arrangements and poses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from tomojax.core.geometry.cone import is_cone_beam
from tomojax.core.validation import validate_projection_stack

from .fbp import FBPConfig, fbp

if TYPE_CHECKING:
    import numpy as np

    from tomojax.core.geometry.base import Detector, Geometry, Grid


def fbp_start(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    *,
    config: FBPConfig,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    out: np.ndarray | None = None,
) -> jnp.ndarray | np.ndarray:
    """An approximate FBP/FDK image; iterative steps use the matched projectors.

    Cone volumes given host storage ``out``, or too large for the device, are
    made by host FDK slabs into ``out`` or a new Fortran-ordered host array.
    """
    validate_projection_stack(
        projections, detector, geometry=geometry, context="warm_start projections"
    )
    if is_cone_beam(geometry) and (out is not None or _exceeds_device(grid)):
        import numpy as np

        from .fdk import FDKConfig, FDKHostConfig, fdk_host

        target = (
            np.empty((grid.nx, grid.ny, grid.nz), np.float32, order="F") if out is None else out
        )
        host = FDKHostConfig(fdk=FDKConfig(devices=config.devices))
        return fdk_host(geometry, grid, detector, np.asarray(projections), config=host, out=target)
    return fbp(geometry, grid, detector, projections, config=config, det_grid=det_grid)


def _exceeds_device(grid: Grid) -> bool:
    """Whether a volume would take more than 40% of free device memory, as for public FBP."""
    from tomojax.backends import device_free_memory_bytes

    free = device_free_memory_bytes()
    return free is not None and 4 * grid.nx * grid.ny * grid.nz > 0.4 * free
