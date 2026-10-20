"""FBP/FDK initialization retaining all calibrated per-view arrangements and poses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from tomojax.core.validation import validate_projection_stack

from ._fbp_storage import fbp_volume

if TYPE_CHECKING:
    import numpy as np

    from tomojax.core.geometry.base import Detector, Geometry, Grid

    from .fbp import FBPConfig


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

    Volumes given host storage ``out``, or too large for the device, are made
    in host slabs as public FBP makes them, into ``out`` or a new host array.
    """
    validate_projection_stack(
        projections, detector, geometry=geometry, context="warm_start projections"
    )
    return fbp_volume(geometry, grid, detector, projections, config, det_grid, out)[0]
