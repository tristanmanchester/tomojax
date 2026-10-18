"""FBP/FDK initialization retaining all calibrated per-view arrangements and poses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

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
) -> jnp.ndarray:
    """An approximate FBP/FDK image; iterative steps use the matched projectors."""
    validate_projection_stack(
        projections, detector, geometry=geometry, context="warm_start projections"
    )
    return fbp(geometry, grid, detector, projections, config=config, det_grid=det_grid)
