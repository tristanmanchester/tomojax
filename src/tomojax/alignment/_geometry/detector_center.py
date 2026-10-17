"""Projection-domain detector-centre estimates."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np

from tomojax.core.validation import validate_projection_stack
from tomojax.geometry import ParallelGeometry

from .initializers import (
    DetectorCenterSeed,
    projection_com_det_u_seed,
    projection_pair_det_u_seed,
    sinogram_det_u_seed,
    train_heldout_view_indices,
)

if TYPE_CHECKING:
    import jax

    from tomojax.geometry import Geometry


def estimate_center(geometry: Geometry, projections: np.ndarray | jax.Array) -> ParallelGeometry:
    """Estimate an axial parallel detector's absolute physical u centre from line integrals.

    Vo's half-turn Fourier estimator on up to three detector rows. Geometry only:
    nothing is reconstructed. Non-axial, posed and partial-arc geometries raise.

    Parameters
    ----------
    geometry : Geometry
        Scan geometry with its view angles and physical detector placement.
    projections : array-like
        Line integrals with shape (views, nv, nu), in geometry view order.

    Returns
    -------
    ParallelGeometry
        Geometry with the estimated absolute physical detector-u centre.

    Raises
    ------
    ValueError
        If the scan is non-axial, posed, lacks half-turn coverage or has incompatible
        data.
    """
    if type(geometry) is not ParallelGeometry:
        raise ValueError("estimate_center requires an unposed axial ParallelGeometry")
    geometry = cast("ParallelGeometry", geometry)
    validate_projection_stack(
        projections, geometry.detector, geometry=geometry, context="estimate_center"
    )
    seed = sinogram_det_u_seed(projections, np.asarray(geometry.angles))
    if seed is None:
        raise ValueError(
            "estimate_center needs a usable half-turn (at least 16 views spanning 175 degrees) "
            "and finite, nonconstant sampled sinograms"
        )
    detector = geometry.detector
    moved = replace(detector, center=(seed.det_u_px * detector.du, detector.center[1]))
    return replace(geometry, detector=moved)


__all__ = [
    "DetectorCenterSeed",
    "estimate_center",
    "projection_com_det_u_seed",
    "projection_pair_det_u_seed",
    "train_heldout_view_indices",
]
