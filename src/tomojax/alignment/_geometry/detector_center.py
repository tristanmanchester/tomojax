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

    Vo's half-turn Fourier estimator uses up to three detector rows. Wrapped
    or reversed angles are ordered by direction, repeated angles averaged,
    and irregular angular samples linearly resampled before the Fourier metric.
    This estimates geometry only: it neither reconstructs nor refines a volume.
    Non-axial, posed and partial-arc geometries are not supported.
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
