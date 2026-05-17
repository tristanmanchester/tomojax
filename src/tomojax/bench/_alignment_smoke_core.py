"""Tiny v2 alignment smoke path."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.lax

from tomojax.forward import project_parallel_reference, residual_loss
from tomojax.geometry import CanonicalizedGeometry, GeometryState, canonicalize_geometry_gauges
from tomojax.recon import reconstruct_average_reference


@dataclass(frozen=True)
class AlignmentSmokeReport:
    initial_loss: float
    canonical_loss: float
    valid_count: float
    canonicalized_geometry: CanonicalizedGeometry


def run_alignment_smoke(
    observed: jax.Array,
    geometry: GeometryState,
    *,
    mask: jax.Array | None = None,
    sigma: float = 1.0,
    delta: float = 1.0,
) -> AlignmentSmokeReport:
    """Run a tiny stopped-volume projection residual and gauge smoke check."""
    preview_volume = jax.lax.stop_gradient(reconstruct_average_reference(observed))
    initial_predicted = project_parallel_reference(preview_volume, geometry)
    initial = residual_loss(initial_predicted, observed, mask=mask, sigma=sigma, delta=delta)

    canonicalized = canonicalize_geometry_gauges(geometry)
    canonical_predicted = project_parallel_reference(preview_volume, canonicalized.state)
    canonical = residual_loss(canonical_predicted, observed, mask=mask, sigma=sigma, delta=delta)

    return AlignmentSmokeReport(
        initial_loss=float(initial.loss),
        canonical_loss=float(canonical.loss),
        valid_count=float(canonical.valid_count),
        canonicalized_geometry=canonicalized,
    )
