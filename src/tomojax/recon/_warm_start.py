"""FBP/FDK initialization, retaining each cone segment's calibrated poses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from tomojax.core.geometry.cone import cone_parts, segments_of
from tomojax.core.validation import validate_projection_stack

from .fbp import FBPConfig, fbp

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from tomojax.core.geometry.base import Detector, Geometry, Grid, PoseMatrix
    from tomojax.core.geometry.cone import ConeBeam


@dataclass(frozen=True)
class _SegmentViews:
    """One arrangement's view labels with the outer geometry's corrected poses."""

    geometry: Geometry
    offset: int
    angles: Sequence[float]
    beam: ConeBeam

    def pose_for_view(self, i: int) -> PoseMatrix:
        return self.geometry.pose_for_view(self.offset + i)


def fbp_start(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    *,
    config: FBPConfig,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> jnp.ndarray:
    """An FBP start, or a view-count-weighted mean of each segment's FDK.

    Segments must each cover an arc accepted by FDK. This is only an
    approximate initial image, not a segmented FDK inverse or an adjoint;
    subsequent iterative steps use the original matched projectors.
    """
    segments = segments_of(geometry)
    if segments is None:
        return fbp(geometry, grid, detector, projections, config=config, det_grid=det_grid)
    if det_grid is not None:
        raise ValueError("warm_start: cone-beam geometry needs the canonical detector grid")
    n_views, _, _ = validate_projection_stack(
        projections, detector, geometry=geometry, context="warm_start projections"
    )
    parts = cone_parts(geometry, detector)
    assert parts is not None
    start, offset = None, 0
    for segment, (_, beam, own_detector) in zip(segments.segments, parts, strict=True):
        count = len(segment.angles)
        views = _SegmentViews(geometry, offset, segment.angles, beam)
        image = fbp(views, grid, own_detector, projections[offset : offset + count], config=config)
        weighted = (count / n_views) * image
        start = weighted if start is None else start + weighted
        offset += count
    assert start is not None
    return start
