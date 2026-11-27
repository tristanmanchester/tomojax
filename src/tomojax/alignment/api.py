"""Detector disturbance screening and opposing-view center measurements."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tomojax.alignment._config import AlignConfig
from tomojax.alignment._geometry.detector_center import (
    CenterEstimate,
    CenterPair,
    estimate_center,
    estimate_center_report,
)
from tomojax.alignment._integration import acquisition_segments, trajectory_positions
from tomojax.alignment._joint._disturbance import DisturbedViews, disturbed_report
from tomojax.alignment._joint._fit import preview_disturbance
from tomojax.alignment._joint._pose import ViewOrder

if TYPE_CHECKING:
    from tomojax.geometry import ScanGeometry


def disturbed_views(
    projections: np.ndarray,
    geometry: ScanGeometry,
    *,
    view_positions: np.ndarray | None = None,
    resolution: int = 320,
) -> DisturbedViews:
    """Measure transient detector-column disturbances before a joint alignment fit.

    Parameters
    ----------
    projections : numpy.ndarray
        Corrected line integrals with shape (view, row, column).
    geometry : ScanGeometry
        The acquisition geometry of those views.
    view_positions : numpy.ndarray or None, optional
        Acquisition-time positions of the views, as on the scan. Decreases
        mark trajectory boundaries. None uses successive indices.
    resolution : int, optional
        Working detector width of the intended fit. Default is 320.

    Returns
    -------
    dict
        Report with ``score`` (per-view shared power fraction between the top
        and bottom detector bands, or None when too few paired measurements
        exist; values above 0.5 flag a view), ``ranges`` (flagged zero-based
        half-open ``[start, stop]`` index intervals), ``angles`` (first and last
        exposure angle in degrees for each range), and ``message`` (review
        summary).

    Raises
    ------
    ValueError
        If ``projections`` does not match the view count and detector shape of
        ``geometry``.
    """
    config = AlignConfig(resolution=resolution)
    segments = acquisition_segments(geometry, view_positions)
    if projections.shape != (len(segments), geometry.detector.nv, geometry.detector.nu):
        raise ValueError("projections must match the view count and detector shape")
    order = ViewOrder(len(segments), trajectory_positions(view_positions, segments))
    scores = preview_disturbance(projections, geometry.grid, geometry.detector, config, order)
    return disturbed_report(scores, np.asarray(geometry.angles), segments)


__all__ = [
    "CenterEstimate",
    "CenterPair",
    "disturbed_views",
    "estimate_center",
    "estimate_center_report",
]
