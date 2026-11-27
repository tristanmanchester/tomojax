"""What a scan's rows say about an object that extends outside the field of view.

The analytic methods continue each detector row past the detector's edges as
the projection of a cylinder about the rotation axis
(:mod:`tomojax.recon._continuation`). :func:`truncation_record` measures, from
the same fits, how far the rows run past the detector and how much an interior
value rests on the continuation. The methods that fit a grid take the data as
measured: they are exact when the grid holds the object, and the record says
when the rows do not end inside the detector.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from tomojax.core.geometry.cone import cone_parts

from ._continuation import (
    Continuation,
    cone_axis_distance,
    continued_columns,
    parallel_axis_distance,
    radius_views,
)

if TYPE_CHECKING:
    import jax

    from tomojax.geometry import Detector, Geometry

    from ._continuation import Host

# Views and rows sampled for the record.
_RECORD_SAMPLES = 64


def _axis_distance(geometry: Geometry, detector: Detector, pad: int) -> Host:
    """Distance from the rotation axis of each ray of a row continued by ``pad`` columns a side."""
    parts = cone_parts(geometry, detector)
    if parts is None:
        return parallel_axis_distance(detector, detector.nu, pad)
    return cone_axis_distance(parts[0][1], parts[0][2], pad)


def truncation_record(
    geometry: Geometry, detector: Detector, projections: np.ndarray | jax.Array
) -> dict[str, float]:
    """How far a scan's rows run past the detector, and what the continuation does about it.

    Parameters
    ----------
    geometry : Geometry
        Scan geometry.
    detector : Detector
        Detector of the projections.
    projections : numpy.ndarray or jax.Array
        Line integrals, shape ``(views, nv, nu)``.

    Returns
    -------
    dict
        ``edge_ratio``: the mean line integral at the detector's edges over the
        mean of the rows, zero for an object inside the field of view.
        ``outside_fraction``: the share of the continued scan's attenuation
        that lies beyond the detector, in the continuation and not the
        measurement. ``interior_shift``: the change the continuation makes to
        the value on the rotation axis, in the volume's units (per length);
        negative, as zero beyond the detector counts outside material inside.
        ``radius``: the median radius of the rows' fitted cylinders, and
        ``field_of_view_radius``: the distance from the axis to the nearer
        detector edge, both in the scan's length unit.

    Notes
    -----
    Computed from at most 64 evenly spaced views and rows. What lies beyond
    the detector is a model: interior values carry an unknown smooth
    background of the order of a fraction of ``interior_shift``, while edges,
    contrast between neighbouring regions and fine detail do not depend on it.
    """
    views, nv, nu = projections.shape
    pad = continued_columns(nu)
    distance = _axis_distance(geometry, detector, pad)
    fits = Continuation(distance, nu, pad)
    view = radius_views(views)[:: max(1, math.ceil(min(views, 256) / _RECORD_SAMPLES))]
    row = np.unique(np.linspace(0, nv - 1, min(nv, _RECORD_SAMPLES)).round().astype(np.int64))
    picked: Host = np.asarray(projections[view[:, None], row[None, :]], np.float64)
    sample = picked.swapaxes(0, 1)  # (rows, views, columns)
    low, high = sample[..., : fits.window], sample[..., nu - fits.window :]
    radius2: Host = np.asarray(fits.radius(np.asarray, low, high), np.float64)
    amplitudes = fits.amplitudes(np.asarray, low, high, radius2)
    edges: Host = np.concatenate(amplitudes)
    # All views share each row's profile. Its integral depends only on the
    # mean edge amplitudes, so form one pair of tails per row.
    average = tuple(a.mean(axis=1, keepdims=True) for a in amplitudes)
    tails: Host = np.concatenate(fits.tails(np.asarray, average[0], average[1], radius2), axis=-1)
    # Each ray's share of the row, and of the level on the axis: FBP weighs a ray at
    # distance s from the axis by -1 / (2 pi**2 s**2) there, over a half turn.
    width: Host = abs(np.gradient(distance))
    outside = np.concatenate([np.arange(pad), np.arange(pad + nu, nu + 2 * pad)])
    measured = float(np.sum(sample * width[pad : pad + nu]))
    continued = float(np.sum(tails * width[outside])) * view.size
    on_axis: Host = np.sum(tails * (width[outside] / distance[outside] ** 2), axis=-1)
    total = measured + continued
    mean = float(np.mean(sample))
    return {
        "edge_ratio": float(np.mean(edges)) / mean if mean else 0.0,
        "outside_fraction": continued / total if total else 0.0,
        "interior_shift": -float(np.mean(on_axis)) / (2 * np.pi),
        "radius": float(np.median(radius2**0.5)) * fits.scale,
        "field_of_view_radius": min(abs(distance.item(pad)), abs(distance.item(pad + nu - 1))),
    }
