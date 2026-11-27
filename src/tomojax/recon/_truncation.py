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

from functools import lru_cache
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
    from collections.abc import Callable

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
    contrast between neighboring regions and fine detail do not depend on it.
    """
    return record_values(truncation_values(geometry, detector, projections))


def record_values(values: np.ndarray | jax.Array) -> dict[str, float]:
    """Collect the five scalar diagnostics after their projection band has completed."""
    keys = ("edge_ratio", "outside_fraction", "interior_shift", "radius", "field_of_view_radius")
    return dict(zip(keys, map(float, np.asarray(values)), strict=True))


def truncation_values(
    geometry: Geometry, detector: Detector, projections: np.ndarray | jax.Array
) -> np.ndarray | jax.Array:
    """Submit the record on the source device; only the five scalars travel to the host."""
    distance = _axis_distance(geometry, detector, continued_columns(projections.shape[-1]))
    host = isinstance(projections, np.ndarray)
    measure = _measure(distance.tobytes(), host=host)
    if host:
        with np.errstate(invalid="ignore", divide="ignore"):
            return measure(projections)
    import jax

    with jax.enable_x64(True):
        values = measure(projections)
        jax.copy_to_host_async(values)
        return values


@lru_cache(maxsize=16)
def _measure(
    coordinates: bytes, *, host: bool
) -> Callable[[np.ndarray | jax.Array], np.ndarray | jax.Array]:
    """Prepare a sampled row reduction for one detector's physical column coordinates."""
    distance: Host = np.frombuffer(coordinates, np.float64)
    nu = len(distance) // 3
    pad = continued_columns(nu)
    fits = Continuation(distance, nu, pad)
    width = abs(np.gradient(distance))
    outside = np.r_[0:pad, pad + nu : pad + nu + pad]
    fov = min(abs(distance.item(pad)), abs(distance.item(pad + nu - 1)))
    if host:
        xp = np
    else:
        import jax
        import jax.numpy as xp

    def measure(projections: np.ndarray | jax.Array) -> np.ndarray | jax.Array:
        views, nv, _ = projections.shape
        view = radius_views(views)[:: max(1, math.ceil(min(views, 256) / _RECORD_SAMPLES))]
        row = np.unique(np.linspace(0, nv - 1, min(nv, _RECORD_SAMPLES)).round().astype(np.int64))
        sample = projections[view[:, None], row[None, :]].astype(xp.float64).swapaxes(0, 1)
        low, high = sample[..., : fits.window], sample[..., nu - fits.window :]
        radius2 = fits.radius(xp.asarray, low, high)
        low, high = fits.amplitudes(xp.asarray, low, high, radius2)
        low, high = low.mean(axis=1, keepdims=True), high.mean(axis=1, keepdims=True)
        before, after = fits.tails(xp.asarray, low, high, radius2)
        tails = xp.concatenate((before, after), axis=-1)
        measured = xp.sum(sample * xp.asarray(width[pad : pad + nu]))
        continued = xp.sum(tails * xp.asarray(width[outside])) * view.size
        on_axis = (tails * xp.asarray(width[outside] / distance[outside] ** 2)).sum(axis=-1)
        total, mean = measured + continued, xp.mean(sample)
        return xp.stack(
            (
                xp.where(mean != 0, (low.mean() + high.mean()) / (2 * mean), 0),
                xp.where(total != 0, continued / total, 0),
                -xp.mean(on_axis) / (2 * np.pi),
                xp.median(radius2**0.5) * fits.scale,
                xp.asarray(fov),
            )
        )

    return measure if host else jax.jit(measure)
