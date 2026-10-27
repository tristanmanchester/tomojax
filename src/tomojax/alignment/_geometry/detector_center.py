"""Projection-domain detector-centre estimates."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.ndimage import gaussian_filter1d

from tomojax.core.validation import validate_projection_stack
from tomojax.geometry import ParallelGeometry

from .initializers import (
    DetectorCenterSeed,
    projection_com_det_u_seed,
    projection_pair_det_u_seed,
    train_heldout_view_indices,
)

if TYPE_CHECKING:
    import jax

    from tomojax.geometry import Geometry


def _opposite_pairs(angles: np.ndarray, width: int) -> list[tuple[int, int, float]]:
    """At most sixteen uniformly distributed near-opposite view pairs."""
    angles = np.mod(np.asarray(angles, np.float64), 360)
    if len(angles) < 2:
        return []
    order = np.argsort(angles, kind="stable")
    targets = (angles + 180) % 360
    right = np.searchsorted(angles[order], targets)
    candidates = np.column_stack((order[(right - 1) % len(order)], order[right % len(order)]))
    distances = np.abs((angles[candidates] - targets[:, None] + 180) % 360 - 180)
    which = np.argmin(distances, axis=1)
    pairs: dict[tuple[int, int], float] = {}
    for i, (j, error) in enumerate(
        zip(
            candidates[np.arange(len(angles)), which],
            distances[np.arange(len(angles)), which],
            strict=True,
        )
    ):
        if i != j and error <= 2:
            pair = (i, int(j)) if angles[i] <= angles[j] else (int(j), i)
            pairs[pair] = float(error)
    if not pairs:
        return []
    # Small encoder variations must not concentrate all evidence in one sector.
    # Admit pairs within half a detector-edge pixel of the best angular match:
    # radius (width / 2) times the extra angle is at most half a pixel.
    # Local angular resampling below removes the remaining mismatch.
    best = min(pairs.values())
    tolerance = np.rad2deg(1.0 / width)
    ordered = sorted(
        (p for p, error in pairs.items() if error <= best + tolerance),
        key=lambda p: (angles[p[0]], angles[p[1]]),
    )
    # Spread the evidence in angle, not in acquisition density.
    directions = np.asarray([angles[p[0]] for p in ordered])
    targets = np.linspace(directions[0], directions[-1], min(16, len(ordered)))
    right = np.clip(np.searchsorted(directions, targets), 0, len(ordered) - 1)
    left = np.maximum(right - 1, 0)
    picks = np.unique(
        np.where(targets - directions[left] <= directions[right] - targets, left, right)
    )
    return [(*ordered[i], pairs[ordered[i]]) for i in picks]


def _angular_sample(angles: np.ndarray, index: int, shift: float) -> list[tuple[int, float]]:
    """Locally interpolate/extrapolate one view to an exactly opposing direction."""
    if abs(shift) < 1e-4:
        return [(index, 1.0)]
    delta = (angles - angles[index] + 180) % 360 - 180
    local = np.flatnonzero((np.abs(delta) > 1e-4) & (np.abs(delta) <= 2))
    if not len(local):
        return []
    directed = local[delta[local] * shift > 0]
    candidates = directed if len(directed) else local
    neighbor = int(candidates[np.argmin(np.abs(delta[candidates]))])
    weight = float(shift / delta[neighbor])
    return [(index, 1 - weight), (neighbor, weight)]


def _sample_pairs(
    angles: np.ndarray,
    width: int,
) -> list[tuple[list[tuple[int, float]], list[tuple[int, float]], float]]:
    pairs = []
    for i, j, error in _opposite_pairs(angles, width):
        # Share the interpolation distance between both ends instead of treating
        # a nearly opposite pair as exact (which confounds angle and centre).
        delta = float((angles[j] - angles[i]) % 360 - 180)
        first = _angular_sample(angles, i, delta / 2)
        second = _angular_sample(angles, j, -delta / 2)
        if first and second:
            pairs.append((first, second, error))
    return pairs


def _correlation_peak(spectrum: np.ndarray, width: int) -> float | None:
    """Find the global subpixel peak, not just the largest integer-lag sample."""
    # Narrow competing peaks can exchange their sampled heights as the detector
    # moves by a fraction of a pixel. Interpolate the *whole* correlation before
    # choosing its mode. Refining only the winning integer sample cannot fix it.
    upsample = 8
    size = 2 * (len(spectrum) - 1)
    spectrum = spectrum.copy()
    # The original Nyquist coefficient splits into positive/negative frequencies
    # when zero-padding a real spectrum; irfft supplies the negative half.
    spectrum[-1] *= 0.5
    correlation = np.fft.irfft(spectrum, n=size * upsample)
    extent = (width - 1) * upsample
    correlation = np.r_[correlation[-extent:], correlation[: extent + 1]]
    peak = int(np.argmax(correlation))
    if peak == 0 or peak == len(correlation) - 1:
        return None  # No overlapping support at an endpoint maximum.
    left, middle, right = correlation[peak - 1 : peak + 2]
    curvature = left - 2 * middle + right
    if curvature >= 0:
        return None
    return float(peak - extent + 0.5 * (left - right) / curvature) / upsample


def _paired_offset(first: np.ndarray, second: np.ndarray, angle_error: float) -> float | None:
    """Correlate corresponding rows, not their averages (phase contrast can cancel)."""
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return None
    if np.ptp(first, axis=-1).max() == 0 or np.ptp(second, axis=-1).max() == 0:
        return None
    nu = first.shape[-1]
    # Angular mismatch moves features by up to radius * delta-angle. Marginalize
    # that unresolved bandwidth rather than trying to correlate fine texture.
    sigma = max(0.5, nu * np.deg2rad(angle_error) / 4)
    # Derivatives remove row backgrounds without introducing the padded-edge
    # bias of subtracting a finite detector mean before linear correlation.
    a = gaussian_filter1d(first, sigma, axis=-1, order=1)
    b = gaussian_filter1d(second[:, ::-1], sigma, axis=-1, order=1)
    size = 1 << (2 * nu - 2).bit_length()
    spectrum = (np.fft.rfft(a, n=size) * np.conj(np.fft.rfft(b, n=size))).sum(axis=0)
    lag = _correlation_peak(spectrum, nu)
    return None if lag is None else -0.5 * lag


def estimate_center(geometry: Geometry, projections: np.ndarray | jax.Array) -> ParallelGeometry:
    """Estimate an axial parallel detector's absolute physical u centre from projections.

    Opposing projections agree after detector-u reflection and a shift of twice
    the centre offset. Gradient correlations of up to 64 corresponding rows are summed,
    retaining signed phase contrast rather than assuming positive absorption.
    At most sixteen near-opposite pairs (within two degrees), spread across the
    scan, and their angular neighbors are read. Pairs within half a detector-edge
    pixel of the best angular match are eligible. Linear angular resampling
    removes the mismatch. Nothing is reconstructed. Non-axial and posed
    geometries raise.

    Parameters
    ----------
    geometry : Geometry
        Scan geometry with its view angles and physical detector placement.
    projections : array-like
        Projections with shape (views, nv, nu), in geometry view order.

    Returns
    -------
    ParallelGeometry
        Geometry with the estimated absolute physical detector-u centre.

    Raises
    ------
    ValueError
        If the scan is non-axial, posed, lacks usable near-opposite projections,
        or has incompatible data.
    """
    if type(geometry) is not ParallelGeometry:
        raise ValueError("estimate_center requires an unposed axial ParallelGeometry")
    geometry = cast("ParallelGeometry", geometry)
    validate_projection_stack(
        projections, geometry.detector, geometry=geometry, context="estimate_center"
    )
    detector = geometry.detector
    pairs = _sample_pairs(np.asarray(geometry.angles, np.float64), detector.nu)
    offsets = []
    if pairs:
        views = np.unique([i for first, second, _ in pairs for i, _ in first + second])
        rows = np.linspace(0, detector.nv - 1, min(64, detector.nv)).round().astype(int)
        # One bounded gather for device arrays and lazy corrected detector rows.
        data = np.asarray(projections[views[:, None], rows[None, :], :], np.float64)
        for first, second, error in pairs:
            a = sum(weight * data[np.searchsorted(views, i)] for i, weight in first)
            b = sum(weight * data[np.searchsorted(views, i)] for i, weight in second)
            offset = _paired_offset(np.asarray(a), np.asarray(b), error)
            if offset is not None:
                offsets.append(offset)
    if not offsets:
        raise ValueError(
            "estimate_center needs finite, nonconstant near-opposite projections (within 2 degrees)"
        )
    moved = replace(detector, center=(float(np.median(offsets)) * detector.du, detector.center[1]))
    return replace(geometry, detector=moved)


__all__ = [
    "DetectorCenterSeed",
    "estimate_center",
    "projection_com_det_u_seed",
    "projection_pair_det_u_seed",
    "train_heldout_view_indices",
]
