"""Projection-domain detector-centre estimates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import chi2

from tomojax.core.validation import validate_projection_stack
from tomojax.geometry import ParallelGeometry

from .center_statistics import PairStatistics, refine_pair
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


def _correlation_modes(spectrum: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
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
    peaks, _ = find_peaks(correlation)
    if not len(peaks):
        return np.empty(0), np.empty(0)
    peaks = peaks[np.argsort(correlation[peaks])[::-1]]
    left, middle, right = correlation[peaks - 1], correlation[peaks], correlation[peaks + 1]
    curvature = left - 2 * middle + right
    valid = np.flatnonzero(curvature < 0)[:4]
    peaks, left, middle, right, curvature = (
        value[valid] for value in (peaks, left, middle, right, curvature)
    )
    if not len(peaks):
        return np.empty(0), np.empty(0)
    lags = (peaks - extent + 0.5 * (left - right) / curvature) / upsample
    return lags, middle / middle[0]


def _paired_measurement(
    first: np.ndarray, second: np.ndarray, angle_error: float
) -> tuple[PairStatistics, np.ndarray, np.ndarray] | None:
    """Correlate corresponding rows, not their averages (phase contrast can cancel)."""
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return None
    if np.ptp(first, axis=-1).max() == 0 or np.ptp(second, axis=-1).max() == 0:
        return None
    nu = first.shape[-1]
    # Angular mismatch moves features by up to radius * delta-angle. Marginalize
    # that unresolved bandwidth rather than trying to correlate fine texture.
    # A one-pixel derivative bandwidth suppresses near-Nyquist noise before
    # interpolation. A half-pixel derivative gives that noise nearly the same
    # weight as resolved edges, making the subpixel mode unstable.
    sigma = max(1.0, nu * np.deg2rad(angle_error) / 4)
    # Derivatives remove row backgrounds without introducing the padded-edge
    # bias of subtracting a finite detector mean before linear correlation.
    a = gaussian_filter1d(first, sigma, axis=-1, order=1)
    b = gaussian_filter1d(second[:, ::-1], sigma, axis=-1, order=1)
    size = 1 << (2 * nu - 2).bit_length()
    spectrum = (np.fft.rfft(a, n=size) * np.conj(np.fft.rfft(b, n=size))).sum(axis=0)
    lags, heights = _correlation_modes(spectrum, nu)
    if not len(lags):
        return None
    return refine_pair(a, b, lag=float(lags[0]), bandwidth=sigma), -0.5 * lags[1:], heights[1:]


def _paired_offset(first: np.ndarray, second: np.ndarray, angle_error: float) -> float | None:
    result = _paired_measurement(first, second, angle_error)
    return None if result is None else result[0].offset_px


@dataclass(frozen=True)
class CenterPair:
    """Opposing-view constraint, in native detector pixels.

    ``first`` and ``second`` are ``(view index, angular interpolation weight)``
    tuples. A small-motion linearization constrains the sum of opposing
    shifts, not separate poses. Angular resampling blends images: with large
    or discontinuous motion its correlation location is not an exact weighted
    average of the contributing shifts. ``local_sigma_px`` and
    ``local_interval90_px`` describe only the selected matching mode.
    Alternatives and their correlation heights expose competing modes;
    global coverage is uncalibrated, not included in the local interval.

    Parameters
    ----------
    first, second : tuple of (int, float)
        View indices and angular interpolation weights at each end.
    angular_mismatch_deg : float
        Angular departure of the nearest acquired pair from exact opposition.
    offset_px : float
        Detector-centre placement inferred from this pair, in detector pixels.
    local_sigma_px : float
        Local matching scale; infinity denotes no local information.
        Global coverage is uncalibrated.
    local_interval90_px : tuple of float
        Nominal local 90% interval, not a calibrated global centre interval.
    alternatives_px : tuple of float
        Other correlation-peak locations, in detector-centre pixels.
    relative_peak_heights : tuple of float
        Correlation heights of alternatives relative to the selected peak.
    """

    first: tuple[tuple[int, float], ...]
    second: tuple[tuple[int, float], ...]
    angular_mismatch_deg: float
    offset_px: float
    local_sigma_px: float
    local_interval90_px: tuple[float, float]
    alternatives_px: tuple[float, ...]
    relative_peak_heights: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-native constraints, with null for unbounded uncertainty.

        Returns
        -------
        dict
            Pair constraints, conditional uncertainty and alternative modes.
        """
        return {
            "first": [list(item) for item in self.first],
            "second": [list(item) for item in self.second],
            "angular_mismatch_deg": self.angular_mismatch_deg,
            "offset_px": self.offset_px,
            "local_matching_sigma_px": (
                self.local_sigma_px if np.isfinite(self.local_sigma_px) else None
            ),
            "local_matching_interval90_px": (
                list(self.local_interval90_px)
                if np.isfinite(self.local_interval90_px).all()
                else None
            ),
            "alternatives_px": list(self.alternatives_px),
            "relative_peak_heights": list(self.relative_peak_heights),
        }


@dataclass(frozen=True)
class CenterEstimate:
    """Centre geometry and the opposing-view evidence behind it.

    A common centre is the median pair location. ``pair_scatter_px`` describes
    disagreement, not the standard error of that median. ``consistency_pvalue``
    is a conditional inverse-variance heterogeneity diagnostic; correlated
    pairs and ambiguous matches invalidate its chi-square reference. Apparent
    row slope is in horizontal pixels per detector row, not a calibrated 3D
    rotation-axis tilt. Arbitrary sample motion aliases the physical centre:
    no absolute-centre confidence interval is identifiable without a motion
    assumption. Pair intervals are local matching diagnostics; global coverage
    is explicitly uncalibrated even when the pairs do not disagree.

    Parameters
    ----------
    geometry : ParallelGeometry
        Geometry with the median pair centre in physical detector units.
    pairs : tuple of CenterPair
        Opposing-view constraints and their conditional uncertainties.
    row_positions_px : tuple of float
        Mean original detector-row index of each sampled row band.
    row_offsets_px : tuple of tuple
        Pair-by-row-band centre locations, or None for unusable bands.
    row_slope_px_per_row : float or None
        Median pair slope, or None when fewer than two row bands are usable.
    pair_scatter_px : float
        Sample standard deviation of pair centre locations, not their mean error.
    consistency_pvalue : float or None
        Conditional chi-square diagnostic, or None without two informative pairs.
    """

    geometry: ParallelGeometry
    pairs: tuple[CenterPair, ...]
    row_positions_px: tuple[float, ...]
    row_offsets_px: tuple[tuple[float | None, ...], ...]
    row_slope_px_per_row: float | None
    pair_scatter_px: float
    consistency_pvalue: float | None

    def to_dict(self) -> dict[str, object]:
        """Return the estimate and diagnostics without serializing the geometry.

        Returns
        -------
        dict
            JSON-native pair constraints, row trend and consistency diagnostics.
        """
        return {
            "offset_px": self.geometry.detector.center[0] / self.geometry.detector.du,
            "pairs": [pair.to_dict() for pair in self.pairs],
            "row_positions_px": list(self.row_positions_px),
            "row_offsets_px": [list(row) for row in self.row_offsets_px],
            "row_slope_px_per_row": self.row_slope_px_per_row,
            "pair_scatter_px": self.pair_scatter_px,
            "consistency_pvalue": self.consistency_pvalue,
            "uncertainty_scope": "conditional rigid pair match; physical centre aliases motion",
            "global_coverage_calibrated": False,
            "uncertainty_status": "local_matching_only",
            "uncertainty_reason": (
                "Local matching uncertainty is not a calibrated centre interval; "
                "wrong matching modes and sample motion are excluded."
            ),
        }


def _center_summary(
    geometry: ParallelGeometry,
    pairs: list[CenterPair],
    row_positions: np.ndarray,
    row_offsets: list[tuple[float | None, ...]],
) -> CenterEstimate:
    offsets = np.asarray([p.offset_px for p in pairs])
    sigma = np.asarray([p.local_sigma_px for p in pairs])
    good = np.isfinite(sigma) & (sigma > 0)
    pvalue = None
    if np.count_nonzero(good) > 1:
        weights = 1 / sigma[good] ** 2
        mean = np.sum(weights * offsets[good]) / weights.sum()
        statistic = float(np.sum(weights * (offsets[good] - mean) ** 2))
        pvalue = float(chi2.sf(statistic, np.count_nonzero(good) - 1))
    slopes = []
    for values in row_offsets:
        y = np.asarray(values, dtype=float)
        valid = np.isfinite(y)
        if np.count_nonzero(valid) > 1:
            x = row_positions[valid] - np.mean(row_positions[valid])
            slopes.append(float(np.sum(x * y[valid]) / np.sum(x**2)))
    detector = replace(
        geometry.detector,
        center=(float(np.median(offsets)) * geometry.detector.du, geometry.detector.center[1]),
    )
    return CenterEstimate(
        replace(geometry, detector=detector),
        tuple(pairs),
        tuple(float(r) for r in row_positions),
        tuple(row_offsets),
        float(np.median(slopes)) if slopes else None,
        float(np.std(offsets, ddof=1)) if len(offsets) > 1 else 0.0,
        pvalue,
    )


def estimate_center_report(
    geometry: Geometry, projections: np.ndarray | jax.Array
) -> CenterEstimate:
    """Estimate an axial parallel centre and retain pair and row diagnostics.

    At most sixteen near-opposite pairs and 64 rows are sampled. Gradient
    correlations select matching modes; a symmetric fit on overlapping
    support estimates each location. Eight detector bands give conditional
    sandwich standard errors with leverage correction. Up to eight row bands
    measure the apparent row-centre trend. No reconstruction is performed.

    Parameters
    ----------
    geometry : Geometry
        Unposed axial parallel geometry, including angles in degrees.
    projections : array-like
        Corrected line integrals shaped (views, rows, columns).

    Returns
    -------
    CenterEstimate
        Geometry, weighted view-pair constraints, conditional 90% intervals,
        alternative correlation peaks, pair disagreement and row-centre slope.
        Global coverage is uncalibrated: sample motion and wrong-mode selection
        are outside the local intervals, including for a single opposing pair.

    Raises
    ------
    ValueError
        If geometry is unsupported, usable opposing data are absent, or
        near-opposite pairs lack angular neighbours within two degrees.
        Exact opposing views do not require angular interpolation.
    """
    if type(geometry) is not ParallelGeometry:
        raise ValueError("estimate_center requires an unposed axial ParallelGeometry")
    geometry = cast("ParallelGeometry", geometry)
    validate_projection_stack(
        projections, geometry.detector, geometry=geometry, context="estimate_center"
    )
    detector = geometry.detector
    samples = _sample_pairs(np.asarray(geometry.angles, np.float64), detector.nu)
    rows = np.linspace(0, detector.nv - 1, min(64, detector.nv)).round().astype(int)
    groups = np.array_split(np.arange(len(rows)), min(8, len(rows)))
    row_positions = np.asarray([np.mean(rows[group]) for group in groups])
    pairs, row_offsets = [], []
    if samples:
        views = np.unique([i for first, second, _ in samples for i, _ in first + second])
        data = np.asarray(projections[views[:, None], rows[None, :], :], np.float64)
        for first, second, error in samples:
            a = np.asarray(sum(w * data[np.searchsorted(views, i)] for i, w in first))
            b = np.asarray(sum(w * data[np.searchsorted(views, i)] for i, w in second))
            result = _paired_measurement(a, b, error)
            if result is None:
                continue
            stat, alternatives, heights = result
            pairs.append(
                CenterPair(
                    tuple(first),
                    tuple(second),
                    error,
                    stat.offset_px,
                    stat.sigma_px,
                    stat.interval90_px,
                    tuple(float(x) for x in alternatives),
                    tuple(float(x) for x in heights),
                )
            )
            row_offsets.append(tuple(_paired_offset(a[group], b[group], error) for group in groups))
    if not pairs:
        raise ValueError(
            "estimate_center needs finite, nonconstant opposite projections. "
            "Near-opposite pairs need angular neighbours within 2 degrees; "
            "use finer angular sampling when subsampling views."
        )
    return _center_summary(geometry, pairs, row_positions, row_offsets)


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
    return estimate_center_report(geometry, projections).geometry


__all__ = [
    "CenterEstimate",
    "CenterPair",
    "DetectorCenterSeed",
    "estimate_center",
    "estimate_center_report",
    "projection_com_det_u_seed",
    "projection_pair_det_u_seed",
    "train_heldout_view_indices",
]
