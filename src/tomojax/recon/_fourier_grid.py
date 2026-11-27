"""Physical grids and interpolation coefficients for Fourier reconstruction."""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING

import numpy as np

from tomojax.errors import InvalidParameter
from tomojax.geometry import Detector, Grid, grid_volume_origin

from tomojax.core.geometry.coverage import REPEAT_TOLERANCE, group_repeats

if TYPE_CHECKING:
    import jax
    from numpy.typing import NDArray

    from tomojax._typed_arrays import Float64Array


def next_fast_len(target: int) -> int:
    """Return the smallest 5-smooth length >= ``target``, as ``scipy.fft`` does for real FFTs.

    A local helper keeps SciPy's comparatively slow FFT import off the reconstruction path.
    """
    if target <= 6:
        return max(target, 1)
    best = 1 << (target - 1).bit_length()
    power5 = 1
    while power5 < best:
        odd = power5
        while odd < best:
            # Smallest odd * 2**k reaching the target.
            best = min(best, odd << (-(-target // odd) - 1).bit_length())
            odd *= 3
        power5 *= 5
    return best


def sorted_half_turn(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    """Sort half-turn directions, retaining redundant exposures and opposing orientations."""
    if angles.ndim != 1 or angles.size < 2 or not np.isfinite(angles).all():
        raise ValueError("fourier_reconstruct: requires at least two finite angles")
    degrees = angles.astype(np.float64)
    # Opposing rays share one Fourier direction. Doubling the angle gives the
    # same rounding tolerance as parallel FBP for recorded and corrected poses.
    representatives, inverse, counts = group_repeats(2 * np.deg2rad(degrees))
    if representatives.size < 2:
        raise ValueError("fourier_reconstruct: requires at least two distinct half-turn directions")
    reduced = representatives[inverse] / 2
    unique = counts[inverse] == 1
    reduced[unique] = np.deg2rad(np.mod(degrees[unique], 180.0))
    reduced[np.minimum(reduced, np.pi - reduced) <= REPEAT_TOLERANCE / 2] = 0
    order = np.argsort(reduced, kind="stable")
    flipped = np.cos(np.deg2rad(degrees[order]) - reduced[order]) < 0
    return order, flipped.astype(np.int8), tuple(reduced[order])


def transform_grid(grid: Grid, detector: Detector) -> tuple[tuple[int, int], tuple[int, int]]:
    """Pad around both the requested ROI and the zero-extended detector field."""
    radius = abs(detector.center[0]) + detector.nu * detector.du / 2
    origin = grid_volume_origin(grid)
    shape, crop = [], []
    for count, spacing, first in zip(
        (grid.ny, grid.nx), (grid.vy, grid.vx), (origin[1], origin[0]), strict=True
    ):
        lo = min(0, math.floor((-radius - first) / spacing))
        hi = max(count - 1, math.ceil((radius - first) / spacing))
        support = hi - lo + 1
        padded = next_fast_len(2 * support)
        shape.append(padded)
        crop.append(-lo + (padded - support) // 2)
    return (shape[0], shape[1]), (crop[0], crop[1])


@functools.lru_cache(maxsize=16)
def radial_coefficients(nu: int, nfft: int, du: float) -> tuple[np.ndarray, ...]:
    """Six-point Kaiser--Bessel interpolation with analytic deapodization."""
    width, mid = 6, (nu - 1) / 2
    ratio = nfft / nu
    beta = np.pi * np.sqrt((width / ratio * (ratio - 0.5)) ** 2 - 0.8)
    phase = du * np.exp(2j * np.pi * np.arange(nfft // 2 + 1) * mid / nfft)
    positions = (np.arange(nu) - mid) / nfft
    argument = np.sqrt(beta**2 - (np.pi * width * positions) ** 2)
    deapod = argument * np.i0(beta) / (width * np.sinh(argument))
    offset = np.linspace(0, width / 2, 16385)
    table = np.i0(beta * np.sqrt(np.maximum(0, 1 - (2 * offset / width) ** 2))) / np.i0(beta)
    result = (phase, deapod, table)
    for array in result:
        _ = array.setflags(write=False)
    return result


def check_samples(samples: np.ndarray) -> None:
    """Reject a nonfinite sampled FP32 detector value at its input field."""
    if not np.isfinite(samples).all():
        raise InvalidParameter(
            "source", "fourier_reconstruct: sampled projection rows must be finite in FP32"
        )


def sample_projection_rows(
    projections: np.ndarray | jax.Array,
    order: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Interpolate physical detector-v rows with zero extension and host storage."""
    if positions.ndim == 2:
        samples, fraction = gather_per_view_rows(projections, order, positions)
        count = len(positions)
        samples[:count] *= (1 - fraction)[..., None]
        samples[count:] *= fraction[..., None]
        return samples[:count] + samples[count:]
    nv = projections.shape[1]
    # Far outside rows contribute zero, including coordinates beyond int64.
    positions = np.clip(positions, -1, nv)
    indices = np.floor(positions).astype(np.int64)
    fraction = np.asarray(positions - indices, dtype=np.float32)
    if np.all(fraction == 0) and np.all((indices >= 0) & (indices < nv)):
        # Advanced indexing puts the selected rows first, the FFT batch layout.
        data = projections[order[None, :], indices[:, None], :]
        return np.ascontiguousarray(data, dtype=np.float32)
    # Advanced indexing creates independent writable arrays. Reuse those
    # copies instead of allocating weighted arrays and fancy-indexed output
    # temporaries for both interpolation neighbors.
    if np.all((fraction > 0) & (fraction < 1) & (indices >= 0) & (indices + 1 < nv)):
        lower = np.require(
            projections[order[None, :], indices[:, None], :], np.float32, requirements=["W"]
        )
        upper = np.require(
            projections[order[None, :], (indices + 1)[:, None], :], np.float32, requirements=["W"]
        )
        lower = np.multiply(lower, (1 - fraction)[:, None, None], out=lower)
        upper = np.multiply(upper, fraction[:, None, None], out=upper)
        lower = np.add(lower, upper, out=lower)
        # Match the original zero-initialized accumulation's signed zeros.
        return np.add(lower, np.float32(0), out=lower)
    shape = (len(positions), len(order), projections.shape[2])
    result = np.zeros(shape, dtype=np.float32)
    for shift, weight in ((0, 1 - fraction), (1, fraction)):
        rows = indices + shift
        valid = (rows >= 0) & (rows < nv) & (weight != 0)
        if np.any(valid):
            selected = projections[order[None, :], rows[valid, None], :]
            result[valid] += np.asarray(selected, np.float32) * weight[valid, None, None]
    return result


def gather_per_view_rows(
    projections: np.ndarray | jax.Array, order: np.ndarray, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Copy at most two contributing rows per slice/view, with zero extension."""
    nv = projections.shape[1]
    positions = np.clip(positions, -1, nv)
    lower = np.floor(positions).astype(np.int64)
    fraction = np.asarray(positions - lower, dtype=np.float32)
    rows = np.concatenate((lower, lower + 1))
    valid = (rows >= 0) & (rows < nv)
    # A neighbor whose FP32 weight is zero does not contribute and is not read.
    valid[: len(lower)] &= fraction != 1
    valid[len(lower) :] &= fraction != 0
    samples = np.require(
        projections[order[None, :], np.clip(rows, 0, nv - 1), :], np.float32, requirements=["W"]
    )
    np.copyto(samples, 0, where=~valid[..., None])
    return samples, fraction


def interpolate_numpy(
    spectrum: np.ndarray,
    angle: np.ndarray,
    radial: np.ndarray,
    phase: np.ndarray,
    detector_phase: np.ndarray,
    flipped: np.ndarray,
    table: np.ndarray,
    nfft: int,
    period_phase: complex,
    groups: np.ndarray | None = None,
    shifts: np.ndarray | None = None,
    *,
    replicas: bool = False,
) -> np.ndarray:
    """Vectorized reference for linear angular and Kaiser radial interpolation.

    ``groups`` are the first source view of each distinct direction (and the view
    count); the views of one direction share its angular cell.
    """
    angle, radial, phase, detector_phase = np.broadcast_arrays(angle, radial, phase, detector_phase)
    inside = radial <= nfft / 2
    if replicas or np.all(inside):
        return _interpolate_inside(
            spectrum,
            angle,
            radial,
            phase,
            detector_phase,
            flipped,
            table,
            nfft,
            period_phase,
            groups,
            shifts,
            replicas=replicas,
        )
    result = np.zeros((spectrum.shape[0], *angle.shape), dtype=np.complex128)
    if np.any(inside):
        result[:, inside] = _interpolate_inside(
            spectrum,
            angle[inside],
            radial[inside],
            phase[inside],
            detector_phase[inside],
            flipped,
            table,
            nfft,
            period_phase,
            groups,
            shifts,
            replicas=replicas,
        )
    return result


def _interpolate_inside(
    spectrum: np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
    angle: Float64Array,
    radial: Float64Array,
    phase: np.ndarray,
    detector_phase: np.ndarray,
    flipped: NDArray[np.int8],
    table: np.ndarray,
    nfft: int,
    period_phase: complex,
    groups: NDArray[np.int32] | None = None,
    shifts: np.ndarray | None = None,
    *,
    replicas: bool = False,
) -> np.ndarray:
    """Vectorized reference for linear angular and Kaiser radial interpolation."""
    layers, sources, _ = spectrum.shape
    nviews = sources if groups is None else len(groups) - 1
    ai = np.floor(angle).astype(np.int64)
    lookup = radial
    signed = radial
    if replicas:
        # Centered detector samples acquire a parity phase each whole period.
        periods = np.rint(radial / nfft)
        signed = radial - periods * nfft
        lookup = np.abs(signed)
        detector_phase = detector_phase * np.where(periods % 2, period_phase.real, 1)
    ri = np.floor(lookup).astype(np.int64)
    result = np.zeros((layers, *angle.shape), dtype=np.complex128)
    repeats = 1 if groups is None else int(np.max(np.diff(groups)))
    for da in (0, 1):
        view = (ai + da) % (2 * nviews)
        opposing = view >= nviews
        view %= nviews
        angular = 1 - (angle - ai) if da == 0 else angle - ai
        counts = 1 if groups is None else groups[view + 1] - groups[view]
        for repeat in range(repeats):
            valid = repeat < counts
            if not np.any(valid):
                continue
            # An inclusive scan repeats only one direction: do not repeat the
            # radial work at every Cartesian site for its second endpoint.
            active = Ellipsis if np.all(valid) else valid
            source = view if groups is None else np.minimum(groups[view] + repeat, sources - 1)
            conjugate = opposing != flipped[source].astype(bool)
            row = _radial_values(
                spectrum, source[active], lookup[active], ri[active], table, nfft, period_phase
            )
            if replicas:
                row = np.conjugate(row, out=row, where=signed[active] < 0)
            row *= detector_phase[active]
            if shifts is not None:
                # Shift at the query frequency, AFTER radial interpolation: a
                # binwise phase would also translate its deapodization window.
                row *= np.exp(2j * np.pi * radial[active] * shifts[source[active]] / nfft)
            row = np.conjugate(row, out=row, where=conjugate[active][None, ...])
            result[:, active] += row * (angular / counts)[active]
    result *= phase
    return result


def _radial_values(
    spectrum: np.ndarray,
    source: np.ndarray,
    radial: np.ndarray,
    ri: np.ndarray,
    table: np.ndarray,
    nfft: int,
    period_phase: complex,
) -> np.ndarray:
    """Interpolate before physical phase/orientation and redundant-view averaging."""
    row = np.zeros((spectrum.shape[0], *radial.shape), dtype=np.complex128)
    for dr in range(-2, 4):
        k = ri + dr
        reflected = np.where(k < 0, -k, np.where(k > nfft // 2, nfft - k, k))
        # Advanced indexing owns a copy, even for scalar coordinates.
        values = spectrum[:, source, reflected]
        values = np.conjugate(values, out=values, where=((k < 0) | (k > nfft // 2))[None, ...])
        values = np.multiply(values, period_phase, out=values, where=(k > nfft // 2)[None, ...])
        distance = np.abs(radial - k)
        coordinate = np.minimum(distance * (16384 / 3), 16384)
        index = np.minimum(coordinate.astype(np.int64), 16383)
        fraction = coordinate - index
        weight = table[index] * (1 - fraction) + table[index + 1] * fraction
        values *= np.where(distance < 3, weight, 0)
        row += values
    return row
