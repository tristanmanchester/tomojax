"""Physical grids and interpolation coefficients for Fourier reconstruction."""

from __future__ import annotations

import functools
import math

import numpy as np

from tomojax.geometry import Detector, Grid, grid_volume_origin

from ._directions import group_repeats

_INTERPOLATION_WORKSPACE_BYTES = 8 * 1024**2


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
    # Adding a half turn and reducing it can perturb the same physical
    # direction by a few FP64 ulps. Exact uniqueness would give redundant
    # exposures separate, almost-zero angular cells instead of averaging them.
    tolerance = np.deg2rad(
        8 * np.finfo(np.float64).eps * max(180.0, float(np.max(np.abs(degrees))))
    )
    representatives, inverse, counts = group_repeats(
        2 * np.deg2rad(degrees), tolerance=2 * tolerance
    )
    if representatives.size < 2:
        raise ValueError("fourier_reconstruct: requires at least two distinct half-turn directions")
    reduced = representatives[inverse] / 2
    unique = counts[inverse] == 1
    reduced[unique] = np.deg2rad(np.mod(degrees[unique], 180.0))
    reduced[np.minimum(reduced, np.pi - reduced) <= tolerance] = 0
    order = np.argsort(reduced, kind="stable")
    flipped = np.cos(np.deg2rad(degrees[order]) - reduced[order]) < 0
    return order, flipped.astype(np.int8), tuple(reduced[order])


def transform_grid(grid: Grid, detector: Detector) -> tuple[tuple[int, int], tuple[int, int]]:
    """Pad around both the requested ROI and the zero-extended detector field."""
    radius = abs(detector.center[0]) + detector.nu * detector.du / 2
    origin = grid_volume_origin(grid)
    shape, crop = [], []
    for count, spacing, first in zip(
        (grid.nx, grid.ny), (grid.vx, grid.vy), origin[:2], strict=True
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
        array.setflags(write=False)
    return result


def sample_projection_rows(
    projections: np.ndarray,
    order: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Interpolate physical detector-v rows with zero extension and host storage."""
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
        lower = np.asarray(projections[order[None, :], indices[:, None], :], np.float32)
        upper = np.asarray(projections[order[None, :], (indices + 1)[:, None], :], np.float32)
        np.multiply(lower, (1 - fraction)[:, None, None], out=lower)
        np.multiply(upper, fraction[:, None, None], out=upper)
        np.add(lower, upper, out=lower)
        # Match the original zero-initialized accumulation's signed zeros.
        np.add(lower, np.float32(0), out=lower)
        return lower
    shape = (len(positions), len(order), projections.shape[2])
    result = np.zeros(shape, dtype=np.float32)
    for shift, weight in ((0, 1 - fraction), (1, fraction)):
        rows = indices + shift
        valid = (rows >= 0) & (rows < nv) & (weight != 0)
        if np.any(valid):
            selected = projections[order[None, :], rows[valid, None], :]
            result[valid] += np.asarray(selected, np.float32) * weight[valid, None, None]
    return result


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
) -> np.ndarray:
    """Bound point-wise temporaries without changing interpolation or slab precision."""
    angle, radial, phase, detector_phase = np.broadcast_arrays(angle, radial, phase, detector_phase)
    coordinates = (angle, radial, phase, detector_phase)
    # Scalar/index fields and four complex slab arrays dominate the working set.
    # More simultaneous slices therefore require fewer Cartesian sites per tile.
    points = max(1, _INTERPOLATION_WORKSPACE_BYTES // (256 + 64 * spectrum.shape[0]))
    if coordinates[0].size <= points:
        return _interpolate_block(
            spectrum, *coordinates, flipped, table, nfft, period_phase, groups
        )
    result = np.empty((spectrum.shape[0], coordinates[0].size), dtype=np.complex128)
    for start in range(0, coordinates[0].size, points):
        stop = min(start + points, coordinates[0].size)
        # flat slices copy only this window, even for broadcast/strided fields.
        tile_angle, tile_radial, tile_phase, tile_detector = (
            coordinate.flat[start:stop] for coordinate in coordinates
        )
        result[:, start:stop] = _interpolate_block(
            spectrum,
            tile_angle,
            tile_radial,
            tile_phase,
            tile_detector,
            flipped,
            table,
            nfft,
            period_phase,
            groups,
        )
    return result.reshape((spectrum.shape[0], *coordinates[0].shape))


def _interpolate_block(
    spectrum: np.ndarray,
    angle: np.ndarray,
    radial: np.ndarray,
    phase: np.ndarray,
    detector_phase: np.ndarray,
    flipped: np.ndarray,
    table: np.ndarray,
    nfft: int,
    period_phase: complex,
    groups: np.ndarray | None,
) -> np.ndarray:
    """Literal interpolation for broadcast coordinates, with detector-Nyquist cutoff."""
    inside = radial <= nfft / 2
    if np.all(inside):
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
        )
    return result


def _interpolate_inside(
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
) -> np.ndarray:
    """Vectorized reference for linear angular and Kaiser radial interpolation."""
    layers, sources, _ = spectrum.shape
    nviews = sources if groups is None else len(groups) - 1
    ai, ri = np.floor(angle).astype(np.int64), np.floor(radial).astype(np.int64)
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
                spectrum, source[active], radial[active], ri[active], table, nfft, period_phase
            )
            row *= detector_phase[active]
            np.conjugate(row, out=row, where=conjugate[active][None, ...])
            result[:, active] += row * (angular / counts)[active]
    result *= phase
    return result


def _radial_values(spectrum, source, radial, ri, table, nfft, period_phase):
    """Interpolate before physical phase/orientation and redundant-view averaging."""
    row = np.zeros((spectrum.shape[0], *radial.shape), dtype=np.complex128)
    for dr in range(-2, 4):
        k = ri + dr
        reflected = np.where(k < 0, -k, np.where(k > nfft // 2, nfft - k, k))
        # Advanced indexing owns a copy, even for scalar coordinates.
        values = spectrum[:, source, reflected]
        np.conjugate(values, out=values, where=((k < 0) | (k > nfft // 2))[None, ...])
        np.multiply(values, period_phase, out=values, where=(k > nfft // 2)[None, ...])
        distance = np.abs(radial - k)
        coordinate = np.minimum(distance * (16384 / 3), 16384)
        index = np.minimum(coordinate.astype(np.int64), 16383)
        fraction = coordinate - index
        weight = table[index] * (1 - fraction) + table[index + 1] * fraction
        values *= np.where(distance < 3, weight, 0)
        row += values
    return row
