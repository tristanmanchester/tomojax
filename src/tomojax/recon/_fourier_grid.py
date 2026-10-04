"""Physical grids and interpolation coefficients for Fourier reconstruction."""

from __future__ import annotations

import functools
import math

import numpy as np
from scipy.fft import next_fast_len

from tomojax.geometry import Detector, Grid, grid_volume_origin


def uniform_half_turn(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Sort a uniform half turn, retaining the orientation of opposing views."""
    if angles.ndim != 1 or angles.size < 2 or not np.isfinite(angles).all():
        raise ValueError("fourier_reconstruct: requires at least two finite angles")
    reduced = np.mod(angles.astype(np.float64), 180.0)
    order = np.argsort(reduced)
    sorted_angles = reduced[order]
    gaps = np.diff(np.append(sorted_angles, sorted_angles[0] + 180.0))
    if not np.allclose(gaps, 180.0 / len(angles), rtol=1e-5, atol=2e-5):
        raise ValueError("fourier_reconstruct: angles must cover one uniform half turn")
    flipped = np.mod(angles[order], 360.0) >= 180.0
    return order, flipped.astype(np.int8), float(np.deg2rad(sorted_angles[0]))


def transform_grid(grid: Grid, detector: Detector) -> tuple[tuple[int, int], tuple[int, int]]:
    """Pad around both the requested ROI and the zero-extended detector field."""
    radius = abs(detector.det_center[0]) + detector.nu * detector.du / 2
    origin = grid_volume_origin(grid)
    shape, crop = [], []
    for count, spacing, first in zip(
        (grid.nx, grid.ny), (grid.vx, grid.vy), origin[:2], strict=True
    ):
        lo = min(0, math.floor((-radius - first) / spacing))
        hi = max(count - 1, math.ceil((radius - first) / spacing))
        support = hi - lo + 1
        padded = next_fast_len(2 * support, real=True)
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
) -> np.ndarray:
    """Vectorized reference for linear angular and Kaiser radial interpolation."""
    layers, nviews, _ = spectrum.shape
    ai, ri = np.floor(angle).astype(np.int64), np.floor(radial).astype(np.int64)
    inside = radial <= nfft / 2
    result = np.zeros((layers, *angle.shape), dtype=np.complex128)
    for da in (0, 1):
        view = (ai + da) % (2 * nviews)
        conjugate = (view >= nviews) != flipped[view % nviews].astype(bool)
        view %= nviews
        angular = 1 - (angle - ai) if da == 0 else angle - ai
        row = np.zeros_like(result)
        for dr in range(-2, 4):
            k = np.where(inside, ri + dr, 0)
            reflected = np.where(k < 0, -k, np.where(k > nfft // 2, nfft - k, k))
            values = spectrum[:, view, reflected]
            values = np.where((k < 0) | (k > nfft // 2), values.conj(), values)
            values = np.where(k > nfft // 2, values * period_phase, values)
            distance = np.abs(radial - k)
            coordinate = np.minimum(distance * (16384 / 3), 16384)
            index = np.minimum(coordinate.astype(np.int64), 16383)
            fraction = coordinate - index
            weight = table[index] * (1 - fraction) + table[index + 1] * fraction
            row += values * np.where(inside & (distance < 3), weight, 0)
        row *= detector_phase
        row = np.where(conjugate, row.conj(), row)
        result += row * angular
    return result * phase
