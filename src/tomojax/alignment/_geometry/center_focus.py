"""Axial center calibration from half-turn projection moments and Fourier slices."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import numpy as np

from tomojax import progress
from tomojax._typed_arrays import numpy_float64_array
from tomojax.geometry import ParallelGeometry

from .center_parity import parity_scores

if TYPE_CHECKING:
    import jax
    from numpy.typing import NDArray

    from tomojax.geometry import ScanGeometry


type FloatImage = np.ndarray[tuple[int, int], np.dtype[np.float64]]
type FocusScores = tuple[NDArray[np.float64], NDArray[np.float64]]


@dataclass(frozen=True)
class AngularFocus:
    """A half-turn sampled at the angular resolution of a bounded Fourier slice."""

    views: NDArray[np.intp]
    angles: NDArray[np.float64]
    binning: int

    def scores(
        self, projections: np.ndarray | jax.Array, positions: np.ndarray | None
    ) -> FocusScores:
        """Score detector centers in native pixels by angular parity and edge sparsity."""
        nv, nu = map(int, np.shape(projections)[1:])
        rows = np.linspace(0, nv - 1, min(16, nv)).round().astype(int)
        profiles: list[NDArray[np.float64]] = []
        means: list[NDArray[np.float64]] = []
        for ids in np.array_split(self.views, math.ceil(len(self.views) / 16)):
            progress.check_canceled()
            slab = np.asarray(projections[ids[:, None], rows[None, :], :], np.float64)
            slab = np.where(np.isfinite(slab), slab, np.nan)
            means.append(np.nan_to_num(np.asarray(np.nanmean(slab, axis=1), np.float64)))
            profiles.append(np.asarray(np.nanmedian(slab, axis=1), np.float64))
        y: FloatImage = np.concatenate(profiles)
        # Project out view pedestal, stationary gain and linear flat-field drift.
        y -= np.nanmedian(y, axis=1, keepdims=True)
        sampled = self.views if positions is None else positions[self.views]
        t = np.asarray(sampled, np.float64) - np.mean(sampled, dtype=np.float64)
        y = np.where(np.isfinite(y), y, np.nan)
        y -= np.asarray(np.nanmean(y, axis=0), np.float64)
        y = np.nan_to_num(y)
        drift = np.linalg.lstsq(t[:, None], y, rcond=None)[0]
        y -= t[:, None] * drift
        b = self.binning
        n = nu // b
        left = (nu - n * b) // 2
        y = np.asarray(y[:, left : left + n * b].reshape(len(y), n, b).mean(axis=2), np.float64)
        spectrum, radius = _polar_grid(y, self.angles)
        # Off-axis rays can reach one full detector width from the axis.
        # Compare all candidates on that support without periodic image wrap.
        x = np.arange(2 * n) - n
        disk = x[:, None] ** 2 + x**2 < (n - 1) ** 2
        costs: list[float] = []
        trials = np.arange(-(n // 2), (n + 1) // 2)
        for shift in trials:
            progress.check_canceled()
            image = np.fft.fftshift(
                np.fft.ifft2(spectrum * np.exp(-2j * np.pi * shift * radius))
            ).real
            dy, dx = (np.asarray(component) for component in np.gradient(image))
            mag = np.hypot(dx[disk], dy[disk])
            norm = np.linalg.norm(mag)
            costs.append(float(np.sum(mag, dtype=np.float64) / norm) if norm else 1.0)
        # Cropping extra pixels moves our working detector midpoint.
        centers = trials * b + (nu - n * b) / 2 - left
        # One angular degree of freedom per resolved radial cell. A Fourier slice
        # interpolates neighboring rays, so treating n*n pixels as independent
        # would swamp the native opposing-ray measurements.
        native = np.concatenate(means)
        parity_centers, parity = parity_scores(native, self.angles, t)
        fine = np.arange(-16 * (nu - 1), 16 * (nu - 1) + 1, dtype=np.float64) / 32
        sharpness = np.interp(fine, centers, -2 * n * np.log(costs))
        return fine, sharpness + np.interp(fine, parity_centers, parity) + _moment_scores(
            native, self.angles, t, fine
        )


def _moment_scores(
    profiles: FloatImage,
    angles: NDArray[np.float64],
    time: NDArray[np.float64],
    centers: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Fit the first Radon moment, propagating detector background and missing flux."""
    width = profiles.shape[1]
    u = np.arange(width, dtype=np.float64) - (width - 1) / 2
    edges = np.asarray(profiles[:, [0, -1]], np.float64)
    # A uniform view pedestal carries no position information. Use the two
    # boundary rays as its reference; their signal also sets the uncertainty.
    pedestal = np.asarray(np.mean(edges, axis=1, keepdims=True), np.float64)
    mass = np.sum(profiles - pedestal, axis=1, dtype=np.float64)
    # At the axis, the first moment follows a sinusoid. Allow a detector drift
    # linear in acquisition time in addition to that specimen trajectory.
    design = np.column_stack((np.cos(angles), np.sin(angles), time))
    moments = np.column_stack((mass, profiles @ u))
    detrended = np.asarray(
        moments - design @ np.linalg.lstsq(design, moments, rcond=None)[0], np.float64
    )
    flux, first = detrended[:, 0], detrended[:, 1]
    norm = float(np.linalg.norm(flux))
    if not norm:
        return np.zeros_like(centers)
    direction = flux / norm
    offset = float(direction @ first)
    residual = first - offset * direction
    variance = np.mean(residual**2, dtype=np.float64).item()
    # A stationary column error biases all the views together. Estimate its
    # scale at the detector edges. Correlated changes in total mass can also
    # come from clipping; that missing mass has at least the detector's radius.
    fixed = np.asarray(np.mean(edges, axis=0), np.float64)
    background = np.mean(fixed**2, dtype=np.float64).item() * float(u @ u)
    boundary = np.mean(abs(edges), dtype=np.float64).item() * width
    drift = np.column_stack((np.ones(len(time)), time))
    missing = mass - drift @ np.linalg.lstsq(drift, mass, rcond=None)[0]
    lost = max(0.0, np.mean(missing[1:] * missing[:-1], dtype=np.float64).item())
    systematic = background + boundary**2 + lost * ((width - 1) / 2) ** 2
    variance += np.sum(direction, dtype=np.float64).item() ** 2 * systematic
    variance += (norm * (centers.item(1) - centers.item(0))) ** 2 / 12
    return -np.log1p((norm * centers + offset) ** 2 / variance)


def _polar_grid(
    y: FloatImage, theta: NDArray[np.float64]
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """Interpolate the Fourier slices to a Cartesian grid; retain signed radial frequency."""
    n = y.shape[1]
    pad = 2 * n
    ft = np.fft.rfft(y, n=pad)
    ft *= np.exp(2j * np.pi * np.fft.rfftfreq(pad) * (n - 1) / 2)
    frequency = np.fft.fftfreq(pad)
    ky, kx = frequency[:, None], frequency[None, :]
    first, last = theta.item(0), theta.item(-1)
    angle = (np.arctan2(ky, kx) - first) % (2 * np.pi)
    alpha = first + angle % np.pi
    radius = np.hypot(kx, ky)
    i = np.interp(alpha, theta, np.arange(len(theta)))
    j = np.minimum(radius * pad, n)
    lo, col = i.astype(int), j.astype(int)
    hi, nxt = np.minimum(lo + 1, len(theta) - 1), np.minimum(col + 1, n)
    wi, wj = i - lo, j - col
    z: NDArray[np.complex128] = (1 - wi) * ((1 - wj) * ft[lo, col] + wj * ft[lo, nxt])
    z += wi * ((1 - wj) * ft[hi, col] + wj * ft[hi, nxt])
    # Taper the half-turn seam, where an incorrect trial center has a phase jump.
    taper = numpy_float64_array(
        np.clip(np.minimum(alpha - first, last - alpha) / math.radians(5), 0, 1)
    )
    z = np.asarray(z * taper * (radius > 0) * (radius < 0.42), np.complex128)
    return np.where(angle >= np.pi, z.conj(), z), np.where(angle >= np.pi, -radius, radius)


def angular_focus(geometry: ScanGeometry) -> AngularFocus | None:
    """Select a resolved angular half-turn on an unposed axial detector."""
    if type(geometry) is not ParallelGeometry or geometry.detector_roll:
        return None
    angles, order = np.unique(np.deg2rad(geometry.angles) % (2 * np.pi), return_index=True)
    if len(angles) < 3:
        return None
    gaps = np.diff(angles, append=angles[0] + 2 * np.pi)
    start = (int(np.argmax(gaps)) + 1) % len(angles)
    arc = np.r_[angles[start:], angles[:start] + 2 * np.pi]
    order = np.roll(order, -start)
    target = arc[0] + np.linspace(0, np.pi, 401)
    right = np.searchsorted(arc, target).clip(1, len(arc) - 1)
    indices = np.unique(np.where(target - arc[right - 1] < arc[right] - target, right - 1, right))
    theta = arc[indices]
    gap = max(float(np.max(np.diff(theta))), float(target[-1] - theta[-1]))
    # Angular Nyquist for a disk: its outermost ray moves at most one cell per view.
    width = min(geometry.detector.nu, 256, int(2 / gap))
    if len(theta) < 6 or width < max(2, min(32, geometry.detector.nu)):
        return None
    return AngularFocus(order[indices], theta, max(1, math.ceil(geometry.detector.nu / width)))
