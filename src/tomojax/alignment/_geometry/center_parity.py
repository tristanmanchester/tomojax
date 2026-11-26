"""Half-turn consistency of the real and imaginary parts of Fourier slices."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.signal.windows import dpss

type Floats = NDArray[np.float64]
type Matrix = np.ndarray[tuple[int, int], np.dtype[np.float64]]


def _angular_covariance(
    components: Floats, density: Matrix, design: Floats, theta: Floats, parity: int
) -> Floats:
    """Whiten even or odd angular harmonics and profile out detector gain and drift."""
    phase = np.exp(-1j * parity * theta)
    scale = phase / math.sqrt(len(theta))
    observations = np.fft.fft(components * scale[:, None, None], axis=0)
    nuisance = np.fft.fft(design * scale[:, None], axis=0)
    if parity:
        density = (density + np.roll(density, -1, axis=0)) / 2
    covariance: list[Floats] = []
    for column in range(density.shape[1]):
        power: Floats = density[:, column]
        observed: NDArray[np.complex128] = observations[:, column, :]
        # Spectrally empty channels carry no directional information.
        weight = np.divide(1, np.sqrt(power), out=np.zeros_like(power), where=power > 0)
        a, b = nuisance * weight[:, None], observed * weight[:, None]
        x, y = np.vstack((a.real, a.imag)), np.vstack((b.real, b.imag))
        residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
        covariance.append(residual.T @ residual)
    return np.asarray(covariance, np.float64)


def parity_scores(profiles: Matrix, angles: Floats, positions: Floats) -> tuple[Floats, Floats]:
    """Measure the phase ramp that closes a half turn after detector reflection.

    At each radial frequency, a centered Fourier slice has a periodic real part
    and an antiperiodic imaginary part over a half turn. A detector translation
    rotates these components through a known frequency-dependent phase.
    """
    views, columns = profiles.shape
    # Bound spectral work. Opposing rays retain the native detector resolution.
    binning = math.ceil(columns / 128)
    width = columns // binning
    left = (columns % binning) // 2
    binned: Matrix = np.asarray(
        profiles[:, left : left + width * binning].reshape(views, width, binning).mean(2),
        np.float64,
    )
    theta = np.arange(views, dtype=np.float64) * (np.pi / views)
    sample = angles.item(0) + theta
    right = np.searchsorted(angles, sample).clip(1, views - 1)
    fraction = ((sample - angles[right - 1]) / (angles[right] - angles[right - 1]))[:, None]
    profiles = binned[right - 1] + fraction * (binned[right] - binned[right - 1])
    profiles -= profiles[:1].copy()
    time = positions[right - 1] + fraction[:, 0] * (positions[right] - positions[right - 1])
    design = np.column_stack((np.ones(views), time - time.mean()))
    frequency = np.fft.rfftfreq(2 * width)[1:-1]
    midpoint = ((columns - binning) / 2 - left) / binning
    slices: NDArray[np.complex128] = np.fft.rfft(profiles, n=2 * width)[:, 1:-1]
    slices *= np.exp(2j * np.pi * midpoint * frequency)

    # Estimate angular power from the interior, without imposing either seam.
    # Three orthogonal tapers reduce periodogram noise without mixing radial bins.
    residual: NDArray[np.complex128] = (
        slices - design @ np.linalg.lstsq(design, slices, rcond=None)[0]
    )
    tapers: Matrix = np.asarray(dpss(views, 2.5, Kmax=3), np.float64)
    tapered = np.fft.fft(tapers[:, :, None] * residual[None, :, :], axis=1)
    density: Matrix = np.asarray(np.mean(np.abs(tapered) ** 2, axis=0), np.float64)
    density = (density + np.roll(density[::-1], 1, axis=0)) / 2
    components = np.stack((slices.real, slices.imag), axis=-1)
    even, odd = (_angular_covariance(components, density, design, theta, p) for p in (0, 1))
    # Rotate the 2x2 covariance forms analytically. Only the double-angle
    # terms depend on center, so one inverse FFT scores every trial position.
    diff = even - odd
    coefficient = (diff[:, 0, 0] - diff[:, 1, 1]) / 2 - 1j * diff[:, 0, 1]
    # Missing or spurious projection mass violates the zero-order Radon moment.
    # A localized flux error can perturb every complex radial coefficient.
    # Include that systematic variance in the inferred phase precision.
    mass = np.asarray(np.sum(profiles, axis=1, dtype=np.float64), np.float64)
    mass_error = mass - design @ np.linalg.lstsq(design, mass, rcond=None)[0]
    variance: Floats = np.asarray(np.mean(abs(residual) ** 2, axis=0), np.float64)
    denominator = variance + abs(coefficient) * np.mean(mass_error**2, dtype=np.float64).item()
    coefficient *= np.divide(
        variance, denominator, out=np.zeros_like(variance), where=denominator > 0
    )
    count = 16 * width
    cost = np.fft.irfft(np.concatenate((np.zeros(1, np.complex128), coefficient)), n=count) * (
        count / 2
    )
    excess = np.maximum(
        0, np.fft.fftshift(cost) + np.sum(np.abs(coefficient), dtype=np.float64).item()
    )
    bins = len(frequency)
    centers = (np.arange(count, dtype=np.float64) - count // 2) * (binning / 16)
    return centers, -0.5 * bins * np.log1p(excess / bins)
