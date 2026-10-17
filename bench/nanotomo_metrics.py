"""Independent half-data resolution and reference-shift checks for nano CT.

No alignment objective is used as an accuracy metric. FSC compares two volumes
on one physical grid; phase registration compares measured views, not a fitted
reprojection. These helpers do not import JAX or require a GPU.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import fft, optimize
from scipy.signal.windows import tukey

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomojax import Scan
    from tomojax.recon.api import MethodConfig


def interleaved_views(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Alternate distinct angle groups, keeping all repeats in the same half.

    Sorting before splitting matters for interleaved sub-tomograms. Repeated
    angles can be copied observations, not independent exposures: splitting
    those copies across halves falsely inflates FSC. Conservatively keep each
    exact-angle group together even when its exposures really are independent.
    Return indices in acquisition order for ``scan.selected(indices)``.
    """
    angles = np.asarray(angles).ravel()
    order = np.argsort(angles, kind="stable")
    group = np.cumsum(np.r_[True, np.diff(angles[order]) != 0]) - 1
    return np.sort(order[group % 2 == 0]), np.sort(order[group % 2 == 1])


def apodized(volume: np.ndarray, *, taper: float) -> np.ndarray:
    """Remove the mean and apply the same separable Tukey window to both halves."""
    result = np.asarray(volume, dtype=np.float32).copy()
    result -= np.mean(result, dtype=np.float64)
    for axis, size in enumerate(result.shape):
        shape = [1] * result.ndim
        shape[axis] = size
        result *= tukey(size, taper).astype(np.float32).reshape(shape)
    return result


def threshold_crossing(
    frequency: np.ndarray, correlation: np.ndarray, threshold: np.ndarray, *, nyquist: float
) -> dict[str, Any]:
    """First downward half-bit crossing, linearly interpolated; never a later recrossing.

    DC is excluded. A curve below threshold at its first measured frequency
    is unresolved; a curve above throughout is Nyquist-limited. Both cases are
    explicitly labelled instead of claiming an observed crossing.
    """
    valid = (frequency > 0) & np.isfinite(correlation)
    frequencies = frequency[valid]
    difference = (correlation - threshold)[valid]
    below = np.flatnonzero(difference < 0)
    if not len(frequencies):
        return {"status": "no_signal", "frequency": None, "half_period": None, "full_period": None}
    if len(below) and below[0] == 0:
        return {
            "status": "below_band",
            "frequency": None,
            "half_period": None,
            "full_period": None,
        }
    if not len(below):
        crossing, status = nyquist, "nyquist_limited"
    else:
        index = below[0]
        lo, hi = frequencies[index - 1 : index + 1]
        a, b = difference[index - 1 : index + 1]
        crossing, status = lo + (hi - lo) * a / (a - b), "crossing"
    return {
        "status": status,
        "frequency": float(crossing),
        "half_period": float(0.5 / crossing),
        "full_period": float(1 / crossing),
    }


def fourier_shell_correlation(
    first: np.ndarray,
    second: np.ndarray,
    *,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    taper: float = 0.1,
    shell_width: int = 1,
) -> dict[str, Any]:
    """Signed FSC and the van Heel--Schatz half-bit threshold on a physical grid.

    ``spacing`` has the volumes' axis order, and the resolution uses those units.
    Shell width is in units of the coarsest Fourier-grid spacing; only the common
    Nyquist sphere is used. Real-FFT multiplicities reproduce full-shell voxel
    counts, including odd dimensions and DC/Nyquist planes. The half-bit curve is
    ``(s + (2*sqrt(s)+1)/sqrt(n)) / (1+s+2*sqrt(s)/sqrt(n))``, with
    ``s=(sqrt(2)-1)/2`` and full-shell counts ``n``. It is not a fixed 0.143 cut.

    A fixed common ROI/window is essential: data-dependent tight masks inflate
    correlation. Tapering itself couples Fourier samples, so this conventional
    count-based threshold is a resolution diagnostic, not a confidence interval.
    Half-period resolution matches the PSI nanotomography convention. Full
    period is returned as well to avoid a factor-of-two ambiguity.
    """
    if first.shape != second.shape or first.ndim != 3:
        raise ValueError("FSC requires two matching three-dimensional volumes")
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("spacing must contain three positive physical voxel lengths")
    if not 0 <= taper <= 1 or shell_width < 1:
        raise ValueError("taper must be in [0, 1] and shell_width positive")
    shape = first.shape
    df = float(shell_width * np.max(1 / (np.asarray(shape) * spacing)))
    nyquist = float(np.min(0.5 / spacing))
    count = int(np.floor(nyquist / df)) + 1
    fa = fft.rfftn(apodized(first, taper=taper), workers=-1)
    fb = fft.rfftn(apodized(second, taper=taper), workers=-1)
    fx, fy = (fft.fftfreq(shape[i], spacing[i]) for i in range(2))
    fz = fft.rfftfreq(shape[2], spacing[2])
    radius2 = fy[:, None] ** 2 + fz[None, :] ** 2
    multiplicity = np.full(len(fz), 2.0)
    multiplicity[0] = 1
    if shape[2] % 2 == 0:
        multiplicity[-1] = 1
    weight = np.broadcast_to(multiplicity, radius2.shape)
    totals = np.zeros((4, count), dtype=np.float64)
    for index, x in enumerate(fx):
        radius = np.sqrt(x * x + radius2)
        shell = np.floor(radius / df + 0.5).astype(np.int32)
        keep = (radius <= nyquist) & (shell < count)
        a, b = fa[index], fb[index]
        values = (np.ones_like(radius), (a * b.conj()).real, np.abs(a) ** 2, np.abs(b) ** 2)
        for total, value in zip(totals, values, strict=True):
            total[:] += np.bincount(shell[keep], weights=(value * weight)[keep], minlength=count)
    samples, cross, power_a, power_b = totals
    correlation = np.full(count, np.nan)
    np.divide(cross, np.sqrt(power_a * power_b), out=correlation, where=(power_a * power_b) > 0)
    s = (np.sqrt(2) - 1) / 2
    threshold = (s + (2 * np.sqrt(s) + 1) / np.sqrt(samples)) / (
        1 + s + 2 * np.sqrt(s) / np.sqrt(samples)
    )
    frequency = np.arange(count) * df
    return {
        "frequency": frequency,
        "correlation": correlation,
        "half_bit": threshold,
        "samples": samples.astype(np.int64),
        "resolution": threshold_crossing(frequency, correlation, threshold, nyquist=nyquist),
        "spacing": spacing.tolist(),
        "taper": taper,
        "shell_width": shell_width,
    }


def _correlation_shift(
    reference: np.ndarray, moving: np.ndarray, *, taper: float = 0.1
) -> tuple[np.ndarray, float]:
    """Return the subpixel ``(v, u)`` shift applied to moving to match reference.

    Remove a nuisance plane from each image, then maximize signed unnormalized
    Fourier correlation. Continuous local refinement uses the Fourier shift
    theorem, not a quadratic approximation to the sampled peak. The returned
    normalized peak is diagnostic: a shift alone cannot establish matching data.
    Inputs must cover the same region; crop uncertain borders before calling.
    """
    if reference.shape != moving.shape or reference.ndim != 2:
        raise ValueError("registration requires matching two-dimensional images")
    shape = reference.shape
    axes = [np.arange(size, dtype=np.float64) - (size - 1) / 2 for size in shape]

    def spectrum(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32).copy()
        image -= np.mean(image, dtype=np.float64)
        for axis, coordinate in enumerate(axes):
            slope = np.dot(np.mean(image, axis=1 - axis), coordinate) / np.dot(
                coordinate, coordinate
            )
            view = coordinate[:, None] if axis == 0 else coordinate[None, :]
            image -= slope * view
        return fft.fft2(apodized(image, taper=taper))

    a, b = spectrum(reference), spectrum(moving)
    normalizer = np.sqrt(
        np.sum(np.abs(a) ** 2, dtype=np.float64) * np.sum(np.abs(b) ** 2, dtype=np.float64)
    )
    cross = (a * b.conj()).astype(np.complex128) / normalizer
    peak = np.unravel_index(np.argmax(fft.ifft2(cross).real), shape)
    shift = np.asarray(peak, dtype=float)
    shift -= (shift >= (np.asarray(shape) + 1) // 2) * np.asarray(shape)
    frequencies = [2j * np.pi * fft.fftfreq(size) for size in shape]

    def objective(position: np.ndarray) -> tuple[float, np.ndarray]:
        vertical, horizontal = (np.exp(f * p) for f, p in zip(frequencies, position, strict=True))
        along_u = cross @ horizontal
        value = vertical @ along_u
        gradient = np.array(
            [
                (vertical * frequencies[0]) @ along_u,
                vertical @ (cross @ (horizontal * frequencies[1])),
            ]
        )
        return -float(value.real), -gradient.real

    fitted = optimize.minimize(
        objective,
        shift,
        jac=True,
        method="L-BFGS-B",
        bounds=list(zip(shift - 1, shift + 1, strict=True)),
        options={"ftol": 1e-12, "gtol": 1e-8},
    )
    return fitted.x, -float(fitted.fun)


def register_translation(
    reference: np.ndarray, moving: np.ndarray, *, taper: float = 0.1
) -> tuple[np.ndarray, float]:
    """Subpixel ``(v,u)`` shift to apply to moving, and normalized plane-free match score.

    Initially find the global Fourier correlation peak, removing an independent
    phase plane from each view. Then fit the original moving image over
    only the common non-wrapped region, eliminating phase planes by least squares.
    Fixed windows and nuisance
    plane removal are not translation equivariant: one global tapered peak can
    be biased by tenths of a pixel even on an exact translated image.
    A fixed overlap loss avoids moving either window with the fitted shift. Border
    pixels known to be unreliable should still be cropped before this call.
    """
    shift, _ = _correlation_shift(reference, moving, taper=taper)
    radius = max(2, min(moving.shape) // 8)
    lower = np.ceil(np.maximum(shift + radius, 0)).astype(int) + 2
    upper = np.floor(np.asarray(moving.shape) + np.minimum(shift - radius, 0)).astype(int) - 2
    overlap = tuple(slice(lo, hi) for lo, hi in zip(lower, upper, strict=True))
    target = np.asarray(reference[overlap], dtype=np.float64)
    v, u = np.meshgrid(*(np.linspace(-1, 1, n) for n in target.shape), indexing="ij")
    basis = np.stack([np.ones_like(v), v, u], axis=-1).reshape(-1, 3)
    weight = np.outer(*(tukey(n, taper) for n in target.shape)).ravel()
    gram = basis.T @ (weight[:, None] * basis)

    def without_plane(image: np.ndarray) -> np.ndarray:
        values = image.ravel()
        plane = np.linalg.solve(gram, basis.T @ (weight * values))
        return values - basis @ plane

    target = without_plane(target)
    normalization = np.dot(weight * target, target)
    spectrum = fft.fftn(moving)
    fv, fu = (fft.fftfreq(size) for size in moving.shape)
    derivatives = (-2j * np.pi * fv[:, None], -2j * np.pi * fu[None, :])

    def objective(position: np.ndarray) -> tuple[float, np.ndarray]:
        phase = np.exp(-2j * np.pi * (fv[:, None] * position[0] + fu[None, :] * position[1]))
        shifted = spectrum * phase
        residual = without_plane(fft.ifftn(shifted).real[overlap]) - target
        weighted = weight * residual
        gradient = np.array(
            [
                np.dot(weighted, without_plane(fft.ifftn(shifted * derivative).real[overlap]))
                for derivative in derivatives
            ]
        )
        return float(np.dot(weighted, residual) / normalization), 2 * gradient / normalization

    fitted = optimize.minimize(
        objective,
        shift,
        jac=True,
        method="L-BFGS-B",
        bounds=list(zip(shift - radius, shift + radius, strict=True)),
        options={"ftol": 1e-13, "gtol": 1e-9},
    )
    # Unit peak means exact equality after the independent nuisance planes,
    # not merely a high correlation of bright smooth backgrounds.
    return fitted.x, float(1 - fitted.fun)


def shift_error(
    actual: np.ndarray,
    reference: np.ndarray,
    rotations: np.ndarray,
    *,
    pixel_size: Sequence[float] = (1.0, 1.0),
) -> dict[str, Any]:
    """Score detector ``(u,v)`` displacements after only one object translation.

    ``rotations`` are world-from-object rotation matrices. Their world x/z rows
    project a common object translation into detector u/v. This removes three
    global gauge freedoms, not independent offsets or sinusoids for each axis.
    ``pixel_size`` is detector ``(du,dv)`` in the rotations' physical length unit.
    """
    difference = np.asarray(actual) - np.asarray(reference)
    design = np.asarray(rotations)[:, [0, 2], :] / np.asarray(pixel_size)[None, :, None]
    translation = np.linalg.lstsq(design.reshape(-1, 3), difference.ravel(), rcond=None)[0]
    residual = difference - design @ translation
    error = np.linalg.norm(residual, axis=1)
    return {
        "translation": translation.tolist(),
        "residual_uv": residual,
        "vector_rmse_px": float(np.sqrt(np.mean(error**2))),
        "median_px": float(np.median(error)),
        "p95_px": float(np.quantile(error, 0.95)),
        "max_px": float(np.max(error)),
    }


def half_dataset_fsc(
    scan: Scan,
    *,
    method: str = "fbp",
    config: MethodConfig | None = None,
    region: tuple[slice, slice, slice] | None = None,
    taper: float = 0.1,
    shell_width: int = 1,
) -> dict[str, Any]:
    """Reconstruct alternating angle-group halves and score on one common grid.

    Per-view poses are preserved. Both candidates must use identical ``method``,
    ``config``, ``region`` and grid; no image registration, clipping or fitted
    amplitude is applied to improve FSC. A region is in volume ``(x,y,z)`` axes
    and must use unit strides. Full volumes are discarded after scoring; only
    scalar timings, split indices and FSC arrays are returned. Timing includes
    both reconstructions, device-to-host transfers and the Fourier diagnostic,
    but not the preceding phase preparation or alignment. The grid shape and
    exact scoring region are recorded so guarded and full-volume results cannot
    be mistaken for the same protocol.
    """
    import tomojax as tj

    if region is not None and any(part.step not in (None, 1) for part in region):
        raise ValueError("FSC regions must use unit strides to preserve physical voxel spacing")
    start = time.perf_counter()
    halves = interleaved_views(scan.angles)
    volumes, times = [], []
    for indices in halves:
        before = time.perf_counter()
        volume = np.asarray(tj.reconstruct(scan.selected(indices), method, config=config).volume)
        volumes.append(volume if region is None else volume[region])
        times.append(time.perf_counter() - before)
    grid = scan.grid
    result = fourier_shell_correlation(
        *volumes, spacing=(grid.vx, grid.vy, grid.vz), taper=taper, shell_width=shell_width
    )
    return {
        **result,
        "halves": [indices.tolist() for indices in halves],
        "method": method,
        "grid_shape": [grid.nx, grid.ny, grid.nz],
        "region": None if region is None else [[s.start, s.stop, s.step] for s in region],
        "scored_shape": list(volumes[0].shape),
        "reconstruction_seconds": times,
        "whole_seconds": time.perf_counter() - start,
    }
