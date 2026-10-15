"""Bounded acquisition-domain motion from compact Fourier-line observations.

Only 256 views are sampled, in 16-view FFT batches. Sixty-four linear motion
knots prevent a denser scan from introducing more poorly constrained phase
variables. Per-edge correlation tables hide the frequency dimension from the
optimizer; the returned agreement is evaluated on the original complex data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import fft, ndimage, optimize, sparse

from .common_lines import PhaseShiftSeed, angular_pairs, line_coordinates, preview_spectra

if TYPE_CHECKING:
    import jax

    from tomojax.core.geometry.base import Detector


def _observations(projections, angles, detector, axis):
    keep = np.linspace(0, len(angles) - 1, min(256, len(angles))).astype(int)
    theta = np.deg2rad(np.mod(np.asarray(angles)[keep], 360))
    pairs = angular_pairs(theta)
    if len(pairs) < 3 or min(projections.shape[1:]) < 8:
        return None
    spectra, pitch, frequency = preview_spectra(projections[keep[:16]], detector)
    coordinates = line_coordinates(theta, pairs, frequency, axis)
    samples = [np.empty((len(pairs), len(frequency)), np.complex64) for _ in range(2)]
    for first in range(0, len(keep), 16):
        if first:
            spectra, _, _ = preview_spectra(projections[keep[first : first + 16]], detector)
        nv, nu = spectra.shape[1:]
        for side in range(2):
            take = np.flatnonzero(
                (pairs[:, side] >= first) & (pairs[:, side] < first + len(spectra))
            )
            q = coordinates[side][take]
            u, v = q[..., 0] * pitch[0] * nu + nu // 2, q[..., 1] * pitch[1] * nv + nv // 2
            points = [np.broadcast_to((pairs[take, side] - first)[:, None], u.shape), v, u]
            samples[side][take] = ndimage.map_coordinates(spectra, points, order=1, mode="constant")
    return keep, pairs, frequency, np.asarray(pitch), coordinates, samples


def _motion_basis(n):
    knots = min(64, n)
    where = np.linspace(0, n - 1, knots)
    positions = np.arange(n)
    left = np.clip(np.searchsorted(where, positions, side="right") - 1, 0, knots - 2)
    weight = (positions - where[left]) / (where[left + 1] - where[left])
    basis = sparse.coo_matrix(
        (
            np.stack([1 - weight, weight], axis=1).ravel(),
            (np.repeat(positions, 2), np.stack([left, left + 1], axis=1).ravel()),
        ),
        shape=(n, knots),
    )
    basis = sparse.kron(basis, sparse.eye(2), format="csr")
    rows = np.repeat(np.arange(knots - 2), 3)
    cols = np.stack([np.arange(knots - 2) + offset for offset in range(3)], axis=1).ravel()
    difference = sparse.coo_matrix(
        (np.tile([1.0, -2.0, 1.0], knots - 2), (rows, cols)), shape=(knots - 2, knots)
    )
    difference = sparse.kron(difference, sparse.eye(2), format="csr")
    regularizer = 0.02 / (2 * (knots - 2)) * (difference.T @ difference)
    regularizer += sparse.eye(2 * knots) * (1e-5 / knots)
    return basis, regularizer


def _correlation_table(cross, frequency, coefficients):
    nfft = 2048
    step = 1 / (nfft * (frequency[1] - frequency[0]))
    radius = int(np.ceil(4 * np.max(np.sum(abs(coefficients), axis=1)) / step)) + 2
    offsets = np.arange(-radius, radius + 1)
    table = np.empty((len(cross), len(offsets)))
    derivative = np.empty_like(table)
    phase = np.exp(2j * np.pi * frequency[0] * offsets * step)
    for first in range(0, len(cross), 256):
        values = np.asarray(cross[first : first + 256], np.complex128)
        transformed = np.asarray(fft.ifft(values, n=nfft, axis=-1)) * nfft
        gradient = np.asarray(fft.ifft(values * frequency, n=nfft, axis=-1)) * nfft
        table[first : first + 256] = (transformed[:, offsets % nfft] * phase).real
        derivative[first : first + 256] = -2 * np.pi * (gradient[:, offsets % nfft] * phase).imag
    return table, derivative, step, radius


def _table_values(table, derivative, step, radius, positions):
    cell = np.floor(positions / step).astype(int)
    z = positions / step - cell
    cell += radius
    rows = np.arange(len(positions))
    v0, v1 = table[rows, cell], table[rows, cell + 1]
    g0, g1 = derivative[rows, cell], derivative[rows, cell + 1]
    value = (
        (2 * z**3 - 3 * z**2 + 1) * v0
        + (z**3 - 2 * z**2 + z) * step * g0
        + (-2 * z**3 + 3 * z**2) * v1
        + (z**3 - z**2) * step * g1
    )
    gradient = (
        (6 * z**2 - 6 * z) / step * v0
        + (3 * z**2 - 4 * z + 1) * g0
        + (-6 * z**2 + 6 * z) / step * v1
        + (3 * z**2 - 2 * z) * g1
    )
    return value, gradient


def _fit_knots(cross, frequency, coefficients, columns, initial):
    basis, regularizer = _motion_basis(len(initial))
    table = _correlation_table(cross, frequency, coefficients)
    start = np.linalg.solve((basis.T @ basis).toarray(), basis.T @ initial.ravel())

    def objective(knots):
        shifts = basis @ knots
        positions = np.sum(coefficients * shifts[columns], axis=1)
        value, slope = _table_values(*table, positions)
        gradient = np.zeros(len(shifts))
        np.add.at(gradient, columns.ravel(), (-slope[:, None] * coefficients).ravel())
        loss = 1 - float(value.sum()) + 0.5 * knots @ (regularizer @ knots)
        return loss, basis.T @ gradient + regularizer @ knots

    fit = optimize.minimize(
        objective,
        start,
        jac=True,
        method="L-BFGS-B",
        bounds=[(-4.0, 4.0)] * len(start),
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    status = "accepted" if fit.success else "not_converged"
    if np.max(np.abs(fit.x)) >= 4 - 1e-3:
        status = "boundary"
    return np.asarray(basis @ fit.x).reshape(-1, 2), int(fit.nit), status


def refine_common_line_motion(
    projections: np.ndarray | jax.Array,
    angles: np.ndarray,
    detector: Detector,
    axis: np.ndarray,
    *,
    center_u: float,
    initial_shifts: np.ndarray,
) -> PhaseShiftSeed:
    """Refine a smooth seed, returning physical shifts for every acquisition view.

    This assumes a calibrated, unrolled parallel detector. The caller owns
    object-translation gauge removal and any fallback to the supplied seed.
    Boundary/nonconverged fits are declined, even if their coherence improves;
    reconstruction/pose accuracy must still validate an accepted phase fit.
    """
    observations = _observations(projections, angles, detector, axis)
    if observations is None:
        return PhaseShiftSeed(None, 0, 0, 0, "insufficient_evidence")
    keep, pairs, frequency, pitch, coordinates, samples = observations
    q1, q2 = (q * pitch for q in coordinates)
    cross = samples[0] * samples[1].conj()
    amplitude = np.sqrt(np.abs(cross))
    total = float(amplitude.sum())
    if not total:
        return PhaseShiftSeed(None, 0, 0, 0, "no_signal")
    cross = np.divide(cross, np.abs(cross), out=np.zeros_like(cross), where=amplitude > 0)
    center = np.array([center_u, detector.center[1]]) / pitch
    cross *= np.exp(-2j * np.pi * np.einsum("pfi,i->pf", q1 - q2, center))
    cross = (amplitude / total).astype(float) * cross.astype(complex)
    frequency = frequency * min(pitch)
    coefficients = np.concatenate([q1[:, -1], -q2[:, -1]], axis=1) / frequency[-1]
    left, right = pairs.T
    columns = np.stack([2 * left, 2 * left + 1, 2 * right, 2 * right + 1], axis=1)
    initial = np.asarray(initial_shifts)[keep] / pitch
    shifts, iterations, status = _fit_knots(cross, frequency, coefficients, columns, initial)

    def coherence(values):
        positions = np.sum(coefficients * values.ravel()[columns], axis=1)
        return float(np.sum((cross * np.exp(2j * np.pi * positions[:, None] * frequency)).real))

    full = np.column_stack(
        [np.interp(np.arange(len(angles)), keep, shifts[:, k] * pitch[k]) for k in range(2)]
    )
    before, after = coherence(initial), coherence(shifts)
    if status == "accepted" and after <= before:
        status = "not_improved"
    return PhaseShiftSeed(full if status == "accepted" else None, before, after, iterations, status)
