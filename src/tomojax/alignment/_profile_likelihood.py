"""Noise-weighted registration and uncertainty for a shared axial mass profile."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft, interpolate, linalg, optimize


@dataclass(frozen=True)
class ProfileLikelihood:
    """Translations and local likelihood errors in the zero-mean gauge, in pixels."""

    shifts: np.ndarray
    standard_error: np.ndarray
    covariance: np.ndarray
    relative_noise_variance: np.ndarray
    ungauged_variance: np.ndarray


def _stationary_covariance(residual, jacobian, parameter_covariance):
    """Shrink the finite-window lag estimate towards white noise by its sampling risk."""
    count, width = residual.shape
    size = fft.next_fast_len(2 * width)
    power = abs(np.asarray(fft.rfft(residual, n=size, axis=1))) ** 2
    transformed = np.asarray(fft.rfft(jacobian, n=size, axis=1))
    # A fitted residual has covariance C - J (J.T C^-1 J)^-1 J.T.
    power += np.einsum(
        "nka,nab,nkb->nk", transformed, parameter_covariance, transformed.conj()
    ).real
    individual = np.asarray(fft.irfft(power, n=size, axis=1))[:, :width] / width
    # The common template also consumes one of the independent view replicates.
    individual *= count / (count - 1)
    covariance = individual.mean(axis=0)
    variance = individual.var(axis=0, ddof=1) / count
    multiplicity = 2 * (width - np.arange(1, width))
    sampling = multiplicity @ variance[1:]
    signal = multiplicity @ covariance[1:] ** 2
    shrinkage = np.clip(sampling / max(signal, np.finfo(float).tiny), 0, 1)
    covariance[1:] *= 1 - shrinkage
    matrix = linalg.toeplitz(covariance)
    matrix.flat[:: width + 1] += np.finfo(float).eps * covariance[0] * width
    return matrix


def _weighted_columns(lower, columns):
    if lower is None:
        return columns
    count, width, rank = columns.shape
    white = linalg.solve_triangular(
        lower, columns.transpose(1, 0, 2).reshape(width, -1), lower=True, check_finite=False
    )
    return white.reshape(width, count, rank).transpose(1, 0, 2)


def _profile_covariance(observations, template, coordinates, shifts, nuisance, lower):
    count, width = observations.shape
    model = np.asarray([template(coordinates - shift) for shift in shifts])
    design = np.concatenate(
        [model[:, :, None], np.broadcast_to(nuisance, (count, *nuisance.shape))], axis=2
    )
    weighted = _weighted_columns(lower, design)
    target = _weighted_columns(lower, observations[:, :, None])[:, :, 0]
    gram = np.einsum("nmi,nmj->nij", weighted, weighted)
    rhs = np.einsum("nmi,nm->ni", weighted, target)
    coefficients = np.linalg.solve(gram, rhs[..., None])[..., 0]
    residual = observations - np.einsum("nmi,ni->nm", design, coefficients)
    derivative = -coefficients[:, 0, None] * np.asarray(
        [template(coordinates - shift, 1) for shift in shifts]
    )
    jacobian = np.concatenate([design, derivative[:, :, None]], axis=2)
    white = _weighted_columns(lower, jacobian)
    gram = np.einsum("nmi,nmj->nij", white, white)
    inverse = np.linalg.pinv(gram, hermitian=True)
    if lower is None:
        inverse *= np.mean(residual**2) * width / (width - jacobian.shape[-1])
    return _stationary_covariance(residual, jacobian, inverse)


def _profile_translations(observations, template, coordinates, shifts, nuisance, lower):
    basis = np.asarray(
        linalg.qr(linalg.solve_triangular(lower, nuisance, lower=True), mode="economic")[0]
    )

    def whiten(values):
        white = linalg.solve_triangular(lower, values, lower=True, check_finite=False)
        return white - basis @ (basis.T @ white)

    targets = whiten(observations.T).T
    variance = np.empty(len(shifts))
    scales = np.empty(len(shifts))
    for i, target in enumerate(targets):

        def objective(shift, target=target):
            prediction = whiten(template(coordinates - shift))
            residual = target - (target @ prediction) / (prediction @ prediction) * prediction
            return residual @ residual

        result = optimize.minimize_scalar(
            objective,
            bounds=(shifts[i] - 2, shifts[i] + 2),
            method="bounded",
            options={"xatol": 1e-7},
        )
        shifts[i] = result.x
        prediction = whiten(template(coordinates - shifts[i]))
        derivative = whiten(-template(coordinates - shifts[i], 1))
        gain = target @ prediction / (prediction @ prediction)
        # Profile gain as well as background: uncertainty is not conditional on
        # the fitted contrast. This is the scalar Schur complement of their Gram.
        derivative = gain * (
            derivative - prediction * (prediction @ derivative) / (prediction @ prediction)
        )
        # Views can have different noise amplitudes with the same correlation
        # shape. Normalize relative residual energies across independent views,
        # not by pixel count: shrinkage of the covariance changes the effective
        # rank of a whitened residual, especially for smoothly correlated noise.
        residual = target - gain * prediction
        scales[i] = residual @ residual
        variance[i] = scales[i] / (derivative @ derivative)
    relative_noise = scales / scales.mean()
    return shifts - shifts.mean(), variance / scales.mean(), relative_noise


def fit_profile_likelihood(profiles: np.ndarray, initial: np.ndarray) -> ProfileLikelihood:
    """Fit translated profiles with an estimated stationary detector covariance.

    Each view has its own gain, affine background and noise amplitude.
    Return shifts and conditional standard errors in the zero-mean gauge.
    """
    profiles = np.asarray(profiles, np.float64)
    shifts = np.asarray(initial, np.float64).copy()
    count, width = profiles.shape
    shifts -= shifts.mean()
    axis = np.arange(width, dtype=float)
    margin = max(width // 8, int(np.ceil(abs(shifts).max())) + 2)
    coordinates = axis[margin:-margin]
    background = np.column_stack([np.ones(width), np.linspace(-1, 1, width)])
    data = profiles - (background @ np.linalg.lstsq(background, profiles.T, rcond=None)[0]).T
    observations = data[:, margin:-margin]
    nuisance = np.column_stack([np.ones(len(coordinates)), np.linspace(-1, 1, len(coordinates))])
    splines = [interpolate.CubicSpline(axis, row, extrapolate=False) for row in data]
    lower = None
    for _ in range(4):
        aligned = np.asarray(
            [spline(axis + shift) for spline, shift in zip(splines, shifts, strict=True)]
        )
        valid = np.isfinite(aligned)
        reference = np.where(valid, aligned, 0).sum(0) / np.maximum(valid.sum(0), 1)
        template = interpolate.CubicSpline(axis, reference, extrapolate=False)
        covariance = _profile_covariance(
            observations, template, coordinates, shifts, nuisance, lower
        )
        lower = linalg.cholesky(covariance, lower=True)
        shifts, variance, relative_noise = _profile_translations(
            observations, template, coordinates, shifts, nuisance, lower
        )
    gauged_variance = variance * (1 - 2 / count) + variance.sum() / count**2
    return ProfileLikelihood(shifts, np.sqrt(gauged_variance), covariance, relative_noise, variance)
