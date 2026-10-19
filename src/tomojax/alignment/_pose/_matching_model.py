"""Band-limited parallel-beam translation model for joint image/pose fitting.

The measured rows and the physical prediction use the same centred sampling.
The prediction is a fixed Joseph map, finite zero-extended sinc translation,
and detector low-pass sampling; its transpose reverses these operations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy import fft, signal

from tomojax.alignment._geometry.parametrizations import apply_pose_update
from tomojax.core.joseph import (
    forward_project_planes,
    plane_coefficients,
    sum_backproject_planes,
)
from tomojax.geometry import Detector


@dataclass(frozen=True)
class JointRefinement:
    """Native detector sampling and the joint image metric's ramp damping."""

    native_width: int
    ramp_damping: float = 1e-3


def resample_rows(data: np.ndarray, width: int) -> np.ndarray:
    """Low-pass rows at the target pixel centres without changing their field of view."""
    size = data.shape[-1]
    offset = (size / width - 1) / 2
    spectrum = fft.fft(data, axis=-1)
    spectrum *= np.exp(2j * np.pi * fft.fftfreq(size) * offset)
    resampled = np.asarray(signal.resample(spectrum, width, axis=-1, domain="freq"))
    return np.real(resampled).astype(data.dtype)


def nuisance_filter(native_width: int, detector: Detector) -> Callable:
    """Profile native row offsets and a shared native detector slope after sampling."""
    native = np.arange(native_width, dtype=np.float32) - (native_width - 1) / 2
    ramp = jnp.broadcast_to(
        jnp.asarray(resample_rows(native, detector.nu)), (detector.nv, detector.nu)
    )
    ramp /= jnp.linalg.norm(ramp)

    def apply(value):
        centred = value - jnp.mean(value, axis=-1, keepdims=True)
        return centred - jnp.sum(centred * ramp, axis=(-2, -1), keepdims=True) * ramp

    return apply


def _sampling(width: int, *, absolute: bool):
    size = 2 * width
    frequency = jnp.fft.rfftfreq(size)
    transfer = (jnp.arange(size // 2 + 1) <= width // 2) * jnp.exp(1j * jnp.pi * frequency)
    if absolute:
        transfer = jnp.fft.rfft(jnp.abs(jnp.fft.irfft(transfer, n=size)), n=size)

    def reduce(value):
        return jnp.fft.irfft(jnp.fft.rfft(value, n=size) * transfer, n=size)[..., ::2]

    def expand(value):
        padded = jnp.zeros((*value.shape[:-1], size), value.dtype).at[..., ::2].set(value)
        return jnp.fft.irfft(jnp.fft.rfft(padded, n=size) * jnp.conj(transfer), n=size)

    return reduce, expand


class FixedPoseOperators(NamedTuple):
    """Paired Krylov products and value/first/second derivatives at a fitted image."""

    forward: Callable
    adjoint: Callable
    value_and_derivatives: Callable


class ProjectionOperators(NamedTuple):
    """Physical products and a transient fixed-pose linearization."""

    forward: Callable
    adjoint: Callable
    derivative: Callable
    fixed_pose: Callable


class _FiniteSampling:
    """Own detector coordinates, Fourier factors and the nominal Joseph map."""

    def __init__(self, grid, detector, backend, masked, absolute) -> None:
        self.grid, self.backend, self.masked, self.absolute = grid, backend, masked, absolute
        self.width = 2 * detector.nu
        self.pitch = detector.du / 2
        self.length = 2 * self.width + 1
        self.begin = (self.length - self.width) // 2
        self.size = self.length + self.width - 1
        centre = detector.center[0] + self.pitch * ((self.length - self.width) / 2 - self.begin)
        self.extended = Detector(
            self.length, detector.nv, self.pitch, detector.dv, (centre, detector.center[1])
        )
        index = jnp.arange(self.size, dtype=jnp.float32)
        self.offset = jnp.where(index <= self.begin + self.width - 1, index, index - self.size)
        self.reduce, self.expand = _sampling(detector.nu, absolute=absolute)

    def kernel(self, poses, order=0):
        shift = poses[:, 0, 3, None, None] / self.pitch

        def first(value):
            return jax.jvp(
                lambda u: jnp.sinc(self.offset - u), (value,), (jnp.ones_like(value) / self.pitch,)
            )

        value, tangent = first(shift)
        if order == 2:
            tangent = jax.jvp(
                lambda u: first(u)[1], (shift,), (jnp.ones_like(shift) / self.pitch,)
            )[1]
        return jnp.fft.rfft(
            tangent if order else jnp.abs(value) if self.absolute else value, n=self.size
        )

    def nominal(self, poses):
        return plane_coefficients(poses.at[:, 0, 3].set(0), self.grid, self.extended)

    def spectrum(self, coefficients, volume):
        rays = forward_project_planes(
            coefficients, self.masked(volume), self.grid, self.extended, backend=self.backend
        )
        return jnp.fft.rfft(rays, n=self.size)

    def sampled(self, spectrum, transfer):
        rays = jnp.fft.irfft(spectrum * transfer, n=self.size)
        return self.reduce(rays[..., self.begin : self.begin + self.width])

    def padded_spectrum(self, rows):
        padded = jnp.pad(
            self.expand(rows), ((0, 0), (0, 0), (self.begin, self.size - self.width - self.begin))
        )
        return jnp.fft.rfft(padded, n=self.size)

    def transpose_spectrum(self, coefficients, spectrum):
        rays = jnp.fft.irfft(spectrum, n=self.size)[..., : self.length]
        return self.masked(
            sum_backproject_planes(
                coefficients, rays, self.grid, self.extended, backend=self.backend
            )
        )

    def forward(self, poses, volume, _frames=None):
        return self.sampled(self.spectrum(self.nominal(poses), volume), self.kernel(poses))

    def derivative(self, poses, volume):
        return self.sampled(self.spectrum(self.nominal(poses), volume), self.kernel(poses, 1))

    def adjoint(self, poses, value, _frames=None, accumulate=None):
        spectrum = self.padded_spectrum(value) * jnp.conj(self.kernel(poses))
        volume = self.transpose_spectrum(self.nominal(poses), spectrum)
        return volume if accumulate is None else volume + accumulate

    def fixed_pose(self, poses):
        """Share one physical projection/backprojection between the value and derivative."""
        coefficients = self.nominal(poses)
        h, dh = self.kernel(poses), self.kernel(poses, 1)

        def value_and_derivative(volume):
            rays = self.spectrum(coefficients, volume)
            return self.sampled(rays, h), self.sampled(rays, dh)

        def transpose_pair(value, tangent):
            spectrum = self.padded_spectrum(value) * jnp.conj(h) + self.padded_spectrum(
                tangent
            ) * jnp.conj(dh)
            return self.transpose_spectrum(coefficients, spectrum)

        def value_and_derivatives(volume):
            rays = self.spectrum(coefficients, volume)
            return tuple(
                self.sampled(rays, transfer) for transfer in (h, dh, self.kernel(poses, 2))
            )

        return FixedPoseOperators(value_and_derivative, transpose_pair, value_and_derivatives)


def projectors(grid, detector, *, backend: str, masked: Callable, absolute: bool = False):
    """Matched products; absolute factors bound the operator, not the likelihood."""
    model = _FiniteSampling(grid, detector, backend, masked, absolute)
    return ProjectionOperators(model.forward, model.adjoint, model.derivative, model.fixed_pose)


def pose_columns(grid, detector, *, backend: str, indices: tuple[int, ...]) -> Callable:
    """Bind the same physical prediction and its active translation columns."""
    assert set(indices) <= {3, 5}
    model = projectors(grid, detector, backend=backend, masked=lambda x: x)

    def columns(params, nominal, _frame, volume, target, weight):
        poses = apply_pose_update(nominal, params, translation_frame="detector")[None]
        prediction, du = model.fixed_pose(poses).forward(volume)
        prediction, du = prediction[0], du[0]
        values = jnp.stack([du if i == 3 else jnp.zeros_like(du) for i in indices])
        return (
            weight.ravel() * (prediction - target).ravel(),
            weight.ravel() * values.reshape(len(indices), -1),
        )

    return columns


def constrain_pose_inverse(solve: Callable, poses, counts, across: Callable) -> Callable:
    """Fix the two global image-translation modes inside the pose inverse.

    ``across`` sums over view shares. Padding contributes neither a constraint
    nor a correction; no observable pose trajectory is penalised.
    """
    basis = jnp.zeros((poses.shape[0], 6, 2), poses.dtype).at[:, 3, :].set(poses[:, 0, :2])
    basis *= counts[:, None, None]
    inverse = jnp.stack([solve(basis[:, :, i]) for i in range(2)], axis=-1)
    precision = jax.lax.Precision.HIGHEST
    gram = across(jnp.einsum("nki,nkj->ij", basis, inverse, precision=precision))

    def constrained(rhs):
        value = solve(rhs)
        rows = across(jnp.einsum("nki,nk->i", basis, value, precision=precision))
        multipliers = jnp.linalg.solve(gram, rows)
        return value - jnp.einsum("nki,i->nk", inverse, multipliers, precision=precision)

    return constrained
