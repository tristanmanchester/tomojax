"""Fourier common lines for parallel-beam setup initialization.

Two projection Fourier planes intersect along one line. Their complex values
agree there, apart from the phase due to the detector origin and sample motion.
Magnitudes therefore constrain the rotation axis without fitting translations;
phase constrains the detector-u centre once the axis is known. These are seeds,
not calibration certificates: truncation and per-view motion break consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy import fft, ndimage, optimize
from scipy.signal.windows import tukey

from tomojax.core.geometry.transforms import align_u_to_v

if TYPE_CHECKING:
    import jax

    from tomojax.core.geometry.base import Detector


@dataclass(frozen=True)
class PhaseShiftSeed:
    """Selected-view physical (u, v) shifts, or a declined fit with diagnostics."""

    shifts: np.ndarray | None
    coherence_before: float
    coherence_after: float
    iterations: int
    status: str


@dataclass(frozen=True)
class CommonLines:
    """Small host spectra and angular pairs shared by setup searches."""

    spectra: np.ndarray
    angles: np.ndarray
    pairs: np.ndarray
    frequency: np.ndarray
    pitch: tuple[float, float]
    native_du: float
    native_width: int
    center_v: float
    center_u: float
    view_indices: np.ndarray

    def _coordinates(self, axis_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rotation = align_u_to_v(np.array([0.0, 0.0, 1.0]), np.asarray(axis_unit))
        c, s = np.cos(self.angles), np.sin(self.angles)
        rz = np.zeros((len(c), 3, 3))
        rz[:, 0, 0], rz[:, 0, 1] = c, -s
        rz[:, 1, 0], rz[:, 1, 1], rz[:, 2, 2] = s, c, 1
        poses = rotation @ rz
        i, j = self.pairs.T
        line = np.cross(poses[i, 1], poses[j, 1])
        norm = np.linalg.norm(line, axis=1, keepdims=True)
        line = np.divide(line, norm, out=np.zeros_like(line), where=norm > 1e-8)
        return tuple(
            np.einsum("pdi,pi->pd", poses[k][:, [0, 2]], line)[:, None, :]
            * self.frequency[None, :, None]
            for k in (i, j)
        )

    def sample(self, axis_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Corresponding complex samples, each shaped ``(pairs, frequencies)``."""
        return self._sample_coordinates(self._coordinates(axis_unit))

    def _sample_coordinates(
        self, coordinates: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        nv, nu = self.spectra.shape[1:]
        du, dv = self.pitch
        values = []
        for indices, q in zip(self.pairs.T, coordinates, strict=True):
            u = q[..., 0] * du * nu + nu // 2
            v = q[..., 1] * dv * nv + nv // 2
            points = [np.broadcast_to(indices[:, None], u.shape), v, u]
            values.append(ndimage.map_coordinates(self.spectra, points, order=1, mode="constant"))
        return values[0], values[1]

    def center(self, axis_unit: np.ndarray) -> float | None:
        """Absolute detector-u centre in native pixels, or None without evidence.

        Search within a quarter detector width of the supplied centre, not zero:
        an already calibrated wide offset must remain inside the capture range.
        A boundary optimum is declined rather than reported as a centre.
        A common object translation cancels between the two Fourier planes;
        independent view translations do not, so this remains only a seed.
        """
        q1, q2 = self._coordinates(axis_unit)
        slope = (q1[..., 0] - q2[..., 0]) * self.native_du
        if np.max(np.abs(slope)) < 1e-5:
            return None  # Axial tomography's common lines cannot locate COR.
        left, right = self._sample_coordinates((q1, q2))
        # A rolled projected axis couples detector-v and u origins. Keep the
        # supplied v reference fixed; its independent calibration is a gauge.
        cross = (
            left * right.conj() * np.exp(-2j * np.pi * (q1[..., 1] - q2[..., 1]) * self.center_v)
        ).ravel()
        amplitude = np.sqrt(np.abs(cross))
        total = float(amplitude.sum())
        if total == 0:
            return None
        cross = np.divide(cross, amplitude, out=np.zeros_like(cross), where=amplitude > 0)
        slope = slope.ravel()

        def score(center: float) -> float:
            return float(np.real(np.sum(cross * np.exp(-2j * np.pi * center * slope))) / total)

        bound = self.native_width / 4
        # Detector centres are physical lengths; search/report native pixels,
        # independently of the additional preview downsampling.
        prior = self.center_u / self.native_du
        candidates = np.linspace(prior - bound, prior + bound, max(65, self.native_width + 1))
        scores = np.asarray([score(c) for c in candidates])
        best = int(np.argmax(scores))
        if best in (0, len(candidates) - 1) or np.ptp(scores) < 1e-3:
            return None
        lower, upper = candidates[max(0, best - 1)], candidates[min(best + 1, len(scores) - 1)]
        fit = optimize.minimize_scalar(
            lambda c: -score(c), bounds=(lower, upper), method="bounded", options={"xatol": 1e-3}
        )
        return float(cast("optimize.OptimizeResult", fit).x)

    def motion(self, axis_unit: np.ndarray, *, center_u: float) -> PhaseShiftSeed:
        """Seed small, smooth detector motion after setup calibration.

        ``center_u`` and returned shifts use physical detector units. Returned
        rows follow ``view_indices``; interpolate in acquisition order before local pose
        refinement. This is not a global shift search: each component is bounded
        to four preview pixels and nonconverged/boundary fits are declined.
        A second-difference prior favours smooth motion, not arbitrary jitter.
        The reconstruction still supplies the acceptance test on truncated data.
        """
        pitch = np.asarray(self.pitch)
        q1, q2 = (q * pitch for q in self._coordinates(axis_unit))
        f, g = self.sample(axis_unit)
        cross = f * g.conj()
        amplitude = np.sqrt(np.abs(cross))
        total = float(amplitude.sum())
        if total == 0:
            return PhaseShiftSeed(None, 0.0, 0.0, 0, "no_signal")
        weight = amplitude / total
        cross = np.divide(cross, np.abs(cross), out=np.zeros_like(cross), where=amplitude > 0)
        center = np.array([center_u, self.center_v]) / pitch
        cross *= np.exp(-2j * np.pi * np.einsum("pfi,i->pf", q1 - q2, center))
        left, right = self.pairs.T
        # Motion follows acquisition order, including repeated/turnaround angles.
        order = np.argsort(self.view_indices)
        step = np.diff(self.view_indices[order]).astype(float)
        step = step[:, None] / step.mean()
        middle_step = (step[:-1] + step[1:]) / 2
        n = len(self.angles)

        def coherence(x: np.ndarray) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
            x = x.reshape(n, 2)
            phase = (
                2
                * np.pi
                * (np.einsum("pfi,pi->pf", q1, x[left]) - np.einsum("pfi,pi->pf", q2, x[right]))
            )
            return cross * np.exp(1j * phase)

        def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
            x = x.reshape(n, 2)
            z = coherence(x)
            loss = float(np.sum(weight * (1 - z.real)))
            grad = np.zeros_like(x)
            np.add.at(grad, left, np.einsum("pf,pfi->pi", 2 * np.pi * weight * z.imag, q1))
            np.add.at(grad, right, -np.einsum("pf,pfi->pi", 2 * np.pi * weight * z.imag, q2))
            slope = np.diff(x[order], axis=0) / step
            difference = np.diff(slope, axis=0) / middle_step
            # Do not wrap the endpoints: a scan-long drift is observable.
            loss += 0.01 * float(np.mean(difference**2)) + 1e-5 * float(np.mean(x**2))
            derivative = 0.02 * difference / (difference.size * middle_step)
            slope_grad = np.zeros_like(slope)
            slope_grad[:-1] -= derivative
            slope_grad[1:] += derivative
            grad[order[:-1]] -= slope_grad / step
            grad[order[1:]] += slope_grad / step
            grad += 2e-5 * x / x.size
            return loss, grad.ravel()

        initial = np.zeros(2 * n)
        fit = optimize.minimize(
            objective,
            initial,
            jac=True,
            method="L-BFGS-B",
            bounds=[(-4.0, 4.0)] * initial.size,
            options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
        )
        shifts = np.asarray(fit.x).reshape(n, 2)
        status = "accepted" if fit.success else "not_converged"
        if np.max(np.abs(shifts)) >= 4.0 - 1e-3:
            status = "boundary"
        return PhaseShiftSeed(
            shifts * pitch if status == "accepted" else None,
            float(np.sum(weight * cross.real)),
            float(np.sum(weight * coherence(fit.x).real)),
            int(fit.nit),
            status,
        )


def _angular_pairs(angles: np.ndarray) -> np.ndarray:
    pairs = set()
    for i, angle in enumerate(angles):
        for offset in np.deg2rad([30, 60, 90, 120, 150]):
            error = np.abs((angles - angle - offset + np.pi) % (2 * np.pi) - np.pi)
            j = int(error.argmin())
            if i != j and error[j] < np.deg2rad(8):
                pairs.add((i, j))
    return np.asarray(sorted(pairs), dtype=np.int32).reshape(-1, 2)


def prepare_common_lines(
    projections: np.ndarray | jax.Array, angles: np.ndarray, detector: Detector
) -> CommonLines | None:
    """Prepare at most 64 views and 192-pixel previews for bounded CPU searches.

    Fourier origins retain half-pixel centres and any asymmetric binning tail,
    so phase estimates remain in the input detector's native pixel convention.
    """
    order = np.argsort(np.mod(angles, 360), kind="stable")
    keep = order[np.linspace(0, len(order) - 1, min(64, len(order))).astype(int)]
    # Canonical physical angles; acquisition chronology is carried by keep.
    theta = np.deg2rad(np.mod(np.asarray(angles)[keep], 360))
    pairs = _angular_pairs(theta)
    if len(pairs) < 3 or min(projections.shape[1:]) < 8:
        return None
    data = np.asarray(projections[keep], np.float32)
    original_nv, original_nu = data.shape[1:]
    factor = max(1, int(np.ceil(max(original_nv, original_nu) / 192)))
    nv, nu = original_nv // factor, original_nu // factor
    data = data[:, : nv * factor, : nu * factor].reshape(-1, nv, factor, nu, factor).mean((2, 4))
    du, dv = detector.du * factor, detector.dv * factor
    blur = 3 * min(du, dv)
    data = data - ndimage.gaussian_filter(data, (0, blur / dv, blur / du))
    data *= tukey(nv, 0.25)[None, :, None] * tukey(nu, 0.25)[None, None, :]
    shape = fft.next_fast_len(4 * nv), fft.next_fast_len(4 * nu)
    spectra = fft.fftshift(fft.fft2(data, s=shape), axes=(-2, -1))
    # Original detector origin, expressed in binned coordinates, including tails.
    origin_u = (original_nu - factor) / (2 * factor)
    origin_v = (original_nv - factor) / (2 * factor)
    fu = fft.fftshift(fft.fftfreq(shape[1]))
    fv = fft.fftshift(fft.fftfreq(shape[0]))
    spectra *= np.exp(2j * np.pi * (fu[None, :] * origin_u + fv[:, None] * origin_v))
    frequency = np.linspace(2 / min(nv * dv, nu * du), 0.45 / max(du, dv), 128)
    return CommonLines(
        spectra,
        theta,
        pairs,
        frequency,
        (du, dv),
        detector.du,
        original_nu,
        detector.center[1],
        detector.center[0],
        keep,
    )
