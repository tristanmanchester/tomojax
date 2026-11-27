"""Physical horizontal rows through a rotated rectangular detector."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import jax

    from tomojax.core.geometry import Detector


@dataclass(frozen=True)
class DetectorRows:
    """Sampling of an axis-aligned lattice about the measured detector's physical center."""

    measured: Detector
    roll: float

    @property
    def axes(self) -> tuple[float, float]:
        # Exact signed permutations at quarter turns, including on nonsquare panels.
        turns, residual = divmod(self.roll, 90.0)
        c, s = np.cos(np.deg2rad(residual)), np.sin(np.deg2rad(residual))
        return ((c, s), (-s, c), (-c, -s), (s, -c))[int(turns) % 4]

    @property
    def horizontal(self) -> Detector:
        c, s = self.axes
        det = self.measured
        du = 1 / math.hypot(c / det.du, s / det.dv)
        dv = 1 / math.hypot(s / det.du, c / det.dv)
        # Retain the complete footprint, with the same sampling at equivalent rolls.
        width, height = det.nu * det.du, det.nv * det.dv
        nu = max(1, math.ceil((abs(c) * width + abs(s) * height) / du - 1e-10))
        nv = max(1, math.ceil((abs(s) * width + abs(c) * height) / dv - 1e-10))
        return replace(det, nu=nu, nv=nv, du=du, dv=dv)

    def coordinates(self, horizontal: Detector, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Measured pixel coordinates of each rectified pixel, before band cropping."""
        det = self.measured
        x = (np.arange(horizontal.nu) - (horizontal.nu - 1) / 2) * horizontal.du
        z = (rows[:, None] - (horizontal.nv - 1) / 2) * horizontal.dv
        c, s = self.axes
        u = (c * x + s * z) / det.du + (det.nu - 1) / 2
        v = (-s * x + c * z) / det.dv + (det.nv - 1) / 2
        return u, v

    def limits(self, horizontal: Detector, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Half-open interval of horizontal pixel centers measured on each row."""
        u, v = self.coordinates(horizontal, rows)
        det = self.measured
        valid = (u >= -0.5) & (u < det.nu - 0.5) & (v >= -0.5) & (v < det.nv - 0.5)
        start = np.argmax(valid, axis=1)
        return start, start + valid.sum(axis=1)

    def band(self, horizontal: Detector, first: int, stop: int) -> tuple[int, int]:
        """Native rows needed to interpolate these complete horizontal lines."""
        _, v = self.coordinates(horizontal, np.array([first, stop - 1]))
        lo = int(np.clip(np.floor(v.min()) - 1, 0, self.measured.nv - 1))
        hi = int(np.clip(np.ceil(v.max()) + 2, lo + 1, self.measured.nv))
        return lo, hi

    @property
    def quarter_turn(self) -> bool:
        return self.roll % 90 == 0

    def filter_detector(self, exterior: str) -> Detector:
        det = self.horizontal
        # Oblique chords end at different columns. Include both true tails in
        # the uniform lattice consumed by the FFT, leaving zero beyond them.
        return (
            replace(det, nu=3 * det.nu) if exterior == "cylinder" and not self.quarter_turn else det
        )

    def row_range(self, positions: np.ndarray) -> tuple[int, int]:
        nv = self.horizontal.nv
        first = int(np.clip(np.floor(positions.min()), 0, nv - 1))
        stop = int(np.clip(np.ceil(positions.max()) + 1, first + 1, nv))
        return first, stop

    def parallel_band(
        self,
        projections: np.ndarray | jax.Array,
        first: int,
        stop: int,
        *,
        native: int,
        exterior: str,
    ) -> np.ndarray:
        """Rectify a band of parallel data and continue its actual measured horizontal chords."""
        from tomojax.core.host_volume import host_array

        from ._continuation import Continuation, parallel_axis_distance

        det = self.horizontal
        if self.quarter_turn:
            # Native is a row band in the measured array; slicing after a full
            # permutation requires its offset along the permuted axis.
            k = -int(self.roll // 90)
            if native == 0 and projections.shape[1] == self.measured.nv:
                return np.rot90(projections, k=k, axes=(1, 2))[:, first:stop]
        wide = self.filter_detector(exterior)
        result = host_array((projections.shape[0], stop - first, wide.nu))
        pad = (wide.nu - det.nu) // 2
        for row in range(first, stop):
            rows = np.array([row])
            u, v = self.coordinates(det, rows)
            lo, hi = (int(edge[0]) for edge in self.limits(det, rows))
            width = hi - lo
            if not width:
                continue
            u = np.clip(u[:, lo:hi], 0, self.measured.nu - 1)
            v = np.clip(v[:, lo:hi], 0, self.measured.nv - 1)
            samples = RowSamples(u, v).read(projections, native)[:, 0]
            begin = pad + lo
            result[:, row - first, begin : begin + width] = samples
            if pad:
                x = (lo + (width - 1) / 2 - (det.nu - 1) / 2) * det.du + det.center[0]
                chord = replace(det, nu=width, center=(x, det.center[1]))
                fit = Continuation(parallel_axis_distance(chord, width, width), width, width)
                low, high = samples[:, : fit.window], samples[:, -fit.window :]
                radius = fit.radius(np.asarray, low.astype(np.float64), high.astype(np.float64))
                before, after = fit.continued(np.asarray, samples, radius)
                result[:, row - first, begin - width : begin] = before
                result[:, row - first, begin + width : begin + 2 * width] = after
        return result


@dataclass
class RowSamples:
    """Separable cubic convolution on the measured lattice (Keys, a = -1/2)."""

    u: np.ndarray
    v: np.ndarray

    @staticmethod
    def taps(coordinate: np.ndarray, size: int) -> list[tuple[np.ndarray, np.ndarray]]:
        lower = np.floor(coordinate).astype(int)
        t = (coordinate - lower).astype(np.float32)
        weights = (
            -0.5 * t * (1 - t) ** 2,
            1 - t * t * (2.5 - 1.5 * t),
            t * (0.5 + t * (2 - 1.5 * t)),
            -0.5 * t * t * (1 - t),
        )
        return [
            (np.clip(lower + delta, 0, size - 1), w)
            for delta, w in zip(range(-1, 3), weights, strict=True)
            if np.any(w)
        ]

    def read(self, frames: np.ndarray | jax.Array, offset: int) -> np.ndarray:
        rows = self.taps(self.v - offset, frames.shape[1])
        columns = self.taps(self.u, frames.shape[2])
        result = np.zeros((len(frames), *self.u.shape), np.float32)
        for row, wy in rows:
            for column, wx in columns:
                w = wy * wx
                result += np.where(w != 0, frames[:, row, column] * w, 0)
        return result
