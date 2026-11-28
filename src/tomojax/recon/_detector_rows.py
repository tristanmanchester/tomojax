"""Physical horizontal rows through a rotated rectangular detector."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING

import numpy as np

from tomojax.geometry.coverage import detector_axes, horizontal_detector

if TYPE_CHECKING:
    import jax
    from numpy.typing import NDArray

    from tomojax._typed_arrays import Float32Array, Float64Array
    from tomojax.core.geometry import Detector


def lattice_geometry(
    detector: Detector, det_grid: tuple[jax.Array, jax.Array]
) -> tuple[Detector, float]:
    """Recover the physical center and in-plane roll of a rigid detector lattice."""
    nu, nv = detector.nu, detector.nv
    x, z = (np.asarray(a, np.float64).reshape(nv, nu) for a in det_grid)
    origin = np.array([x.mean(), z.mean()])
    qu = (np.arange(nu) - (nu - 1) / 2) * detector.du
    qv = (np.arange(nv) - (nv - 1) / 2) * detector.dv
    # Lattice axes from the extreme pixels; a single row or column is completed
    # by the rotation's right-handed perpendicular.
    span_u = np.array([1.0, 0.0])
    if nu > 1:
        span_u = np.array([x[0, -1] - x[0, 0], z[0, -1] - z[0, 0]]) / (qu[-1] - qu[0])
    if nv > 1:
        span_v = np.array([x[-1, 0] - x[0, 0], z[-1, 0] - z[0, 0]]) / (qv[-1] - qv[0])
        if nu == 1:
            span_u = np.array([span_v[1], -span_v[0]])
    else:
        span_v = np.array([-span_u[1], span_u[0]])
    rotation = np.column_stack([span_u, span_v])
    fitted_x = origin[0] + rotation[0, 0] * qu[None, :] + rotation[0, 1] * qv[:, None]
    fitted_z = origin[1] + rotation[1, 0] * qu[None, :] + rotation[1, 1] * qv[:, None]
    extent = float(np.max(np.abs(np.r_[x.ravel(), z.ravel()]))) + max(detector.du, detector.dv)
    residual = max(np.max(np.abs(x - fitted_x)), np.max(np.abs(z - fitted_z)))
    rigid = np.allclose(rotation.T @ rotation, np.eye(2), atol=1e-5) and np.linalg.det(rotation) > 0
    if residual > 1e-5 * extent or not rigid:
        raise ValueError(
            "fbp: det_grid must be the detector lattice rotated and shifted in its plane"
        )
    center = np.asarray(detector.center, np.float64)
    if np.allclose(rotation, np.eye(2), atol=1e-7) and np.allclose(
        origin, center, atol=1e-6 * extent
    ):
        return detector, 0.0
    roll = math.degrees(math.atan2(rotation.item((1, 0)), rotation.item((0, 0))))
    return replace(detector, center=(origin.item(0), origin.item(1))), roll


@dataclass(frozen=True)
class DetectorRows:
    """Sampling of an axis-aligned lattice about the measured detector's physical center."""

    measured: Detector
    roll: float
    full_parallel_turn: bool = False

    @property
    def axes(self) -> tuple[float, float]:
        return detector_axes(self.roll)

    @property
    def horizontal(self) -> Detector:
        return horizontal_detector(self.measured, self.roll)

    def coordinates(
        self, horizontal: Detector, rows: NDArray[np.int64]
    ) -> tuple[Float64Array, Float64Array]:
        """Measured pixel coordinates of each rectified pixel, before band cropping."""
        x = (np.arange(horizontal.nu, dtype=np.float64) - (horizontal.nu - 1) / 2) * horizontal.du
        z = (rows[:, None].astype(np.float64) - (horizontal.nv - 1) / 2) * horizontal.dv
        return self.pixel_coordinates(x, z)

    def pixel_coordinates(
        self, x: Float64Array, z: Float64Array
    ) -> tuple[Float64Array, Float64Array]:
        """Native pixels at physical offsets from the panel center."""
        c, s = self.axes
        det = self.measured
        u = (c * x + s * z) / det.du + (det.nu - 1) / 2
        v = (-s * x + c * z) / det.dv + (det.nv - 1) / 2
        return u, v

    def limits(
        self, horizontal: Detector, rows: NDArray[np.int64]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Half-open interval of horizontal pixel centers measured on each row."""
        u, v = self.coordinates(horizontal, rows)
        det = self.measured
        valid = (abs(u - (det.nu - 1) / 2) <= det.nu / 2 + 1e-10) & (
            abs(v - (det.nv - 1) / 2) <= det.nv / 2 + 1e-10
        )
        start: NDArray[np.int64] = np.asarray(valid.argmax(axis=1), np.int64)
        return start, start + valid.sum(axis=1)

    def band(self, horizontal: Detector, first: int, stop: int) -> tuple[int, int]:
        """Native rows needed to interpolate these complete horizontal lines."""
        lo, hi = self.native_bounds(horizontal, np.array([first, stop - 1]))
        return lo.min().item(), hi.max().item()

    def native_bounds(
        self, horizontal: Detector, rows: NDArray[np.int64]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Native interpolation support of each physical filter line."""
        x = np.array([-0.5, 0.5]) * (horizontal.nu - 1) * horizontal.du
        z = (rows[:, None].astype(np.float64) - (horizontal.nv - 1) / 2) * horizontal.dv
        _, v = self.pixel_coordinates(x, z)
        lower: Float64Array = np.min(v, axis=1)
        upper: Float64Array = np.max(v, axis=1)
        lo = np.clip(np.floor(lower).astype(np.int64) - 1, 0, self.measured.nv - 1)
        hi = np.clip(np.ceil(upper).astype(np.int64) + 2, lo + 1, self.measured.nv)
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

    def row_range(self, positions: Float64Array) -> tuple[int, int]:
        nv = self.horizontal.nv
        first = max(0, min(nv - 1, math.floor(positions.min().item())))
        stop = min(nv, max(first + 1, math.ceil(positions.max().item()) + 1))
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
        from tomojax.geometry.coverage import conjugate_weights, outer_edge

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
            lo, hi = (edge.item(0) for edge in self.limits(det, rows))
            width = hi - lo
            if not width:
                continue
            sampling = RowSamples.on_panel(
                u[:, lo:hi],
                v[:, lo:hi],
                self.measured,
                edge_slopes=bool(pad and self.full_parallel_turn),
            )
            samples = sampling.read(projections, native)[:, 0]
            begin = pad + lo
            x = (lo + (width - det.nu) / 2) * det.du + det.center[0]
            edge = outer_edge(x, full_turn=self.full_parallel_turn and not self.quarter_turn)
            measured = (
                samples * conjugate_weights(width, x / det.du) if edge is not None else samples
            )
            result[:, row - first, begin : begin + width] = measured
            if pad:
                chord = replace(det, nu=width, center=(x, det.center[1]))
                fit = Continuation(
                    parallel_axis_distance(chord, width, width), width, width, edge=edge
                )
                low, high = samples[:, : fit.window], samples[:, -fit.window :]
                radius = fit.radius(np.asarray, low.astype(np.float64), high.astype(np.float64))
                before, after = fit.continued(np.asarray, samples, radius)
                result[:, row - first, begin - width : begin] = before
                result[:, row - first, begin + width : begin + 2 * width] = after
        return result


@dataclass
class RowSamples:
    """Separable cubic convolution on the measured lattice (Keys, a = -1/2)."""

    u: Float64Array
    v: Float64Array
    edge_slopes: bool = False

    @classmethod
    def on_panel(
        cls,
        u: Float64Array,
        v: Float64Array,
        detector: Detector,
        *,
        edge_slopes: bool,
        native: int = 0,
    ) -> RowSamples:
        """Sample a measured chord, including the physical half pixels at its ends."""
        margin = 0.5 if edge_slopes else 0.0
        return cls(
            np.clip(u, -margin, detector.nu - 1 + margin),
            np.clip(v, -margin, detector.nv - 1 + margin) - native,
            edge_slopes=edge_slopes,
        )

    def taps(
        self, coordinate: Float64Array, size: int
    ) -> list[tuple[NDArray[np.int64], Float32Array]]:
        lower = np.floor(coordinate).astype(int)
        if self.edge_slopes and size > 1:
            lower = np.clip(lower, 0, size - 2)
        t = (coordinate - lower).astype(np.float32)
        weights = (
            -0.5 * t * (1 - t) ** 2,
            1 - t * t * (2.5 - 1.5 * t),
            t * (0.5 + t * (2 - 1.5 * t)),
            -0.5 * t * t * (1 - t),
        )
        if self.edge_slopes and size > 1:
            # Clamping a tilted chord at the last pixel center flattens its slope.
            # In the end cells use the line through the two measured centers.
            end = (coordinate < 1) | (coordinate > size - 2)
            weights = tuple(
                np.where(end, linear, w)
                for linear, w in zip((0, 1 - t, t, 0), weights, strict=True)
            )
        return [
            (np.clip(lower + delta, 0, size - 1), w)
            for delta, w in zip(range(-1, 3), weights, strict=True)
            if np.any(w)
        ]

    def read(
        self, frames: np.ndarray | jax.Array, offset: int, views: NDArray[np.int64] | None = None
    ) -> Float32Array:
        rows = self.taps(self.v - offset, frames.shape[1])
        columns = self.taps(self.u, frames.shape[2])
        selected = slice(None) if views is None else views.reshape((-1,) + (1,) * self.u.ndim)
        count = len(frames) if views is None else len(views)
        result = np.zeros((count, *self.u.shape), np.float32)
        for row, wy in rows:
            for column, wx in columns:
                w = wy * wx
                result += np.where(w != 0, frames[selected, row, column] * w, 0)
        return result
