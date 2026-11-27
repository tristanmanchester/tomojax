"""FDK on horizontal chords of a rolled flat panel.

Each calibrated arrangement gets an axis-aligned detector in the same plane.
A chord ends where it meets the measured rectangle. Its two edges determine
conjugate coverage and cylinder continuation before the ordinary 1-D ramp.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from scipy.fft import irfft, rfft

from ._continuation import Continuation, cone_axis_distance, radius_views
from ._detector_rows import DetectorRows, RowSamples
from ._fdk_angles import fan_angles, grouped_angles, ordered_arc, parker_weights
from ._fdk_coverage import full_turn_weights, reliability
from .filters import get_fbp_filter_np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Protocol

    import jax

    from tomojax._parameters import Filter
    from tomojax.core.geometry import ConeBeam, Detector, Geometry

    class ProjectionBands(Protocol):
        def __getitem__(self, views: slice, /) -> np.ndarray | jax.Array: ...

    type Parts = tuple[tuple[int | None, ConeBeam, Detector], ...]


@dataclass
class FanRows:
    parts: Parts
    layouts: tuple[DetectorRows, ...]
    angles: tuple[np.ndarray, np.ndarray, np.ndarray]

    @classmethod
    def for_scan(cls, geometry: Geometry, parts: Parts, n_views: int) -> FanRows | None:
        if not any(beam.detector_roll for _, beam, _ in parts):
            return None
        layouts = tuple(DetectorRows(det, beam.detector_roll) for _, beam, det in parts)
        # Equal physical arrays under a reindexing have the same rectified pitch.
        # Pad smaller segment footprints symmetrically to the common array shape.
        horizontal = [layout.horizontal for layout in layouts]
        du, dv = min(d.du for d in horizontal), min(d.dv for d in horizontal)
        nu = max(int(np.ceil(d.nu * d.du / du - 1e-10)) for d in horizontal)
        nv = max(int(np.ceil(d.nv * d.dv / dv - 1e-10)) for d in horizontal)
        virtual = tuple(
            (n, replace(beam, detector_roll=0), replace(d, nu=nu, nv=nv, du=du, dv=dv))
            for (n, beam, _), d in zip(parts, horizontal, strict=True)
        )
        return cls(virtual, layouts, grouped_angles(geometry, n_views))

    @cached_property
    def permuted(self) -> bool:
        """Whether every horizontal line is exactly a measured row or column."""
        return all(
            layout.quarter_turn and layout.horizontal == det
            for layout, (_, _, det) in zip(self.layouts, self.parts, strict=True)
        )

    @cached_property
    def bounds(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        return tuple(
            layout.limits(det, np.arange(det.nv))
            for layout, (_, _, det) in zip(self.layouts, self.parts, strict=True)
        )

    @cached_property
    def fans(self) -> tuple[np.ndarray, ...]:
        return tuple(fan_angles(beam, det) for _, beam, det in self.parts)

    @cached_property
    def counts(self) -> tuple[int, ...]:
        return tuple(n or len(self.angles[1]) for n, _, _ in self.parts)

    @cached_property
    def ordering(self) -> tuple[np.ndarray, np.ndarray, bool]:
        return ordered_arc(self.angles[0])

    @cached_property
    def measure(self) -> np.ndarray:
        angles, groups, repeats = self.angles
        order = np.argsort(angles)
        nodes = angles[order]
        measure = np.empty_like(nodes)
        wrapped = np.r_[nodes[-1] - 2 * np.pi, nodes, nodes[0] + 2 * np.pi]
        measure[order] = (wrapped[2:] - wrapped[:-2]) / 2
        return measure[groups] / repeats[groups]

    def band(self, first: int, stop: int) -> tuple[int, int]:
        bands = [
            layout.band(det, first, stop)
            for layout, (_, _, det) in zip(self.layouts, self.parts, strict=True)
        ]
        return min(lo for lo, _ in bands), max(hi for _, hi in bands)

    def weights(self, index: int, row: int, views: slice) -> np.ndarray:
        """Angular weights on the horizontal line, including its two measured limits."""
        angles, groups, repeats = self.angles
        limits = []
        for (_, beam, det), (first, stop), fan in zip(
            self.parts, self.bounds, self.fans, strict=True
        ):
            pixel = det.du / beam.source_to_detector
            lo, hi = first[row], stop[row]
            limits.append(
                [fan[lo] - pixel / 2, fan[max(0, hi - 1)] + pixel / 2, pixel]
                if hi > lo
                else [1, -1, pixel]
            )
        fan = self.fans[index]
        panel = np.asarray(limits[index])
        order, arc, full = self.ordering
        if not full:
            step = np.median(np.diff(arc))
            delta = (arc[-1] - arc[0] + step - np.pi) / 2
            return (
                parker_weights(order, arc, groups[views], step, fan, delta, panel)
                / repeats[groups[views], None]
            )
        if len(self.parts) == 1:
            own = reliability(fan, panel)
            total = own + reliability(-fan, panel)
            partition = np.divide(own, total, out=np.zeros_like(own), where=total > 0)
            return self.measure[views, None] * partition
        panels = np.repeat(limits, self.counts, axis=0)
        fans = np.repeat(self.fans, self.counts, axis=0)
        return full_turn_weights(
            angles, groups, repeats, fans, panels[:, 2], self.counts, limits=panels
        )[views]


class PermutedRows:
    """A canonical row band read by exact permutations of native detector pixels."""

    def __init__(
        self, fan: FanRows, data: ProjectionBands, native: int, first: int, count: int
    ) -> None:
        self.fan, self.data, self.native, self.first = fan, data, native, first
        self.shape = (len(fan.angles[1]), count, fan.parts[0][2].nu)

    def __getitem__(self, views: slice) -> np.ndarray:
        start = 0
        batches = []
        for layout, count in zip(self.fan.layouts, self.fan.counts, strict=True):
            lo, hi = max(views.start, start), min(views.stop, start + count)
            start += count
            if hi <= lo:
                continue
            data = np.asarray(self.data[lo:hi], np.float32)
            turn = int(layout.roll // 90) % 4
            origin = (self.native, 0, layout.measured.nv - self.native - data.shape[1], 0)[turn]
            first = self.first - origin
            batches.append(np.rot90(data, k=-turn, axes=(1, 2))[:, first : first + self.shape[1]])
        return batches[0] if len(batches) == 1 else np.concatenate(batches)


class FilteredChords:
    """Read and filter view batches of the physical rows a slab needs."""

    def __init__(
        self,
        fan: FanRows,
        data: ProjectionBands,
        native: int,
        first: int,
        stop: int,
        pad_lo: int,
        pad_hi: int,
        name: Filter,
        *,
        cylinder: bool,
    ) -> None:
        self.fan, self.data, self.native, self.first = fan, data, native, first
        self.pad_lo, self.pad_hi, self.name = pad_lo, pad_hi, name
        self.samples: list[RowSamples] = []
        self.fits: list[list[Continuation | None]] = []
        self.radii: list[list[np.ndarray | None]] = []
        self.shape = (len(fan.angles[1]), stop - first, fan.parts[0][2].nu + pad_lo + pad_hi)
        start = 0
        for index, (count, beam, det) in enumerate(fan.parts):
            size = count or self.shape[0]
            u, v = fan.layouts[index].coordinates(det, np.arange(first, stop))
            measured = fan.layouts[index].measured
            samples = RowSamples(np.clip(u, 0, measured.nu - 1), np.clip(v, 0, measured.nv - 1))
            self.samples.append(samples)
            lo, hi = fan.bounds[index]
            fits = []
            for row in range(first, stop):
                width = int(hi[row] - lo[row])
                if not cylinder or not width:
                    fits.append(None)
                    continue
                shift = (lo[row] + (width - 1) / 2 - (det.nu - 1) / 2) * det.du
                own = replace(det, nu=width, center=(det.center[0] + shift, det.center[1]))
                fits.append(Continuation(cone_axis_distance(beam, own, width), width, width))
            self.fits.append(fits)
            self.radii.append(
                self._radii(samples, fits, start, size, lo[first:stop], hi[first:stop])
                if cylinder
                else []
            )
            start += size

    def _radii(
        self,
        samples: RowSamples,
        fits: Sequence[Continuation | None],
        start: int,
        size: int,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> list[np.ndarray | None]:
        # Only fitted edge pixels enter the radius. Read their interpolation
        # neighbours, accumulating mean squares across the scan in bounded batches.
        widths = np.array([f.window if f else 0 for f in fits])
        row = np.repeat(np.arange(len(fits)), widths)
        starts = np.r_[0, np.cumsum(widths)]
        index = np.arange(len(row)) - starts[row]
        edges = np.stack((lo[row] + index, hi[row] - widths[row] + index))
        coordinates = RowSamples(samples.u[row, edges], samples.v[row, edges])
        squares = np.zeros(edges.shape)
        views = radius_views(size)
        for view in views:
            frame = np.asarray(self.data[start + view : start + view + 1], np.float32)
            squares += np.square(coordinates.read(frame, self.native)[0], dtype=np.float64)
        low, high = squares / len(views)
        return [
            np.asarray(f.mean_square_radius(np.asarray, low[None, a:b], high[None, a:b]))
            if f
            else None
            for f, a, b in zip(fits, starts[:-1], starts[1:], strict=True)
        ]

    def __getitem__(self, views: slice) -> np.ndarray:
        data = np.asarray(self.data[views], np.float32)
        result = np.zeros((len(data), *self.shape[1:]), np.float32)
        start = 0
        for index, ((_, beam, det), size) in enumerate(
            zip(self.fan.parts, self.fan.counts, strict=True)
        ):
            lo, hi = max(start, views.start), min(start + size, views.stop)
            start += size
            if hi <= lo:
                continue
            group = slice(lo - views.start, hi - views.start)
            values = self.samples[index].read(data[group], self.native)
            for r in range(self.shape[1]):
                row = self.first + r
                left, right = (side[row] for side in self.fan.bounds[index])
                width = right - left
                if width == 0:
                    continue
                weights = self.fan.weights(index, row, slice(lo, hi))[:, left:right]
                measured = values[:, r, left:right] * weights
                fit = self.fits[index][r]
                pad = width if fit else 0
                if fit is not None:
                    low, high = fit.continued(np.asarray, measured, self.radii[index][r])
                    measured = np.concatenate((low, measured, high), axis=1)
                x = (np.arange(left - pad, right + pad) - (det.nu - 1) / 2) * det.du
                z = (row - (det.nv - 1) / 2) * det.dv
                center, u, v = beam.detector_frame(det)
                ray = center - beam.source()
                normal = abs(ray @ np.cross(u, v))
                cosine = normal / np.sqrt((x + ray @ u) ** 2 + (z + ray @ v) ** 2 + normal**2)
                kernel = get_fbp_filter_np(
                    self.name,
                    det.nu + pad + max(self.pad_lo, self.pad_hi),
                    det.du / beam.magnification,
                    "float32",
                )
                nfft = 2 * (len(kernel) - 1)
                filtered = np.asarray(
                    irfft(rfft(np.asarray(measured * cosine, np.float32), n=nfft) * kernel, n=nfft)
                )
                columns = np.arange(-self.pad_lo, det.nu + self.pad_hi) - (left - pad)
                result[group, r] = (
                    filtered[:, columns % nfft]
                    * (self.fan.parts[0][1].magnification / beam.magnification) ** 2
                )
        return result
