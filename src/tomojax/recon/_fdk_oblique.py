"""FDK on horizontal chords of a rolled flat panel.

Each calibrated arrangement gets an axis-aligned detector in the same plane.
A chord ends where it meets the measured rectangle. Its two edges determine
conjugate coverage and cylinder continuation before the ordinary 1-D ramp.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property, partial
import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.host_transfer import upload_host

from ._continuation import Continuation, cone_axis_distance, radius_views
from ._detector_rows import DetectorRows, RowSamples
from ._fdk_angles import fan_angles, grouped_angles, ordered_arc, parker_weights
from ._fdk_coverage import pair_reliability, reliability
from ._fdk_filter import cosines
from ._row_filter import filter_rows
from .filters import get_fbp_filter_np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Protocol

    from numpy.typing import NDArray

    from tomojax._parameters import Filter
    from tomojax._typed_arrays import Device, Float64Array
    from tomojax.core.geometry import ConeBeam, Detector, Geometry

    from ._fdk_angles import Grouped

    class ProjectionBands(Protocol):
        def __getitem__(self, views: slice, /) -> np.ndarray | jax.Array: ...

    type Parts = tuple[tuple[int | None, ConeBeam, Detector], ...]


@dataclass
class FanRows:
    parts: Parts
    layouts: tuple[DetectorRows, ...]
    angles: Grouped

    @classmethod
    def for_scan(cls, geometry: Geometry, parts: Parts, n_views: int) -> FanRows | None:
        if not any(beam.detector_roll for _, beam, _ in parts):
            return None
        layouts = tuple(DetectorRows(det, beam.detector_roll) for _, beam, det in parts)
        # Equal physical arrays under a reindexing have the same rectified pitch.
        # Pad smaller segment footprints symmetrically to the common array shape.
        horizontal = [layout.horizontal for layout in layouts]
        du, dv = min(d.du for d in horizontal), min(d.dv for d in horizontal)
        nu = max(math.ceil(d.nu * d.du / du - 1e-10) for d in horizontal)
        nv = max(math.ceil(d.nv * d.dv / dv - 1e-10) for d in horizontal)
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
    def bounds(self) -> tuple[tuple[NDArray[np.int64], NDArray[np.int64]], ...]:
        return tuple(
            layout.limits(det, np.arange(det.nv))
            for layout, (_, _, det) in zip(self.layouts, self.parts, strict=True)
        )

    @cached_property
    def fans(self) -> tuple[Float64Array, ...]:
        return tuple(fan_angles(beam, det) for _, beam, det in self.parts)

    @cached_property
    def counts(self) -> tuple[int, ...]:
        return tuple(n or len(self.angles[1]) for n, _, _ in self.parts)

    @cached_property
    def ordering(self) -> tuple[NDArray[np.int64], Float64Array, bool]:
        return ordered_arc(self.angles[0])

    @cached_property
    def measure(self) -> Float64Array:
        angles, groups, repeats = self.angles
        order = np.argsort(angles)
        nodes = angles[order]
        measure = np.empty_like(nodes)
        wrapped = np.concatenate((nodes[-1:] - 2 * np.pi, nodes, nodes[:1] + 2 * np.pi))
        measure[order] = (wrapped[2:] - wrapped[:-2]) / 2
        return measure[groups] / repeats[groups]

    @cached_property
    def equivalent(self) -> tuple[list[int], ...]:
        """Segments measuring the same pixel lattice up to a signed permutation."""
        groups: dict[tuple[ConeBeam, Detector, float], list[int]] = {}
        runs: list[list[int]] = []
        for index, (layout, (_, beam, _)) in enumerate(zip(self.layouts, self.parts, strict=True)):
            native = layout.measured
            turns, angle = divmod(layout.roll, 90)
            if int(turns) % 2:
                native = replace(native, nu=native.nv, nv=native.nu, du=native.dv, dv=native.du)
            group = groups.setdefault((beam, native, angle), [])
            group.append(index)
            runs.append(group)
        return tuple(runs)

    def band(self, first: int, stop: int) -> tuple[int, int]:
        bands = [
            layout.band(det, first, stop)
            for layout, (_, _, det) in zip(self.layouts, self.parts, strict=True)
        ]
        return min(lo for lo, _ in bands), max(hi for _, hi in bands)

    def weights(self, index: int, row: int, views: slice) -> np.ndarray:
        """Angular weights on the horizontal line, including its two measured limits."""
        angles, groups, repeats = self.angles
        limits: list[list[float]] = []
        for (_, beam, det), (first, stop), fan in zip(
            self.parts, self.bounds, self.fans, strict=True
        ):
            pixel = det.du / beam.source_to_detector
            lo, hi = int(first.item(row)), int(stop.item(row))
            limits.append(
                [fan.item(lo) - pixel / 2, fan.item(max(0, hi - 1)) + pixel / 2, pixel]
                if hi > lo
                else [1, -1, pixel]
            )
        fan = self.fans[index]
        panel = np.asarray(limits[index])
        order, arc, full = self.ordering
        if not full:
            step = float(np.median(np.diff(arc)))
            delta = (arc.item(-1) - arc.item(0) + step - np.pi) / 2
            pair = None
            if len(self.parts) > 1:
                parameters = np.repeat(limits, self.counts, axis=0)
                pair = pair_reliability(
                    angles, groups, repeats, fan[None, :], parameters, groups[views]
                )
            return (
                parker_weights(order, arc, groups[views], step, fan, delta, panel, pair)
                / repeats[groups[views], None]
            )
        own = reliability(fan, panel)
        if len(self.parts) == 1:
            total = own + reliability(-fan, panel)
        else:
            current, opposite = pair_reliability(
                angles,
                groups,
                repeats,
                fan[None, :],
                np.repeat(limits, self.counts, axis=0),
                groups[views],
            )
            total = current + opposite
        partition = np.divide(own, total, out=np.zeros_like(total), where=total > 0)
        return self.measure[views, None] * partition


class PermutedRows:
    """A canonical row band read by exact permutations of native detector pixels."""

    def __init__(
        self, fan: FanRows, data: ProjectionBands, native: int, first: int, count: int
    ) -> None:
        self.fan, self.data, self.native, self.first = fan, data, native, first
        self.shape = (len(fan.angles[1]), count, fan.parts[0][2].nu)

    def __getitem__(self, views: slice) -> np.ndarray:
        start = 0
        batches: list[np.ndarray] = []
        first_view, last_view, _ = views.indices(self.shape[0])
        for layout, count in zip(self.fan.layouts, self.fan.counts, strict=True):
            lo, hi = max(first_view, start), min(last_view, start + count)
            start += count
            if hi <= lo:
                continue
            data = np.asarray(self.data[lo:hi], np.float32)
            turn = int(layout.roll // 90) % 4
            origin = (self.native, 0, layout.measured.nv - self.native - data.shape[1], 0)[turn]
            first = self.first - origin
            batches.append(np.rot90(data, k=-turn, axes=(1, 2))[:, first : first + self.shape[1]])
        return batches[0] if len(batches) == 1 else np.concatenate(batches)


@partial(jax.jit, static_argnames=("pad_lo", "pad_hi"))
def _filter_chords(
    frames: jax.Array,
    coordinates: jax.Array,
    angular: jax.Array,
    measure: jax.Array,
    cosine: jax.Array,
    kernel: jax.Array,
    coefficients: jax.Array | None,
    tails: jax.Array | None,
    *,
    pad_lo: int,
    pad_hi: int,
) -> jax.Array:
    """Interpolate, continue and ramp-filter a batch of horizontal lines on its device."""

    def taps(coord: jax.Array, size: int) -> tuple[jax.Array, jax.Array]:
        lower = jnp.floor(coord).astype(jnp.int32)
        t = coord - lower
        weights = jnp.stack(
            (
                -0.5 * t * (1 - t) ** 2,
                1 - t * t * (2.5 - 1.5 * t),
                t * (0.5 + t * (2 - 1.5 * t)),
                -0.5 * t * t * (1 - t),
            )
        )
        indices = lower + jnp.arange(-1, 3)[:, None, None]
        return jnp.clip(indices, 0, size - 1), weights

    u, wx = taps(coordinates[0], frames.shape[2])
    v, wy = taps(coordinates[1], frames.shape[1])
    rows = jnp.zeros((frames.shape[0], *u.shape[1:]), frames.dtype)
    for i in range(4):
        for j in range(4):
            rows += frames[:, v[i], u[j]] * (wy[i] * wx[j])
    rows *= angular * measure[:, None, None]
    nu = rows.shape[-1]
    if coefficients is not None and tails is not None:
        low = (rows * coefficients[0]).sum(axis=-1, keepdims=True)
        high = (rows * coefficients[1]).sum(axis=-1, keepdims=True)
        rows = jnp.pad(rows, ((0, 0), (0, 0), (nu, nu))) + low * tails[0] + high * tails[1]
    return filter_rows(rows * cosine, kernel, nu, pad_lo, pad_hi)


class FilteredChords:
    """Filter the measured horizontal chords a volume slab needs in bounded view batches."""

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
        self.fan, self.data, self.native = fan, data, native
        self.first, self.stop = first, stop
        self.pad_lo, self.pad_hi, self.name, self.cylinder = pad_lo, pad_hi, name, cylinder
        self.starts: NDArray[np.int64] = np.concatenate(([0], np.cumsum(fan.counts)))

    def panel(self, device: Device, index: int) -> tuple[jax.Array, ...]:
        _, beam, det = self.fan.parts[index]
        layout = self.fan.layouts[index]
        vertical = np.arange(self.first, self.stop)
        u, v = layout.coordinates(det, vertical)
        samples = RowSamples(
            np.clip(u, 0, layout.measured.nu - 1), np.clip(v, 0, layout.measured.nv - 1)
        )
        # Round the full-detector coordinates once. Cropping before the float32
        # conversion changes the cubic fractions in a slab.
        coordinates = np.stack(
            (samples.u.astype(np.float32), samples.v.astype(np.float32) - self.native)
        )
        lo, hi = (side[self.first : self.stop] for side in self.fan.bounds[index])
        mask = (np.arange(det.nu) >= lo[:, None]) & (np.arange(det.nu) < hi[:, None])
        angular = mask.astype(np.float32)
        if len(self.fan.parts) == 1 and self.fan.ordering[2]:
            one = slice(0, 1)
            for r, row in enumerate(range(self.first, self.stop)):
                angular[r] *= self.fan.weights(index, row, one)[0] / self.fan.measure[0]
        pad = det.nu if self.cylinder else 0
        wide = replace(det, nu=det.nu + 2 * pad)
        cosine = cosines(((None, beam, wide),), 1)[self.first : self.stop]
        scale = (self.fan.parts[0][1].magnification / beam.magnification) ** 2
        kernel = (
            get_fbp_filter_np(
                self.name,
                det.nu + pad + max(self.pad_lo, self.pad_hi),
                det.du / beam.magnification,
                "float32",
            )
            * scale
        )
        arrays: list[np.ndarray] = [coordinates, angular, cosine, kernel]
        if self.cylinder:
            fits: list[Continuation | None] = []
            for row in range(len(lo)):
                left, right = lo.item(row), hi.item(row)
                width = right - left
                if width:
                    shift = (left + (width - det.nu) / 2) * det.du
                    chord = replace(det, nu=width, center=(det.center[0] + shift, det.center[1]))
                    fits.append(Continuation(cone_axis_distance(beam, chord, width), width, width))
                else:
                    fits.append(None)
            radii = self._radii(index, fits, lo, hi)
            coeff = np.zeros((2, len(vertical), det.nu), np.float32)
            tails = np.zeros((2, len(vertical), wide.nu), np.float32)
            for r, (fit, radius) in enumerate(zip(fits, radii, strict=True)):
                if fit is None or radius is None:
                    continue
                left, right = lo.item(r), hi.item(r)
                basis = np.eye(fit.window)
                low, high = fit.amplitudes(np.asarray, basis, basis, radius)
                coeff[0, r, left : left + fit.window] = low.ravel()
                coeff[1, r, right - fit.window : right] = high.ravel()
                low, high = fit.tails(
                    np.asarray, np.ones_like(radius), np.ones_like(radius), radius
                )
                tails[0, r, pad + left - fit.pad : pad + left] = low.ravel()
                tails[1, r, pad + right : pad + right + fit.pad] = high.ravel()
            arrays.extend((coeff, tails))
        return tuple(upload_host(np.asarray(a, np.float32), device) for a in arrays)

    def _radii(
        self,
        segment: int,
        fits: Sequence[Continuation | None],
        lo: NDArray[np.int64],
        hi: NDArray[np.int64],
    ) -> list[np.ndarray | None]:
        # Only fitted edge pixels enter the radius. Read their interpolation
        # neighbours, accumulating mean squares across the scan in bounded batches.
        widths = np.array([f.window if f else 0 for f in fits])
        row = np.repeat(np.arange(len(fits)), widths)
        starts: NDArray[np.int64] = np.concatenate(([0], np.cumsum(widths)))
        index = np.arange(len(row)) - starts[row]
        edges = np.stack((lo[row] + index, hi[row] - widths[row] + index))
        det = self.fan.parts[segment][2]
        x = (edges.astype(np.float64) - (det.nu - 1) / 2) * det.du
        z = (row.astype(np.float64) + self.first - (det.nv - 1) / 2) * det.dv
        runs = self.fan.equivalent[segment]
        views = radius_views(sum(self.fan.counts[i] for i in runs))
        squares: Float64Array = np.zeros(edges.shape)
        first = 0
        for run in runs:
            size = self.fan.counts[run]
            layout = self.fan.layouts[run]
            u, v = layout.pixel_coordinates(x, z)
            coordinates = RowSamples(
                np.clip(u, 0, layout.measured.nu - 1), np.clip(v, 0, layout.measured.nv - 1)
            )
            selected = (
                views[(views >= first) & (views < first + size)] + self.starts.item(run) - first
            )
            for i in range(len(selected)):
                view = selected.item(i)
                frame = np.asarray(self.data[view : view + 1], np.float32)
                squares += np.square(coordinates.read(frame, self.native)[0], dtype=np.float64)
            first += size
        low, high = squares / len(views)
        radii: list[np.ndarray | None] = []
        for i, fit in enumerate(fits):
            band = slice(starts.item(i), starts.item(i + 1))
            radii.append(
                np.asarray(fit.mean_square_radius(np.asarray, low[None, band], high[None, band]))
                if fit
                else None
            )
        return radii

    def read(
        self, first: int, stop: int, device: Device, panel: Callable[[int], tuple[jax.Array, ...]]
    ) -> jax.Array:
        batches: list[jax.Array] = []
        for index in range(
            int(np.searchsorted(self.starts, first, side="right")) - 1,
            int(np.searchsorted(self.starts, stop)),
        ):
            lo, hi = max(first, self.starts.item(index)), min(stop, self.starts.item(index + 1))
            views = slice(lo, hi)
            coords, mask, cosine, kernel, *fit = panel(index)
            if len(self.fan.parts) == 1 and self.fan.ordering[2]:
                angular, measure = mask, self.fan.measure[views]
            else:
                weights = np.stack(
                    [self.fan.weights(index, row, views) for row in range(self.first, self.stop)],
                    axis=1,
                )
                angular = upload_host(weights.astype(np.float32), device) * mask
                measure = np.ones(hi - lo, np.float32)
            data = self.data[views]
            frames = (
                jax.device_put(data.astype(np.float32), device)
                if isinstance(data, jax.Array)
                else upload_host(np.asarray(data, np.float32), device)
            )
            batches.append(
                _filter_chords(
                    frames,
                    coords,
                    angular,
                    upload_host(measure.astype(np.float32), device),
                    cosine,
                    kernel,
                    fit[0] if fit else None,
                    fit[1] if fit else None,
                    pad_lo=self.pad_lo,
                    pad_hi=self.pad_hi,
                )
            )
        return batches[0] if len(batches) == 1 else jnp.concatenate(batches)
