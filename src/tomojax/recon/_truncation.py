"""What a scan's rows say about an object that extends outside the field of view.

The analytic methods continue each detector row past the detector's edges as
the projection of a cylinder about the rotation axis
(:mod:`tomojax.recon._continuation`). :func:`truncation_record` measures, from
the same fits, how far the rows run past the detector and how much an interior
value rests on the continuation. The methods that fit a grid take the data as
measured: they are exact when the grid holds the object, and the record says
when the rows do not end inside the detector.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math
from typing import TYPE_CHECKING

import numpy as np

from tomojax.core.geometry.cone import cone_parts
from tomojax.geometry.coverage import conjugate_weights, full_parallel_turn, panel_field

from ._continuation import (
    Continuation,
    cone_axis_distance,
    continued_columns,
    parallel_axis_distance,
    radius_views,
)
from ._detector_rows import DetectorRows, RowSamples

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax
    import numpy.typing as npt

    from tomojax.geometry import Detector, Geometry, ScanGeometry

    from ._continuation import Host

# Views and rows sampled for the record.
_RECORD_SAMPLES = 64


def axis_distance(geometry: Geometry, detector: Detector, pad: int) -> Host:
    """Distance from the rotation axis of each ray of a row continued by ``pad`` columns a side."""
    parts = cone_parts(geometry, detector)
    if parts is None:
        return parallel_axis_distance(detector, detector.nu, pad)
    return cone_axis_distance(parts[0][1], parts[0][2], pad)


def _record_indices(views: int, nv: int) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    view = radius_views(views)[:: max(1, math.ceil(min(views, 256) / _RECORD_SAMPLES))]
    row = np.linspace(0, nv - 1, min(nv, _RECORD_SAMPLES)).round().astype(np.int64)
    return view, row


@dataclass
class _Chord:
    samples: RowSamples
    views: npt.NDArray[np.int64]
    native: tuple[int, int]
    distance: bytes
    z: float
    values: np.ndarray | None = None


class ChordSample:
    """Sample physical chords for the truncation record, including native slab reads."""

    def __init__(self, chords: list[_Chord], field_radius: float, *, full_turn: bool) -> None:
        self.full_turn = full_turn
        self.chords, self.field_radius = chords, field_radius

    @classmethod
    def for_band(
        cls, geometry: Geometry, detector: Detector, n_views: int, first: int, height: int
    ) -> ChordSample | None:
        parts = cone_parts(geometry, detector)
        parallel_roll = float(getattr(geometry, "detector_roll", 0.0))
        if not parallel_roll and not any(b.detector_roll for _, b, _ in parts or ()):
            return None
        full_turn = full_parallel_turn(geometry)
        chosen, _ = _record_indices(n_views, height)
        chords: list[_Chord] = []
        field: dict[tuple[float, bytes], float] = {}
        start = 0
        for count, beam, own in parts or ((n_views, None, detector),):
            stop = start + (count or n_views)
            views = chosen[(chosen >= start) & (chosen < stop)]
            start = stop
            if not len(views):
                continue
            layout = DetectorRows(own, parallel_roll if beam is None else beam.detector_roll)
            horizontal = layout.horizontal
            _, global_rows = _record_indices(n_views, horizontal.nv)
            lower, upper = layout.native_bounds(horizontal, np.arange(horizontal.nv))
            # A file window does not move the panel or create new cylinder
            # edges. Sample complete physical chords supported by this window.
            available = np.flatnonzero((lower >= first) & (upper <= first + height))
            _, selected = _record_indices(n_views, len(available))
            local = available[selected]
            rows = np.union1d(global_rows, local)
            u, v = layout.coordinates(horizontal, rows)
            lo, hi = layout.limits(horizontal, rows)
            for i in range(len(rows)):
                row = rows.item(i)
                left, right = lo.item(i), hi.item(i)
                width = right - left
                if not width:
                    continue
                x = horizontal.center[0] + (left + (width - horizontal.nu) / 2) * horizontal.du
                panel = replace(horizontal, nu=width, center=(x, horizontal.center[1]))
                distance = (
                    parallel_axis_distance(panel, width, width)
                    if beam is None
                    else cone_axis_distance(beam, panel, width)
                )
                z = horizontal.center[1] + (row - (horizontal.nv - 1) / 2) * horizontal.dv
                coordinates = distance.tobytes()
                if row in global_rows:
                    field[z, coordinates] = _field_radius(
                        distance, width, outer_only=full_turn and x != 0
                    )
                if row in local:
                    sample = RowSamples(
                        np.clip(u[i, left:right], 0, own.nu - 1),
                        np.clip(v[i, left:right], 0, own.nv - 1) - first,
                    )
                    band = lower.item(row) - first, upper.item(row) - first
                    chords.append(_Chord(sample, views, band, coordinates, z))
        return cls(
            chords, float(np.median(list(field.values()))) if field else 0.0, full_turn=full_turn
        )

    def add(self, first: int, data: np.ndarray | jax.Array) -> None:
        pending: dict[bytes, list[_Chord]] = {}
        for chord in self.chords:
            lo, hi = chord.native
            if chord.values is None and first <= lo and hi <= first + data.shape[1]:
                pending.setdefault(chord.views.tobytes(), []).append(chord)
        for group in pending.values():
            # One gather per tap covers every sampled chord in this native band.
            samples = RowSamples(
                np.concatenate([c.samples.u for c in group]),
                np.concatenate([c.samples.v for c in group]),
            )
            views = group[0].views
            blocks = [samples.read(data, first, views[i : i + 4]) for i in range(0, len(views), 4)]
            values = np.concatenate(blocks)
            at = 0
            for chord in group:
                stop = at + len(chord.samples.u)
                chord.values = values[:, at:stop]
                at = stop

    def values(self) -> np.ndarray:
        # Pool repeats of each physical chord across acquisition segments.
        if not self.chords:
            return np.array([0.0, 0.0, 0.0, 0.0, self.field_radius])
        groups: dict[tuple[float, bytes], list[np.ndarray]] = {}
        for chord in self.chords:
            assert chord.values is not None
            groups.setdefault((chord.z, chord.distance), []).append(chord.values)
        stats: list[np.ndarray] = []
        axial: list[float] = []
        for (_, coordinates), pieces in groups.items():
            data: npt.NDArray[np.float32] = np.concatenate(pieces, dtype=np.float32)[:, None, :]
            distance = np.frombuffer(coordinates, np.float64)
            width = data.shape[-1]
            outer_only = self.full_turn and distance.item(width) != -distance.item(2 * width - 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                terms: Host = np.asarray(
                    _measure(coordinates, host=True, by_row=True, outer_only=outer_only)(data),
                    np.float64,
                ).reshape(7)
            if distance.item(width) <= 0 <= distance.item(2 * width - 1):
                axial.append(terms.item(4))
            stats.append(terms)
        rows = np.stack(stats)
        # Only chords containing the axis measure an interior change there.
        rows[:, 4] = np.mean(axial) if axial else 0.0
        rows[:, 6] = self.field_radius
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.asarray(_combine(rows))

    def record(self, read: Callable[[int, int], np.ndarray | jax.Array]) -> dict[str, float]:
        for chord in self.chords:
            if chord.values is None:
                lo, hi = chord.native
                self.add(lo, read(lo, hi))
        return record_values(self.values())


class ScanSample:
    """Collect each globally sampled detector row once, including overlapping slab footprints."""

    def __init__(self, geometry: ScanGeometry, detector: Detector) -> None:
        distance = _axis_distance(geometry, detector, continued_columns(detector.nu))
        self.coordinates = distance.tobytes()
        self.outer_only = bool(detector.center[0]) and full_parallel_turn(geometry)
        self.views, self.rows = _record_indices(len(geometry.angles), detector.nv)
        self.statistics = np.empty((len(self.rows), 7), np.float64)
        self.present = np.zeros(len(self.rows), bool)

    def add(self, first: int, data: np.ndarray | jax.Array) -> None:
        indices = np.flatnonzero(
            ~self.present & (self.rows >= first) & (self.rows < first + data.shape[1])
        )
        if indices.size:
            selected = self.rows[indices] - first
            if isinstance(data, np.ndarray):
                self.statistics[indices] = _measure(
                    self.coordinates, outer_only=self.outer_only, host=True, by_row=True
                )(data[self.views[:, None], selected])
            else:
                import jax

                with jax.enable_x64(True):
                    self.statistics[indices] = np.asarray(
                        _measure(
                            self.coordinates, outer_only=self.outer_only, host=False, by_row=True
                        )(data[self.views[:, None], selected])
                    )
            self.present[indices] = True

    def record(self, read: Callable[[int, int], np.ndarray | jax.Array]) -> dict[str, float]:
        """Read missing rows for a restricted grid or resumed run and reduce the complete sample."""
        for row in self.rows[~self.present]:
            self.add(int(row), read(int(row), int(row + 1)))
        with np.errstate(invalid="ignore", divide="ignore"):
            return record_values(_combine(self.statistics))


def truncation_record(
    geometry: Geometry, detector: Detector, projections: np.ndarray | jax.Array
) -> dict[str, float]:
    """How far a scan's rows run past the detector, and what the continuation does about it.

    Parameters
    ----------
    geometry : Geometry
        Scan geometry.
    detector : Detector
        Detector of the projections.
    projections : numpy.ndarray or jax.Array
        Line integrals, shape ``(views, nv, nu)``.

    Returns
    -------
    dict
        ``edge_ratio``: the mean line integral at the detector's edges over the
        mean of the rows, zero for an object inside the field of view.
        ``outside_fraction``: the share of the continued scan's attenuation
        that lies beyond the detector, in the continuation and not the
        measurement. ``interior_shift``: the change the continuation makes to
        the value on the rotation axis, in the volume's units (per length);
        negative, as zero beyond the detector counts outside material inside.
        ``radius``: the median radius of the rows' fitted cylinders, and
        ``field_of_view_radius``: the distance from the axis to the nearer
        detector edge, or to the far edge on an offset axial parallel full turn.
        A rolled detector reports the median over sampled horizontal chords.
        Both radii use the scan's length unit.

    Notes
    -----
    Computed from at most 64 evenly spaced views and rows. What lies beyond
    the detector is a model: interior values carry an unknown smooth
    background of the order of a fraction of ``interior_shift``, while edges,
    contrast between neighboring regions and fine detail do not depend on it.
    """
    return record_values(truncation_values(geometry, detector, projections))


def record_values(values: np.ndarray | jax.Array) -> dict[str, float]:
    """Collect the five scalar diagnostics after their projection band has completed."""
    keys = ("edge_ratio", "outside_fraction", "interior_shift", "radius", "field_of_view_radius")
    return dict(zip(keys, map(float, np.asarray(values)), strict=True))


def truncation_values(
    geometry: Geometry, detector: Detector, projections: np.ndarray | jax.Array, *, first: int = 0
) -> np.ndarray | jax.Array:
    """Submit the record on the source device; only the five scalars travel to the host."""
    sample = ChordSample.for_band(geometry, detector, len(projections), first, projections.shape[1])
    if sample is not None:
        sample.add(0, projections)
        return sample.values()
    distance = axis_distance(geometry, detector, continued_columns(projections.shape[-1]))
    host = isinstance(projections, np.ndarray)
    measure = _measure(
        distance.tobytes(),
        host=host,
        outer_only=bool(detector.center[0]) and full_parallel_turn(geometry),
    )
    if host:
        with np.errstate(invalid="ignore", divide="ignore"):
            return measure(projections)
    import jax

    with jax.enable_x64(True):
        values = measure(projections)
        jax.copy_to_host_async(values)
        return values


def _combine(rows: np.ndarray | jax.Array) -> np.ndarray | jax.Array:
    if isinstance(rows, np.ndarray):
        xp = np
    else:
        import jax.numpy as xp

    edge, mean, measured, continued, shift, radius, fov = (rows[:, i] for i in range(7))
    total = measured.sum() + continued.sum()
    return xp.stack(
        (
            xp.where(mean.sum() != 0, edge.sum() / mean.sum(), 0),
            xp.where(total != 0, continued.sum() / total, 0),
            shift.mean(),
            xp.median(radius),
            fov[0],
        )
    )


def _field_radius(distance: Host, nu: int, *, outer_only: bool) -> float:
    left, right = distance.item(nu), distance.item(2 * nu - 1)
    if outer_only:
        pitch = distance.item(nu) - distance.item(nu - 1)
        return pitch * panel_field(nu, (left + right) / (2 * pitch))[1]
    return min(abs(left), abs(right))


@lru_cache(maxsize=16)
def _measure(
    coordinates: bytes, *, host: bool, outer_only: bool, by_row: bool = False
) -> Callable[[np.ndarray | jax.Array], np.ndarray | jax.Array]:
    """Reduce sampled rows, returning either their individual terms or the scan record."""
    distance: Host = np.frombuffer(coordinates, np.float64)
    nu = len(distance) // 3
    pad = continued_columns(nu)
    fits = Continuation(distance, nu, pad, outer_only=outer_only)
    width: Host = np.abs(np.gradient(distance))
    outside = np.r_[0:pad, pad + nu : pad + nu + pad]
    fov = _field_radius(distance, nu, outer_only=outer_only)
    measured_width = width[pad : pad + nu]
    if outer_only:
        offset = (distance.item(pad) + distance.item(pad + nu - 1)) / (2 * measured_width.item(0))
        measured_width = measured_width * conjugate_weights(nu, offset)
    # An axis outside a panel can coincide with a padded pixel. Only rays
    # outside the complete field contribute to its continuation on a full turn.
    tail_width = width[outside]
    shift_weight = np.divide(
        tail_width,
        distance[outside] ** 2,
        out=np.zeros_like(tail_width),
        where=distance[outside] != 0,
    )
    if host:
        xp = np
    else:
        import jax
        import jax.numpy as xp

    def measure(data: np.ndarray | jax.Array) -> np.ndarray | jax.Array:
        if not by_row:
            view, row = _record_indices(*data.shape[:2])
            data = data[view[:, None], row]
        sample = data.astype(xp.float64).swapaxes(0, 1)
        low, high = sample[..., : fits.window], sample[..., nu - fits.window :]
        radius2 = fits.radius(xp.asarray, low, high)
        low, high = fits.amplitudes(xp.asarray, low, high, radius2)
        low, high = low.mean(axis=1, keepdims=True), high.mean(axis=1, keepdims=True)
        before, after = fits.tails(xp.asarray, low, high, radius2)
        tails = xp.concatenate((before, after), axis=-1)
        measured = xp.sum(sample * xp.asarray(measured_width), axis=(1, 2))
        continued = xp.sum(tails * xp.asarray(width[outside]), axis=(1, 2)) * data.shape[0]
        on_axis = (tails * xp.asarray(shift_weight)).sum(axis=(1, 2))
        statistics = xp.stack(
            (
                (low[:, 0, 0] + high[:, 0, 0]) / 2,
                sample.mean(axis=(1, 2)),
                measured,
                continued,
                -on_axis / (2 * np.pi),
                radius2[:, 0, 0] ** 0.5 * fits.scale,
                xp.ones(measured.shape) * fov,
            ),
            axis=1,
        )

        return statistics if by_row else _combine(statistics)

    return measure if host else jax.jit(measure)
