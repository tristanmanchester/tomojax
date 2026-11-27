"""Continue detector rows past the detector's edges.

A scan of an object wider than the detector ends every row on a nonzero line
integral. Treating the unmeasured rays as zero puts a step at each edge, and
the ramp filter turns the step into a bright rim and a bowl across the field
of view. The unmeasured rays cannot be recovered, but most of their effect on
the interior comes from the rays nearest the edge, which the rows' own edges
predict.

Each detector row is continued as the projection of a uniform cylinder about
the rotation axis, ``c sqrt(R**2 - s**2)`` at distance ``s`` from the axis.
The row has one radius ``R``, fitted to the fall of its views towards both
edges; each view's two continuations take their amplitudes ``c`` from that
view's own edge values. A row that reaches its edges at zero continues as
zero. :class:`Continuation` holds the fits.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax
    import numpy.typing as npt

    from tomojax.core.cuda_types import CuPyValue
    from tomojax.geometry import ConeBeam, Detector

# Rows of line integrals as the caller holds them: a NumPy, JAX or CuPy array.
type Rows = np.ndarray | jax.Array | CuPyValue
# Host coordinates and coefficients of the fits.
type Host = npt.NDArray[np.float64]
# Makes one of the rows' arrays from a host array.
type Place = Callable[[Host], Rows]

# Views that share in a row's radius: enough to average the sample's structure
# at the edges, few enough to read from a file-backed stack at no cost.
_RADIUS_VIEWS = 256


def continued_columns(nu: int) -> int:
    """Columns added to each side of an ``nu``-column row: one detector width.

    The continuation ends within them, so the completed row is at most three
    times the measured width.
    """
    return int(nu)


def radius_views(n_views: int, order: npt.NDArray[np.int64] | None = None) -> npt.NDArray[np.int64]:
    """Indices of the evenly spaced views that share in each row's radius.

    ``order`` gives the scan's view at each position of a reordered stack; the
    result then indexes that stack.
    """
    count = min(n_views, _RADIUS_VIEWS)
    views = np.unique(np.linspace(0, n_views - 1, count).round().astype(np.int64))
    return views if order is None else np.flatnonzero(np.isin(order, views))


def parallel_axis_distance(detector: Detector, nu: int, pad: int) -> Host:
    """Signed distance from the rotation axis of each ray of a continued parallel-beam row.

    ``nu`` measured columns centered in ``detector``'s frame with ``pad`` more on
    each side; the result has length ``nu + 2 * pad``.
    """
    columns = np.arange(nu + 2 * pad, dtype=np.float64) - pad - (nu - 1) / 2
    return columns * float(detector.du) + float(detector.center[0])


def cone_axis_distance(beam: ConeBeam, detector: Detector, pad: int) -> Host:
    """Signed distance from the rotation axis of each ray of a continued cone-beam row.

    A column at ``u`` on the detector is the ray at fan angle ``gamma`` from the
    ray through the axis, which passes ``hypot(source_to_axis, axis_offset) *
    sin(gamma)`` from the axis in every row's fan.
    """
    u = parallel_axis_distance(detector, detector.nu, pad)
    sad, offset = float(beam.source_to_axis), float(beam.axis_offset)
    gamma: Host = np.arctan(u / float(beam.source_to_detector)) - np.arctan(offset / sad)
    distance: Host = np.hypot(sad, offset) * np.sin(gamma)
    return distance


# Below any squared signal, in single precision too.
_TINY = 1e-30


def continued_width(nu: int) -> int:
    """Columns of an ``nu``-column row with its continuations."""
    return nu + 2 * continued_columns(nu)


def row_fits(detector: Detector, *, outer_only: bool = False) -> Continuation:
    """The fits of a parallel-beam ``detector``'s rows, continued one detector width a side."""
    pad = continued_columns(detector.nu)
    return Continuation(
        parallel_axis_distance(detector, detector.nu, pad), detector.nu, pad, outer_only=outer_only
    )


class Continuation:
    """The cylinder fits of rows of ``nu`` columns continued by ``pad`` columns a side.

    Parameters
    ----------
    distance : numpy.ndarray
        Signed distance from the rotation axis of each ray of the continued
        row, shape ``(nu + 2 * pad,)`` (see :func:`parallel_axis_distance`).
    nu : int
        Measured columns of a row.
    pad : int
        Columns the continuation adds on each side, at least one.

    Notes
    -----
    Both fits use the outermost sixteenth of the row at each edge
    (:attr:`window` columns). Squared radii are in units of :attr:`scale`, the
    distance of the farthest continued ray, which keeps single-precision sums
    well scaled.

    The methods serve NumPy, CuPy and JAX arrays alike. Each takes ``place``,
    which makes one of the rows' arrays (and precision) from a host array, and
    otherwise uses only array arithmetic and methods; numbers reach the arrays
    through ``place`` too, because CuPy's operators import its sparse package
    for a Python number.
    """

    def __init__(self, distance: Host, nu: int, pad: int, *, outer_only: bool = False) -> None:
        self.outer_only = outer_only
        self.nu, self.pad = nu, pad
        self.distance = distance
        self.scale = float(np.max(abs(distance)))
        s: Host = np.asarray(distance, np.float64) / self.scale
        window = min(nu, max(2, nu // 16))
        self.window = window
        # Per side: fitted distances, the edge column's distance, the added columns' distances.
        self.sides: tuple[tuple[Host, float, Host], tuple[Host, float, Host]] = (
            (s[pad : pad + window], s.item(pad), s[:pad]),
            (s[pad + nu - window : pad + nu], s.item(pad + nu - 1), s[pad + nu :]),
        )
        # R**2 lies between just outside the farther edge and the last added column.
        self._widest = float(np.max(s * s))
        self._nearest = max(
            edge * edge + (s.item(column) - edge) ** 2
            for (_, edge, _), column in zip(self.sides, (pad - 1, pad + nu), strict=True)
        )

    def radius(self, place: Place, low: Rows, high: Rows) -> Rows:
        """Squared radius of each row's cylinder.

        ``low`` and ``high`` are the first and last :attr:`window` columns of
        the views that share in the radius (see :func:`radius_views`), shape
        ``(..., views, window)``; the result has shape ``(..., 1, 1)``.

        A cylinder's projection satisfies ``p**2 = c**2 (R**2 - s**2)``: a
        straight line in ``s**2`` that crosses zero at ``R**2``. One
        least-squares line through the squared values of every view's fitted
        columns, at both edges, gives the row's ``R``. Views weigh in by their
        squared values, so views that end at zero do not count. A row that is
        level or rising towards its edges takes the widest cylinder the added
        columns hold.
        """
        return self.mean_square_radius(
            place,
            (low * low).mean(axis=-2, keepdims=True),
            (high * high).mean(axis=-2, keepdims=True),
        )

    def mean_square_radius(self, place: Place, low: Rows, high: Rows) -> Rows:
        """:meth:`radius` from the sharing views' mean squared values, ``(..., 1, window)``."""
        first, last = self.sides[0][0] ** 2, self.sides[1][0] ** 2
        if self.outer_only:
            if abs(self.sides[0][1]) < abs(self.sides[1][1]):
                first, low = last, high
            else:
                last, high = first, low
        count = first.size + last.size
        mean = float(np.sum(first) + np.sum(last)) / count
        spread = float(np.sum((first - mean) ** 2) + np.sum((last - mean) ** 2))
        # Each side's weights for the line's slope, and for its value at s = 0: the mean
        # of the squares plus the slope times the mean s**2.
        slopes = ((first - mean) / max(spread, _TINY), (last - mean) / max(spread, _TINY))
        zeros = (1 / count - mean * slopes[0], 1 / count - mean * slopes[1])
        slope = -(
            (low * place(slopes[0])).sum(axis=-1, keepdims=True)
            + (high * place(slopes[1])).sum(axis=-1, keepdims=True)
        )
        level = (low * place(zeros[0])).sum(axis=-1, keepdims=True) + (high * place(zeros[1])).sum(
            axis=-1, keepdims=True
        )
        # level / slope where the row falls, and the widest cylinder where it is level or rises.
        radius2 = level.clip(self._widest * _TINY, None) / slope.clip(_TINY, None)
        return radius2.clip(self._nearest, self._widest)

    def amplitudes(self, place: Place, low: Rows, high: Rows, radius2: Rows) -> tuple[Rows, Rows]:
        """The continuations' amplitudes at the first and last columns of rows.

        ``low`` and ``high`` are the rows' first and last :attr:`window`
        columns, shape ``(..., window)``, and ``radius2`` broadcasts against
        ``low[..., :1]``; each result has shape ``(..., 1)``.

        Each is a straight line through that edge's fitted values, every value
        first divided by the cylinder's fall from its column to the edge:
        exact for the cylinder, and linear in the values, so noise and a single
        bad edge pixel average out and a row of noise about zero continues as
        zero on average.
        """
        half = place(np.asarray(0.5))
        amplitudes: list[Rows] = []
        for values, (fit, edge, _) in zip((low, high), self.sides, strict=True):
            drop = radius2 - place(np.asarray(edge * edge))  # R**2 - s**2 at the edge, positive
            inside = place(np.maximum(edge * edge - fit * fit, 0))
            middle = float(np.mean(fit))
            spread = max(float(np.sum((fit - middle) ** 2)), _TINY)
            line = place(1 / self.window + (fit - middle) * (edge - middle) / spread)
            flattened = values * (drop / (drop + inside)) ** half * line
            amplitudes.append(flattened.sum(axis=-1, keepdims=True))
        if self.outer_only:
            # Only the farther edge bounds measured rays over the complete turn.
            far = int(abs(self.sides[1][1]) > abs(self.sides[0][1]))
            amplitudes[far] = amplitudes[far] * place(np.asarray(2.0))
            amplitudes[1 - far] = amplitudes[1 - far] * place(np.asarray(0.0))
        return amplitudes[0], amplitudes[1]

    def tails(self, place: Place, low: Rows, high: Rows, radius2: Rows) -> tuple[Rows, Rows]:
        """The ``pad`` columns before and after rows whose edge amplitudes are ``low`` and ``high``.

        Each is the cylinder's profile scaled to its amplitude. Between an edge
        and the axis (an edge on the axis side of an offset detector) the
        continuation is level.
        """
        half = place(np.asarray(0.5))
        tails: list[Rows] = []
        for amplitude, (_, edge, beyond) in zip((low, high), self.sides, strict=True):
            drop = radius2 - place(np.asarray(edge * edge))
            reach = place(np.maximum(beyond * beyond, edge * edge))
            tails.append(amplitude * ((radius2 - reach) / drop).clip(0.0, 1.0) ** half)
        return tails[0], tails[1]

    def continued(self, place: Place, rows: Rows, radius2: Rows) -> tuple[Rows, Rows]:
        """The ``pad`` columns before and after ``rows``, ``(..., nu)``, of radius ``radius2``.

        ``radius2`` is each row's :meth:`radius`, shaped to broadcast against
        ``rows[..., :1]``.
        """
        low, high = rows[..., : self.window], rows[..., self.nu - self.window :]
        return self.tails(place, *self.amplitudes(place, low, high, radius2), radius2)

    def stack_radius(self, projections: Rows) -> npt.NDArray[np.float32] | jax.Array:
        """:meth:`radius` of each detector row of a ``(views, nv, nu)`` stack, shape ``(nv, 1)``.

        A host stack is read one sharing view at a time, so a file-backed stack
        stays on disk; a device stack reduces its fitted columns in place.
        """
        n, nv, nu = projections.shape
        window = self.window
        views = radius_views(n)
        if hasattr(projections, "devices") and not isinstance(projections, np.ndarray):
            import jax

            with jax.enable_x64(True):
                return _device_radius(
                    self.distance.tobytes(), nu, self.pad, outer_only=self.outer_only
                )(projections)
        low, high = np.zeros((nv, window)), np.zeros((nv, window))
        for view in map(int, views):
            frame = np.asarray(projections[view : view + 1])
            low += np.square(frame[0, :, :window], dtype=np.float64) / len(views)
            high += np.square(frame[0, :, nu - window :], dtype=np.float64) / len(views)
        radius2 = self.mean_square_radius(np.asarray, low[:, None, :], high[:, None, :])
        return np.asarray(radius2[:, 0], np.float32)


@lru_cache(maxsize=16)
def _device_radius(
    distance: bytes, nu: int, pad: int, *, outer_only: bool
) -> Callable[[jax.Array], jax.Array]:
    """Compile the radius reduction on the device that owns the projections."""
    import jax
    import jax.numpy as jnp

    fits = Continuation(np.frombuffer(distance, dtype=np.float64), nu, pad, outer_only=outer_only)

    @jax.jit
    def radius(projections: jax.Array) -> jax.Array:
        views = radius_views(projections.shape[0])
        low = projections[views, :, : fits.window].swapaxes(0, 1).astype(jnp.float64)
        high = projections[views, :, -fits.window :].swapaxes(0, 1).astype(jnp.float64)
        return jnp.asarray(fits.radius(jnp.asarray, low, high)[:, 0], jnp.float32)

    return radius
