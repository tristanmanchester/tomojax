"""Correction steps: small settings, each acting on one kind of data.

A step acts on ``counts`` (as recorded, before the flat and dark fields),
``transmission`` (after them, before the log) or ``line integrals`` (after it),
which :meth:`tomojax.Frames.corrected` runs in that order. A step that needs
every view of a detector row at once (``whole_rows``) runs on slabs of rows
after the steps before it; one that ``selects_views`` keeps some views (the
scan's geometry keeps the same views); the others run on batches of views on
the device.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import ClassVar, Literal, cast

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
import numpy as np
from scipy.fft import next_fast_len

from tomojax.core.robust import MAD_TO_SD

from ._records import Correction, Json

type Domain = Literal["counts", "transmission", "line integrals"]
DOMAINS: tuple[Domain, ...] = ("counts", "transmission", "line integrals")


@dataclass(frozen=True)
class Step:
    """A correction: the data it acts on, and what it does to a batch of it.

    Subclasses are frozen dataclasses of their settings with a ``domain`` and an
    :meth:`apply` taking and returning a ``(views, rows, columns)`` float32 JAX
    array (every view of a slab of rows when ``whole_rows``). A step that
    ``selects_views`` has instead a :meth:`statistic` of each view in a batch
    and a :meth:`select` of the views to keep from every view's statistic.
    """

    domain: ClassVar[Domain]
    whole_rows: ClassVar[bool] = False
    selects_views: ClassVar[bool] = False

    def apply(self, batch: jax.Array) -> jax.Array:
        """Apply this correction to a batch of detector images.

        Parameters
        ----------
        batch : jax.Array
            Float32 data of shape ``(views, rows, columns)`` in this step's domain.
            Whole-row steps require every view of the selected row slab.

        Returns
        -------
        jax.Array
            Corrected data with the same shape and domain.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement a batch correction.
        """
        raise NotImplementedError

    def retained(self, kept: np.ndarray) -> Step:
        """The step for the views ``kept`` after an earlier step selected them.

        Parameters
        ----------
        kept : numpy.ndarray
            Increasing indices, into this step's views, of those that remain.

        Returns
        -------
        Step
            Unchanged; a step planned from the scan's geometry replans itself.
        """
        del kept  # nothing here depends on the views
        return self

    def workspace_pixels(self, rows: int, cols: int) -> int:
        """Estimate device pixels per view needed in addition to the input.

        Parameters
        ----------
        rows : int
            Detector rows in the batch.
        cols : int
            Detector columns in the batch.

        Returns
        -------
        int
            Workspace estimate used to size batches; zero in the base class.
        """
        del rows, cols
        return 0

    def statistic(self, batch: jax.Array) -> jax.Array:
        """Summarize each view for a view-selecting correction.

        Parameters
        ----------
        batch : jax.Array
            Float32 data of shape ``(views, rows, columns)`` in this step's domain.

        Returns
        -------
        jax.Array
            Per-view statistics with shape ``(views, ...)``.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement view statistics.
        """
        raise NotImplementedError

    def select(
        self, statistics: np.ndarray, views: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Json]]:
        """Select views from their statistics and record what was found.

        Parameters
        ----------
        statistics : numpy.ndarray
            Statistics for every input view, with shape ``(views, ...)``.
        views : numpy.ndarray
            View indices in the scan the correction steps started from.

        Returns
        -------
        keep : numpy.ndarray
            Boolean mask of the views to keep.
        found : dict
            JSON-compatible findings to include in the correction record.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement view selection.
        """
        raise NotImplementedError

    def record(self) -> Correction:
        """Build this step's record from its name and settings.

        Returns
        -------
        Correction
            Step name and JSON-compatible settings, with no findings yet.
        """
        return Correction(
            _snake(type(self).__name__), {k: _json(v) for k, v in asdict(self).items()}
        )


@dataclass(frozen=True)
class Stripes(Step):
    """Correct detector stripes that reconstruct as rings.

    A pixel whose gain or offset the flat field does not describe adds the same
    absorption to every view, a stripe in the sinogram and a ring in the
    volume. Each detector row's values are sorted over views, column by column
    (as in Vo, Atwood and Drakopoulos, 2018), so equal ranks of neighbouring
    columns compare like with like; a pixel's offset is the median over ranks
    of its sorted values minus their median over the window. Only that
    constant is subtracted, preserving differences between views. Widen
    ``width`` to about twice the widest stripe. A window of up to nine columns
    is a sliding median over them. A wider one is taken coarse to fine: at
    each scale the sorted values are averaged over blocks of columns and
    compared with the median of nine neighbouring blocks, from a window of
    blocks that spans ``width`` columns down to nine single columns, so a wide
    window costs the time and memory of a narrow one.

    The median assumes the sample's sorted values are level or change
    steadily across the window. A broad stripe on a steep flank of the
    sample's profile, where the profile changes across one block by more than
    the stripe's height, is not seen. A feature symmetric about the
    rotation axis is also stationary in the sinogram, and one narrower than
    ``width`` can be altered: at the peak of a compact sample's profile, on the
    axis, a wide window leaves a faint ring or dot. An even ``width`` of up to
    nine averages the two windows centred half a pixel either side, and wider
    blocks are centred on the detector's middle, so reversing the detector
    does not change the correction.

    ``varying=True`` instead detects pixels with an anomalous view-to-view
    response and fits a gain and offset to their neighbouring sorted values.
    It leaves unselected pixels unchanged, including stationary centred
    features, but does not remove ordinary constant-offset stripes there.
    Detection compares adjacent-view differences with neighbouring columns;
    it needs sufficiently sampled, noisy views, and sharp moving edges can
    cause false detections. Selected unresponsive pixels receive only the
    neighbouring mean: their missing angular information is not recovered.
    It cannot distinguish a real feature from a defect at a selected pixel.

    Parameters
    ----------
    width : int, optional
        Neighbourhood width in detector columns, at least 3. Default is 9.
    varying : bool, optional
        Fit gain and offset at anomalous pixels instead of subtracting constant
        offsets at all pixels. Default is False.

    Raises
    ------
    ValueError
        If width is not a whole number of columns at least 3.

    Examples
    --------
    >>> from tomojax.corrections import Stripes
    >>> correction = Stripes(width=9)
    >>> correction.domain
    'line integrals'
    """

    domain: ClassVar[Domain] = "line integrals"
    whole_rows: ClassVar[bool] = True
    width: int = 9
    varying: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        if int(self.width) != self.width or self.width < 3:
            raise ValueError(
                f"Stripes width must be a whole number of columns >= 3, not {self.width}"
            )
        object.__setattr__(self, "width", int(self.width))

    def apply(self, batch: jax.Array) -> jax.Array:
        # Amortize map launches across small rows, without ranking the whole
        # slab at once. Multiple-row groups hold at most 4 MiB of ranked pixels;
        # a larger single row retains the original serial mapping.
        """Remove stripes from a batch of detector images.

        Parameters
        ----------
        batch : jax.Array
            Float32 line integrals of shape ``(views, rows, columns)``.
            Supply every view of each row for stripe estimation.

        Returns
        -------
        jax.Array
            Corrected line integrals, retaining the input shape.
        """
        views, rows, cols = batch.shape
        row_bytes = views * cols * batch.dtype.itemsize
        group = max(1, min(rows, 8, (4 << 20) // max(1, row_bytes)))
        if self.varying:
            parameters = jax.lax.map(
                lambda row: _stripe_response(row, self.width),
                jnp.swapaxes(batch, 0, 1),
                batch_size=group if group > 1 else None,
            )
            return batch * parameters[None, :, 0] + parameters[None, :, 1]
        offsets = jax.lax.map(
            lambda row: _stripe_offsets(row, self.width),
            jnp.swapaxes(batch, 0, 1),
            batch_size=group if group > 1 else None,
        )
        return batch - offsets[None]


@dataclass(frozen=True)
class RejectViews(Step):
    """Drop views taken during a beam or detector event: a level that starts and stops abruptly.

    A beam loss, a closed shutter or a missed frame changes every pixel of the
    views it touches at once, so the views' mean line integral steps there
    and steps back afterwards, shifting the views between by a constant. The
    object's own change with angle is smooth, however fast: a container
    turning edge-on raises the mean over many views without a step. Each
    view-to-view change of the mean is predicted by the median of its eight
    nearest changes that are not steps, and is a step when it departs from that
    by more than ``z`` robust standard deviations, of those neighbours and of
    the whole scan (each taken from its quietest quarter, so events may be
    most of them), and by more than a tenth of the robust spread of the means
    themselves (for noise-free scans). The views between a step and a step
    back are rejected, however many, when they form a block: the two steps of
    opposite sign within a factor of two, the inside drifting by less than half
    the smaller step from end to end and stepping by less than that between its
    views (a flux tooth inside is not a shifted block), and standing out from
    four views on each side. A first or
    last view that steps away from the rest is rejected too. A step that never
    returns is a level change, and its views are kept. The mean is the total
    attenuation per pixel, so an empty or uniform part of the detector neither
    dominates nor flips it.

    Known behaviour: a flat, real flux change that steps back (a 0.5% step held
    for 50 views, say) is an event, and its views are rejected although an
    alignment's per-view offset would absorb it. A beam loss that starts at the
    second view reads as a failed first view; the first view is rejected and
    the loss kept. A beam that alternates between two levels for a stretch of
    views (a full-flux view every few among dimmed ones) has its minority level
    taken as the event, so the good views are rejected and the dimmed ones kept.

    Views with a nonfinite mean are also rejected and take no part. Rejected
    views are removed from the scan and its geometry; the scan's
    ``view_positions`` keep when each remaining view was taken, so alignment
    charges the gap as the views it hides. The record lists them by their
    index before the step, nonfinite ones included.

    Parameters
    ----------
    z : float, optional
        Positive robust-standard-deviation threshold for a step. Default is 6.

    Raises
    ------
    ValueError
        If the threshold is nonfinite or not positive.
    """

    domain: ClassVar[Domain] = "line integrals"
    selects_views: ClassVar[bool] = True
    z: float = 6.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.z) or not self.z > 0:
            raise ValueError(f"RejectViews z must be finite and positive, not {self.z}")
        object.__setattr__(self, "z", float(self.z))

    def statistic(self, batch: jax.Array) -> jax.Array:
        """Compute the mean line integral of every view.

        Parameters
        ----------
        batch : jax.Array
            Float32 line integrals of shape ``(views, rows, columns)``.

        Returns
        -------
        jax.Array
            Mean of each detector image, with shape ``(views,)``.
        """
        return jnp.mean(batch.reshape(batch.shape[0], -1), axis=1)

    def select(
        self, statistics: np.ndarray, views: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Json]]:
        """Select the views outside every abrupt event in their means.

        Parameters
        ----------
        statistics : numpy.ndarray
            Mean line integral of every input view, with shape ``(views,)``.
        views : numpy.ndarray
            Original scan indices corresponding to these means.

        Returns
        -------
        keep : numpy.ndarray
            Boolean mask of retained views; nonfinite means are rejected.
        found : dict
            Rejected original indices, the step threshold and any nonfinite-view indices.
        """
        means = np.asarray(statistics, np.float64)
        finite = np.isfinite(means)
        keep = np.zeros(len(views), bool)
        limit = 0.0
        if finite.any():
            inside, limit = _events(means[finite], self.z)
            keep[finite] = ~inside
        found: dict[str, Json] = {"rejected": [int(v) for v in views[~keep]], "step_limit": limit}
        if not finite.all():
            found["nonfinite"] = [int(v) for v in views[~finite]]
        return keep, found


def _quiet_quarter_sd(values: np.ndarray) -> float:
    """A standard deviation from the quietest quarter of ``values``.

    A quarter of normal deviations fall below 0.3186 of it, so up to three
    quarters of the values may be events without inflating it.
    """
    return float(np.quantile(np.abs(values - np.median(values)), 0.25)) / 0.3186


# Views either side of a candidate event that it must stand out from.
_EVENT_SIDE = 4
# Changes around a change that predict it, and set its local scale.
_NEAREST_CHANGES = 8


def _steps(means: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Each change's departure from its neighbourhood, which departures are steps, the floor."""
    change = np.diff(means)
    index = np.arange(len(change))
    # float32 statistics resolve about 1e-6 of the largest mean, whatever the spreads.
    floor = max(
        z * _quiet_quarter_sd(change),
        0.1 * _quiet_quarter_sd(means),
        1e-5 * float(np.max(np.abs(means))),
    )
    step = np.zeros(len(change), bool)
    residual, limit = change, np.full(len(change), floor)
    for _ in range(2):  # the second pass predicts from changes that are not steps
        predicted = np.zeros(len(change))
        for i in index:
            others = index[(index != i) & ~step]
            nearest = others[np.argsort(np.abs(others - i), kind="stable")[:_NEAREST_CHANGES]]
            if len(nearest):
                predicted[i] = np.median(change[nearest])
                limit[i] = max(floor, z * _quiet_quarter_sd(change[nearest]))
        residual = change - predicted
        step = np.abs(residual) > limit
    return residual, step, floor


def _events(means: np.ndarray, z: float) -> tuple[np.ndarray, float]:
    """Views inside an abrupt event of ``means``, and the step floor used.

    An event is a block shifted by a constant: a step, a step back of the
    opposite sign within a factor of two, an inside that drifts by less than
    half the smaller step end to end and steps by less than that within, and an
    inside level beyond both sides' by more than
    the step floor. A first or last view that steps away from the rest is one
    too (it has no step back to see). Candidates are taken shortest first, each
    step in one event; a step that never returns is a level change, not an event.
    """
    n = len(means)
    inside = np.zeros(n, bool)
    if n < 3:
        return inside, 0.0
    residual, step, floor = _steps(means, z)
    steps = [(int(b), float(residual[b])) for b in np.flatnonzero(step)]

    def stands_out(first: int, last: int, sign: float) -> bool:
        level = np.median(means[first : last + 1])
        sides = [
            means[max(first - _EVENT_SIDE, 0) : first],
            means[last + 1 : last + 1 + _EVENT_SIDE],
        ]
        sides = [side for side in sides if len(side)]
        return bool(sides) and all(sign * (level - np.median(side)) > floor for side in sides)

    candidates = []
    for a, (b, size) in enumerate(steps):
        if b == 0:
            candidates.append((1, 0, 0, {a}))
        if b == n - 2:
            candidates.append((1, n - 1, n - 1, {a}))
        for c in range(a + 1, len(steps)):
            back, returned = steps[c]
            drift = [means[back] - means[b + 1]] + [inner for _, inner in steps[a + 1 : c]]
            if (
                np.sign(returned) != np.sign(size)
                and 0.5 <= abs(returned / size) <= 2
                and max(np.abs(drift)) < 0.5 * min(abs(size), abs(returned))
                and stands_out(b + 1, back, float(np.sign(size)))
            ):
                candidates.append((back - b, b + 1, back, {a, c}))
    used: set[int] = set()
    for _, first, last, members in sorted(candidates, key=lambda item: item[:3]):
        if used.isdisjoint(members) and not inside[first : last + 1].any():
            inside[first : last + 1] = True
            used |= members
    return inside, floor


@dataclass(frozen=True)
class Zingers(Step):
    """Replace bright specks (zingers) with the median of the pixels around them.

    A pixel is a zinger when its transmission exceeds the median of the
    ``size x size`` pixels around it by more than ``threshold`` (a fraction of
    the open beam). Zingers, from scattered or direct hits on the detector,
    are far brighter than that; a sharp corner of a strongly absorbing object
    can be too, so raise ``threshold`` if one is touched.

    Parameters
    ----------
    threshold : float, optional
        Positive excess over the neighbourhood median, as a fraction of the
        open beam. Default is 0.1.
    size : int, optional
        Odd side length of the square neighbourhood, at least 3 pixels.
        Default is 3.

    Raises
    ------
    ValueError
        If threshold is not finite and positive, or size is not odd and at least 3.
    """

    domain: ClassVar[Domain] = "transmission"
    threshold: float = 0.1
    size: int = 3

    def __post_init__(self) -> None:
        if not np.isfinite(self.threshold) or not self.threshold > 0:
            raise ValueError(f"Zingers threshold must be finite and positive, not {self.threshold}")
        if int(self.size) != self.size or self.size < 3 or self.size % 2 == 0:
            raise ValueError(f"Zingers size must be an odd number of pixels >= 3, not {self.size}")
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "size", int(self.size))

    def apply(self, batch: jax.Array) -> jax.Array:
        """Replace bright specks in a batch of detector images.

        Parameters
        ----------
        batch : jax.Array
            Float32 transmission of shape ``(views, rows, columns)``.
            Rows and columns form each detector image.

        Returns
        -------
        jax.Array
            Corrected transmission, retaining the input shape.
        """
        half, (_, rows, cols) = self.size // 2, batch.shape
        padded = jnp.pad(batch, ((0, 0), (half, half), (half, half)), mode="edge")
        shifted = [
            padded[:, i : i + rows, j : j + cols]
            for i in range(self.size)
            for j in range(self.size)
        ]
        median = bitonic_median(shifted) if self.size >= 5 else _median_of(shifted)
        return jnp.where(batch - median > self.threshold, median, batch)


@dataclass(frozen=True, kw_only=True)
class RadialDistortion(Step):
    """Unwarp transmission with a calibrated backward radial polynomial.

    ``center`` is ``(column, row)`` in zero-based pixel coordinates of the frames
    (not the rotation axis), and ``coefficients`` are ``(c0, c1, ...)``: an
    undistorted pixel at radius ``r`` from the centre samples the measured image
    at radius ``r * (c0 + c1 r + ...)``, the convention of Algotom and Discorpy
    (``(1.0,)`` is the identity). Bilinear interpolation acts on the normalised
    transmission before the log, repeating the nearest edge pixel beyond the
    detector; the shape is kept and no Jacobian factor is applied.

    The calibration must be in the coordinates of the frames corrected: after a
    crop, subtract its first column and row from ``center``; after mean binning
    by ``f``, use ``(center - (f - 1) / 2) / f`` and ``c_k * f**k``. Correct before
    cropping when the map needs pixels outside the crop.

    Parameters
    ----------
    center : tuple of float
        Distortion centre ``(column, row)`` in zero-based frame pixels.
    coefficients : tuple of float
        Backward radial-polynomial coefficients ``(c0, c1, ...)`` in frame-pixel
        coordinates; ``(1.0,)`` is the identity.

    Raises
    ------
    ValueError
        If center is not a finite pair or coefficients are empty or nonfinite.
    """

    domain: ClassVar[Domain] = "transmission"
    center: tuple[float, float]
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        center = tuple(float(c) for c in self.center)
        coefficients = tuple(float(c) for c in self.coefficients)
        if len(center) != 2 or not np.isfinite(center).all():
            raise ValueError("RadialDistortion center must be a finite (column, row) pair")
        if not coefficients or not np.isfinite(coefficients).all():
            raise ValueError("RadialDistortion coefficients must be nonempty and finite")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "coefficients", coefficients)

    def workspace_pixels(self, rows: int, cols: int) -> int:
        """Estimate additional workspace pixels per view.

        Parameters
        ----------
        rows : int
            Detector rows in the batch.
        cols : int
            Detector columns in the batch.

        Returns
        -------
        int
            Two detector-coordinate arrays for resampling.
        """
        return 2 * rows * cols  # the sampling coordinates

    def apply(self, batch: jax.Array) -> jax.Array:
        """Unwarp a batch of detector images.

        Parameters
        ----------
        batch : jax.Array
            Float32 transmission of shape ``(views, rows, columns)``.
            Rows and columns form each detector image.

        Returns
        -------
        jax.Array
            Corrected transmission, retaining the input shape.
        """
        _, rows, cols = batch.shape
        x = jnp.arange(cols, dtype=jnp.float32)[None, :] - self.center[0]
        y = jnp.arange(rows, dtype=jnp.float32)[:, None] - self.center[1]
        radius = jnp.sqrt(x * x + y * y)
        scale = jnp.zeros_like(radius)
        for c in reversed(self.coefficients):  # Horner's rule
            scale = scale * radius + c
        at = (self.center[1] + scale * y, self.center[0] + scale * x)
        return jax.vmap(lambda view: map_coordinates(view, at, order=1, mode="nearest"))(batch)


@dataclass(frozen=True)
class Paganin(Step):
    """Retrieve single-material phase from propagation contrast.

    Filters each view's transmission by ``1 / (1 + z lambda (delta/beta) |k|^2 / (4 pi))``
    before the log, so the line integrals are of the material's attenuation
    with the edge fringes folded back in. ``delta_beta`` is the material's
    delta / beta, ``distance`` the effective propagation distance (m: the
    sample-to-detector distance, divided by the magnification in a cone beam),
    ``energy_kev`` the X-ray energy and ``pixel_size`` the detector pixel at
    the sample (m; the pixel divided by the magnification in a cone beam).
    Views are padded with repeated edges against wrap-around, by at least 16
    decay lengths of the filter's point response (and up to half the view, at
    most 128 pixels), and to an efficient FFT length.

    Parameters
    ----------
    delta_beta : float
        Positive ratio of the material's refractive-index decrement to absorption.
    distance : float
        Positive effective propagation distance at the sample, in metres.
    energy_kev : float
        Positive X-ray energy in keV.
    pixel_size : float
        Positive detector-pixel size at the sample, in metres.

    Raises
    ------
    ValueError
        If any parameter is nonfinite or nonpositive.
    """

    domain: ClassVar[Domain] = "transmission"
    delta_beta: float
    distance: float
    energy_kev: float
    pixel_size: float

    def __post_init__(self) -> None:
        for name in ("delta_beta", "distance", "energy_kev", "pixel_size"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not value > 0:
                raise ValueError(f"Paganin {name} must be finite and positive, not {value}")
            object.__setattr__(self, name, value)

    def workspace_pixels(self, rows: int, cols: int) -> int:
        # Two padded real arrays and two complex half-spectra: about four
        # padded real arrays per view.
        """Estimate additional workspace pixels per view.

        Parameters
        ----------
        rows : int
            Detector rows in the batch.
        cols : int
            Detector columns in the batch.

        Returns
        -------
        int
            Padded real arrays and complex half-spectra for phase filtering.
        """
        (rows_before, rows_after), (cols_before, cols_after) = self._padding(rows, cols)
        return 4 * (rows + rows_before + rows_after) * (cols + cols_before + cols_after)

    def _scale(self) -> float:
        wavelength = 12.398419843320026e-10 / self.energy_kev  # metres
        return self.distance * wavelength * self.delta_beta / (4 * np.pi)

    def _padding(self, rows: int, cols: int) -> tuple[tuple[int, int], tuple[int, int]]:
        # The filter's point response decays over sqrt(scale) metres. Half a view
        # is too little for a narrow crop and wasteful for a large detector.
        decay = 16 * np.sqrt(self._scale()) / self.pixel_size

        def axis(n: int) -> tuple[int, int]:
            if n == 1:  # constant under edge extension
                return 0, 0
            guard = max(min(n // 2, 128), int(np.ceil(decay)))
            extra = cast(int, next_fast_len(n + 2 * guard, real=True)) - n
            return extra // 2, extra - extra // 2

        return axis(rows), axis(cols)

    def apply(self, batch: jax.Array) -> jax.Array:
        """Retrieve phase from a batch of detector images.

        Parameters
        ----------
        batch : jax.Array
            Float32 transmission of shape ``(views, rows, columns)``.
            Rows and columns form each detector image.

        Returns
        -------
        jax.Array
            Corrected transmission, retaining the input shape.
        """
        _, rows, cols = batch.shape
        pv, pu = self._padding(rows, cols)
        padded = jnp.pad(batch, ((0, 0), pv, pu), mode="edge")
        ku = 2 * np.pi * np.fft.rfftfreq(padded.shape[2], self.pixel_size)
        kv = 2 * np.pi * np.fft.fftfreq(padded.shape[1], self.pixel_size)
        k2 = kv[:, None] ** 2 + ku[None, :] ** 2
        response = jnp.asarray(1.0 / (1.0 + self._scale() * k2), jnp.float32)
        # Transmission and the response are real: keep only the Hermitian half.
        # Odd detector extents also give odd padded lengths, so specify the shape
        # rather than letting irfft2 infer an even last axis.
        filtered = jnp.fft.irfft2(jnp.fft.rfft2(padded) * response, s=padded.shape[-2:])
        return filtered[:, pv[0] : pv[0] + rows, pu[0] : pu[0] + cols].astype(batch.dtype)


def _stripe_offsets(row: jax.Array, width: int) -> jax.Array:
    """Each column's offset in one detector row, ``(views, columns)``.

    Coarse to fine: at scale ``f`` the ranked row is averaged over blocks of
    ``f`` columns, the median of :data:`_STRIPE_WINDOW` neighbouring blocks is
    its stripe-free level, and the offsets found are interpolated back to the
    columns and removed before the next finer scale. The coarsest scale spans
    ``width`` columns and the finest is the columns themselves, so stripes of
    every width up to ``width`` are taken at the scale where they are narrow,
    at a cost and memory that do not grow with ``width``.
    """
    ranked = jnp.sort(row, axis=0)
    columns = ranked.shape[1]
    levels = 0
    while _STRIPE_WINDOW << levels < width:
        levels += 1
    offsets = jnp.zeros(columns, ranked.dtype)
    for level in range(levels, -1, -1):
        scale = 1 << level
        window = -(-width // scale) if level == levels else _STRIPE_WINDOW
        rest = ranked - offsets
        if scale == 1:
            offsets = offsets + jnp.median(rest - _sliding_median(rest, window), axis=0)
            continue
        centres, blocks = _block_means(rest, scale)
        found = jnp.median(blocks - _sliding_median(blocks, window), axis=0)
        offsets = offsets + jnp.interp(jnp.arange(columns, dtype=ranked.dtype), centres, found)
    return offsets


# Blocks in each scale's median; also the widest window taken at full resolution.
_STRIPE_WINDOW = 9


def _block_means(values: jax.Array, scale: int) -> tuple[jax.Array, jax.Array]:
    """Means over ``scale``-column blocks centred on the detector middle, edges repeated.

    Block ``j`` covers ``[m + j scale - scale/2, m + j scale + scale/2]`` with
    ``m`` the middle column, so the blocks, and the correction, are symmetric
    under reversal of the detector. A boundary through a column's middle takes
    half of it. Returns the block centres in columns and the means,
    ``(rows, blocks)``.
    """
    columns = values.shape[1]
    middle = (columns - 1) / 2
    reach = -(-int(np.ceil(middle)) // scale)
    centres = middle + scale * np.arange(-reach, reach + 1)
    # Columns are [i - 1/2, i + 1/2]: blocks end on column edges for an even
    # count and on column middles for an odd one.
    halves = columns % 2
    taps = scale + halves
    weights = np.ones(taps)
    if halves:
        weights[[0, -1]] = 0.5
    first = round(centres[0] - scale / 2 + (0.5 if not halves else 0.0))
    pad = max(0, -first, first + (len(centres) - 1) * scale + taps - columns)
    padded = jnp.pad(values, ((0, 0), (pad, pad)), mode="edge")
    start = first + pad
    stop = start + (len(centres) - 1) * scale + 1
    total = weights[0] * padded[:, start:stop:scale]
    for k in range(1, taps):
        total = total + weights[k] * padded[:, start + k : stop + k : scale]
    return jnp.asarray(centres, values.dtype), total / scale


def _stripe_response(row: jax.Array, width: int) -> jax.Array:
    """Gain and offset for pixels with an anomalous angular response.

    Differences, not the mean absorption, select pixels: a stationary
    feature cannot itself trigger detection. The local median removes the
    slowly varying noise envelope; six robust standard deviations select
    exceptional responses. Extend those seeds to include a defect's edges.
    A zero MAD gives no noise-based evidence, so leave that row alone.
    """
    if row.shape[0] < 3:
        return jnp.stack((jnp.ones_like(row[0]), jnp.zeros_like(row[0])))
    noise = jnp.mean(jnp.abs(jnp.diff(row, axis=0)), axis=0)
    local = _sliding_median(noise[None], width)[0]
    relative = noise / jnp.where(local > 0, local, 1)
    ratio = jnp.where(local > 0, jnp.log(jnp.maximum(relative, jnp.finfo(row.dtype).eps)), 0)
    deviation = jnp.abs(ratio - jnp.median(ratio))
    scale = MAD_TO_SD * jnp.median(deviation)
    bad = (deviation > 6 * scale) & (scale > 0)
    radius = max(1, width // 8)
    bad = jax.lax.reduce_window(bad, False, jax.lax.bitwise_or, (2 * radius + 1,), (1,), "SAME")

    return _fit_stripe_response(row, width, noise, bad)


def _response_parameters(
    ranked: jax.Array, target: jax.Array, noise: jax.Array, bad: jax.Array
) -> jax.Array:
    """Fit selected angular distributions, including saturated detector pixels."""
    xmean, ymean = ranked.mean(-2), target.mean(-2)
    dx, dy = ranked - xmean[..., None, :], target - ymean[..., None, :]
    variance = jnp.sum(dx * dx, axis=-2)
    gain = jnp.sum(dx * dy, axis=-2) / jnp.where(variance > 0, variance, 1)
    # A saturated core has no angular information to fit, but still needs
    # its mean offset restored. Leaving it unchanged while correcting its
    # shoulders creates a sharper stripe. Keep unit gain there rather than
    # inventing view dependence; noise also detects nonzero constant columns
    # whose floating-point mean can leave a spurious nonzero variance.
    responsive = (noise > 0) & (variance > 0)
    gain = jnp.where(bad & responsive, gain, 1)
    bias = jnp.where(bad, ymean - gain * xmean, 0)
    return jnp.stack((gain, bias))


def _fit_stripe_response(row: jax.Array, width: int, noise: jax.Array, bad: jax.Array) -> jax.Array:
    """Rank only selected column tiles, with full median halos and a dense fallback."""
    columns, tile, half = row.shape[1], 256, width // 2

    def dense():
        # Matching distributions retains angular features; do not interpolate
        # measured columns. This is also the reference for densely bad rows.
        ranked = jnp.sort(row, axis=0)
        return _response_parameters(ranked, _sliding_median(ranked, width), noise, bad)

    # Keep packed sorting storage below half a dense row, including its halos.
    slots = columns // (2 * (tile + 2 * half))
    if slots < 1:
        return dense()
    tiles = (columns + tile - 1) // tile
    extra = tiles * tile - columns
    flags = jnp.pad(bad, (0, extra))
    active = flags.reshape(tiles, tile).any(1)
    count = active.sum()

    def sparse():
        starts = jnp.nonzero(active, size=slots, fill_value=0)[0] * tile
        indices = starts[:, None] + jnp.arange(tile + 2 * half)[None]
        padded = jnp.pad(row, ((0, 0), (half, half + extra)), mode="edge")
        values = jnp.take(padded, indices, axis=1).transpose(1, 0, 2)
        ranked = jnp.sort(values, axis=1)
        target = _sliding_median(ranked.reshape(-1, tile + 2 * half), width)
        target = target.reshape(slots, row.shape[0], tile + 2 * half)[:, :, half : half + tile]
        destinations = starts[:, None] + jnp.arange(tile)[None]
        signal = jnp.pad(noise, (0, extra))[destinations]
        params = _response_parameters(
            ranked[:, :, half : half + tile], target, signal, flags[destinations]
        )
        # Unused packed slots repeat the first tile only for static shape;
        # discard their writes rather than overwrite any real response.
        destinations = jnp.where(jnp.arange(slots)[:, None] < count, destinations, tiles * tile)
        out = jnp.stack((jnp.ones(tiles * tile, row.dtype), jnp.zeros(tiles * tile, row.dtype)))
        return out.at[:, destinations.reshape(-1)].set(params.reshape(2, -1), mode="drop")[
            :, :columns
        ]

    def identity():
        return jnp.stack((jnp.ones(columns, row.dtype), jnp.zeros(columns, row.dtype)))

    return jax.lax.cond(count == 0, identity, lambda: jax.lax.cond(count <= slots, sparse, dense))


def _sliding_median(values: jax.Array, width: int) -> jax.Array:
    """The median over ``width`` neighbouring columns, edges repeated.

    Sorting networks of shifted copies use elementwise minima and maxima,
    which fuse into one pass. Wide odd windows retain only comparisons that
    reach the median; even windows also need its neighbouring rank. Each row
    is independent, so rows are taken in blocks small enough that the window's
    copies of a block stay within a fixed working size, whatever the width.
    """
    rows, columns = values.shape
    block = max(1, min(rows, _MEDIAN_BLOCK_BYTES // max(1, 4 * width * (columns + width))))
    if block >= rows:
        return _window_median(values, width)
    blocks = -(-rows // block)
    padded = jnp.pad(values, ((0, blocks * block - rows), (0, 0)))
    medians = jax.lax.map(
        lambda part: _window_median(part, width), padded.reshape(blocks, block, columns)
    )
    return medians.reshape(blocks * block, columns)[:rows]


# The window's copies of one block of rows. Narrow windows fuse into one pass and
# never hold them, so a nine-column window over a few thousand views stays whole.
_MEDIAN_BLOCK_BYTES = 512 << 20


def _window_median(values: jax.Array, width: int) -> jax.Array:
    """``_sliding_median`` of one block of rows."""
    half, columns = width // 2, values.shape[1]
    even = width % 2 == 0
    padded = jnp.pad(values, ((0, 0), (half, half)), mode="edge")
    # An even window is centred between pixels: take one more window and
    # average neighbours onto the pixels, so reversing the detector does not
    # change the correction.
    window = [padded[:, k : k + columns + even] for k in range(width)]
    # Wide stripe windows otherwise trace a quadratic sorting network. Reuse
    # the median-only network, keeping the even-window midpoint convention.
    median = bitonic_median(window) if width >= 15 and not even else _median_of(window)
    return 0.5 * (median[:, :-1] + median[:, 1:]) if even else median


def _median_of(window: list[jax.Array]) -> jax.Array:
    """The elementwise median of equally shaped arrays, by odd-even transposition sort."""
    window, n = list(window), len(window)
    for sweep in range(n):
        for i in range(sweep % 2, n - 1, 2):
            low, high = window[i], window[i + 1]
            window[i], window[i + 1] = jnp.minimum(low, high), jnp.maximum(low, high)
    return window[n // 2] if n % 2 else 0.5 * (window[n // 2 - 1] + window[n // 2])


def bitonic_median(window: list[jax.Array]) -> jax.Array:
    """An odd window's median, without tracing comparisons unrelated to it.

    Larger spatial windows benefit from a shallower bitonic network
    instead of quadratic odd-even sorting. Padded positive-infinity wires are
    symbolic, not image arrays. Every real input reaches the median expression,
    so NaNs propagate even after redundant infinity comparisons are removed.
    """
    operations, output = median_network(len(window))
    values = dict(enumerate(window))
    for dest, minimum, a, b in operations:
        compare = jnp.minimum if minimum else jnp.maximum
        values[dest] = compare(values[a], values[b])
    return values[output]


@lru_cache(maxsize=64)
def median_network(n: int) -> tuple[tuple[tuple[int, bool, int, int], ...], int]:
    """Static median-only instructions; wire -1 represents positive infinity."""
    width = 1 << (n - 1).bit_length()
    wires = list(range(n)) + [-1] * (width - n)
    operations: list[tuple[bool, int, int]] = []
    stage = 2
    while stage <= width:
        distance = stage // 2
        while distance:
            for i in range(width):
                j = i ^ distance
                if j <= i:
                    continue
                a, b = wires[i], wires[j]
                if a == -1 or b == -1:
                    low, high = (b if a == -1 else a), -1
                else:
                    low, high = n + len(operations), n + len(operations) + 1
                    operations.extend(((True, a, b), (False, a, b)))
                wires[i], wires[j] = (high, low) if i & stage else (low, high)
            distance //= 2
        stage *= 2
    output = wires[n // 2]
    live = {output}
    needed: list[tuple[int, bool, int, int]] = []
    for k in reversed(range(len(operations))):
        if n + k in live:
            minimum, a, b = operations[k]
            needed.append((n + k, minimum, a, b))
            live.update((a, b))
    return tuple(reversed(needed)), output


def _snake(name: str) -> str:
    return "".join(f"_{c.lower()}" if c.isupper() else c for c in name).lstrip("_")


def _json(value: object) -> Json:
    if isinstance(value, tuple | list):
        return [_json(v) for v in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"step settings are recorded as JSON: {type(value).__name__} is not")
