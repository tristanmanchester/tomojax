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
class BeamHardening(Step):
    """Linearise line integrals with a polynomial: ``p -> c1 p + c2 p^2 + ...``.

    ``coefficients`` are ``(c1, c2, ...)``, with no constant term, as fitted
    from a step wedge or a homogeneous object; ``(1, 0.05)`` adds 5% of ``p^2``.

    Parameters
    ----------
    coefficients : tuple of float, optional
        Polynomial coefficients ``(c1, c2, ...)`` without a constant term.
        The default ``(1.0,)`` leaves the line integrals unchanged.

    Raises
    ------
    ValueError
        If coefficients are empty or nonfinite.
    """

    domain: ClassVar[Domain] = "line integrals"
    coefficients: tuple[float, ...] = (1.0,)

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficients", tuple(float(c) for c in self.coefficients))
        if not self.coefficients:
            raise ValueError("BeamHardening needs at least one coefficient")
        if not np.isfinite(self.coefficients).all():
            raise ValueError("BeamHardening coefficients must be finite")

    def apply(self, batch: jax.Array) -> jax.Array:
        """Linearize a batch of detector images.

        Parameters
        ----------
        batch : jax.Array
            Float32 line integrals of shape ``(views, rows, columns)``.
            Rows and columns form each detector image.

        Returns
        -------
        jax.Array
            Corrected line integrals, retaining the input shape.
        """
        out = jnp.zeros_like(batch)
        for c in reversed(self.coefficients):  # Horner's rule, from the highest power
            out = (out + c) * batch
        return out


@dataclass(frozen=True)
class Stripes(Step):
    """Correct detector stripes that reconstruct as rings.

    A pixel whose gain or offset the flat field does not describe adds the same
    absorption to every view, a stripe in the sinogram and a ring in the
    volume. Each detector row's values are sorted over views, column by column
    (as in Vo, Atwood and Drakopoulos, 2018), so equal ranks of neighbouring
    columns compare like with like; a pixel's offset is the median over ranks
    of its sorted values minus their median over ``width`` columns. Only that
    constant is subtracted, preserving differences between views; widen
    ``width`` for wider stripes. A feature symmetric about the rotation axis is
    also stationary in the sinogram, and one narrower than ``width`` can be
    altered. An even ``width`` averages the two windows centred half a pixel
    either side, so reversing the detector does not change the correction.

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
    """Drop views whose median line integral jumps from their neighbours'.

    A view taken with the beam off, a shutter closed or the sample out of the
    beam differs as a whole from the views either side of it, while the
    object's own change with angle is smooth. Each view's median is compared
    with the median of the ``neighbours`` views around it (in the order
    recorded); a view is rejected when that difference is more than ``z``
    robust standard deviations (1.4826 times the median absolute deviation)
    of all views' differences and more than ``min_jump`` of the views' typical
    median (so the small, smooth changes of a noise-free scan never count).
    Views with a nonfinite median are also rejected and do not participate
    in the neighbours or robust baseline. The scan's geometry keeps the same
    views; the record lists those rejected, including the nonfinite subset.

    Parameters
    ----------
    z : float, optional
        Positive robust-standard-deviation threshold. Default is 6.
    neighbours : int, optional
        Odd neighbourhood length of at least 3 views. Default is 5.
    min_jump : float, optional
        Nonnegative minimum jump as a fraction of the typical view median.
        Default is 0.05.

    Raises
    ------
    ValueError
        If thresholds are nonfinite or outside their ranges, or the
        neighbourhood length is not odd and at least 3.
    """

    domain: ClassVar[Domain] = "line integrals"
    selects_views: ClassVar[bool] = True
    z: float = 6.0
    neighbours: int = 5
    min_jump: float = 0.05

    def __post_init__(self) -> None:
        if not np.isfinite(self.z) or not self.z > 0:
            raise ValueError(f"RejectViews z must be finite and positive, not {self.z}")
        if not np.isfinite(self.min_jump) or not self.min_jump >= 0:
            raise ValueError(f"RejectViews min_jump must be finite and >= 0, not {self.min_jump}")
        object.__setattr__(self, "min_jump", float(self.min_jump))
        if (
            int(self.neighbours) != self.neighbours
            or self.neighbours < 3
            or self.neighbours % 2 == 0
        ):
            raise ValueError(f"RejectViews neighbours must be odd and >= 3, not {self.neighbours}")
        object.__setattr__(self, "z", float(self.z))
        object.__setattr__(self, "neighbours", int(self.neighbours))

    def statistic(self, batch: jax.Array) -> jax.Array:
        """Compute the median line integral of every view.

        Parameters
        ----------
        batch : jax.Array
            Float32 line integrals of shape ``(views, rows, columns)``.

        Returns
        -------
        jax.Array
            Median of each detector image, with shape ``(views,)``.
        """
        return jnp.median(batch.reshape(batch.shape[0], -1), axis=1)

    def select(
        self, statistics: np.ndarray, views: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Json]]:
        """Select views with medians consistent with their neighbours.

        Parameters
        ----------
        statistics : numpy.ndarray
            Median line integral of every input view, with shape ``(views,)``.
        views : numpy.ndarray
            Original scan indices corresponding to these medians.

        Returns
        -------
        keep : numpy.ndarray
            Boolean mask of retained views; nonfinite medians are rejected.
        found : dict
            Rejected original indices, robust scale and any nonfinite-view indices.
        """
        medians = np.asarray(statistics, np.float64)
        finite = np.isfinite(medians)
        keep = np.zeros(len(views), bool)
        scale = 0.0
        if finite.any():
            medians = medians[finite]
            half = self.neighbours // 2
            # Reflect, so an end view is compared with its neighbours, not itself.
            padded = np.pad(medians, half, mode="reflect")
            around = np.median(
                np.lib.stride_tricks.sliding_window_view(padded, self.neighbours), axis=1
            )
            jumps = medians - around
            scale = 1.4826 * float(np.median(np.abs(jumps - np.median(jumps))))
            floor = self.min_jump * float(np.median(np.abs(medians)))
            keep[finite] = (np.abs(jumps) <= self.z * scale) | (np.abs(jumps) <= floor)
        found: dict[str, Json] = {"rejected": [int(v) for v in views[~keep]], "robust_scale": scale}
        if not finite.all():
            found["nonfinite"] = [int(v) for v in views[~finite]]
        return keep, found


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
        median = _bitonic_median_of(shifted) if self.size >= 5 else _median_of(shifted)
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
    """Each column's offset in one detector row, ``(views, columns)``."""
    ranked = jnp.sort(row, axis=0)
    return jnp.median(ranked - _sliding_median(ranked, width), axis=0)


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
    scale = 1.4826 * jnp.median(deviation)
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
    reach the median; even windows also need its neighbouring rank.
    """
    half, columns = width // 2, values.shape[1]
    even = width % 2 == 0
    padded = jnp.pad(values, ((0, 0), (half, half)), mode="edge")
    # An even window is centred between pixels: take one more window and
    # average neighbours onto the pixels, so reversing the detector does not
    # change the correction.
    window = [padded[:, k : k + columns + even] for k in range(width)]
    # Wide stripe windows otherwise trace a quadratic sorting network. Reuse
    # the median-only network, keeping the even-window midpoint convention.
    median = _bitonic_median_of(window) if width >= 15 and not even else _median_of(window)
    return 0.5 * (median[:, :-1] + median[:, 1:]) if even else median


def _median_of(window: list[jax.Array]) -> jax.Array:
    """The elementwise median of equally shaped arrays, by odd-even transposition sort."""
    window, n = list(window), len(window)
    for sweep in range(n):
        for i in range(sweep % 2, n - 1, 2):
            low, high = window[i], window[i + 1]
            window[i], window[i + 1] = jnp.minimum(low, high), jnp.maximum(low, high)
    return window[n // 2] if n % 2 else 0.5 * (window[n // 2 - 1] + window[n // 2])


def _bitonic_median_of(window: list[jax.Array]) -> jax.Array:
    """An odd window's median, without tracing comparisons unrelated to it.

    Larger spatial windows benefit from a shallower bitonic network
    instead of quadratic odd-even sorting. Padded positive-infinity wires are
    symbolic, not image arrays. Every real input reaches the median expression,
    so NaNs propagate even after redundant infinity comparisons are removed.
    """
    operations, output = _median_network(len(window))
    values = dict(enumerate(window))
    for dest, minimum, a, b in operations:
        compare = jnp.minimum if minimum else jnp.maximum
        values[dest] = compare(values[a], values[b])
    return values[output]


@lru_cache(maxsize=64)
def _median_network(n: int) -> tuple[tuple[tuple[int, bool, int, int], ...], int]:
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
