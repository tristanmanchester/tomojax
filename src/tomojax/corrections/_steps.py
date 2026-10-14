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

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import ClassVar, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from scipy.fft import next_fast_len

from ._records import Correction, Json
from ._view_statistics import view_medians

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
        raise NotImplementedError

    def workspace_pixels(self, rows: int, cols: int) -> int:
        """Device pixels per view this step needs beyond its input, for batch sizing."""
        del rows, cols
        return 0

    def statistic(self, batch: jax.Array) -> jax.Array:
        """A ``(views, ...)`` summary of each view in ``batch``."""
        raise NotImplementedError

    def select(
        self, statistics: np.ndarray, views: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Json]]:
        """Which views to keep (a mask) given every view's statistic, and what was found.

        ``views`` are the views' indices in the scan the steps started from.
        """
        raise NotImplementedError

    def record(self) -> Correction:
        """This step's record: its name and settings."""
        return Correction(
            _snake(type(self).__name__), {k: _json(v) for k, v in asdict(self).items()}
        )


@dataclass(frozen=True)
class BeamHardening(Step):
    """Linearise line integrals with a polynomial: ``p -> c1 p + c2 p^2 + ...``.

    ``coefficients`` are ``(c1, c2, ...)``, with no constant term, as fitted
    from a step wedge or a homogeneous object; ``(1, 0.05)`` adds 5% of ``p^2``.
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
        out = jnp.zeros_like(batch)
        for c in reversed(self.coefficients):  # Horner's rule, from the highest power
            out = (out + c) * batch
        return out


@dataclass(frozen=True)
class Stripes(Step):
    """Remove rings: subtract each detector pixel's constant offset from its line integrals.

    A pixel whose gain or offset the flat field does not describe adds the same
    absorption to every view, a stripe in the sinogram and a ring in the
    volume. Each detector row's values are sorted over views, column by column
    (as in Vo, Atwood and Drakopoulos, 2018), so equal ranks of neighbouring
    columns compare like with like; a pixel's offset is the median over ranks
    of its sorted values minus their median over ``width`` columns. Only that
    constant is subtracted, so data without defects pass unchanged; widen
    ``width`` for wider stripes. An even ``width`` averages the two windows
    centred half a pixel either side, so reversing the detector does not
    change the correction.
    """

    domain: ClassVar[Domain] = "line integrals"
    whole_rows: ClassVar[bool] = True
    width: int = 9

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
        views, rows, cols = batch.shape
        row_bytes = views * cols * batch.dtype.itemsize
        group = max(1, min(rows, 8, (4 << 20) // max(1, row_bytes)))
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
    median (a floor for small, smooth changes).
    Views with a nonfinite median are also rejected and do not participate
    in the neighbours or robust baseline. The scan's geometry keeps the same
    views; the record lists those rejected, including the nonfinite subset.

    This is a local jump heuristic: a majority-bad neighbourhood can hide a
    sustained interruption, and bad neighbours can also cause healthy views
    to be rejected. Inspect the rejected views; use an acquisition mask for
    known invalid views. A wider ``neighbours`` window may expose longer runs
    but also confuse genuine angular changes with faults.
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
        return view_medians(batch)

    def select(
        self, statistics: np.ndarray, views: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Json]]:
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

    def workspace_pixels(self, rows: int, cols: int) -> int:
        # Edge padding and the median network keep whole-image temporaries.
        # Include the larger CPU footprint rather than assuming CUDA fusion
        # removes all scratch: three padded arrays for the small network,
        # five for the bitonic network.
        padded_pixels = (rows + self.size - 1) * (cols + self.size - 1)
        return (3 if self.size == 3 else 5) * padded_pixels

    def apply(self, batch: jax.Array) -> jax.Array:
        half, (_, rows, cols) = self.size // 2, batch.shape
        padded = jnp.pad(batch, ((0, 0), (half, half), (half, half)), mode="edge")
        shifted = [
            padded[:, i : i + rows, j : j + cols]
            for i in range(self.size)
            for j in range(self.size)
        ]
        median = _bitonic_median_of(shifted) if self.size >= 5 else _median_of(shifted)
        return jnp.where(batch - median > self.threshold, median, batch)


@dataclass(frozen=True)
class Paganin(Step):
    """Single-material phase retrieval (Paganin et al. 2002) of propagation-based phase contrast.

    Filters each view's transmission by ``1 / (1 + z lambda (delta/beta) |k|^2 / (4 pi))``
    before the log, so the line integrals are of the material's attenuation
    with the edge fringes folded back in. ``delta_beta`` is the material's
    delta / beta, ``distance`` the effective propagation distance (m: the
    sample-to-detector distance, divided by the magnification in a cone beam),
    ``energy_kev`` the X-ray energy and ``pixel_size`` the detector pixel at
    the sample (m; the pixel divided by the magnification in a cone beam).
    ``pixel_size`` is one pitch for square pixels or ``(u, v)`` pitches for
    rectangular pixels, in the same order as ``Detector.du`` and ``Detector.dv``.
    Edges are repeated over at least sixteen filter decay lengths against
    wrap-around. An additional guard of up to half the image size, capped at
    128 pixels, suppresses tails from the sampled spectrum. FFT extents are
    rounded up to efficient lengths. This also avoids doubling a large detector
    when the filter has only a short range.
    """

    domain: ClassVar[Domain] = "transmission"
    delta_beta: float
    distance: float
    energy_kev: float
    pixel_size: float | tuple[float, float]

    def __post_init__(self) -> None:
        for name in ("delta_beta", "distance", "energy_kev"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not value > 0:
                raise ValueError(f"Paganin {name} must be finite and positive, not {value}")
            object.__setattr__(self, name, value)
        pixels = np.asarray(self.pixel_size, np.float64)
        if pixels.shape not in ((), (2,)) or not np.isfinite(pixels).all() or (pixels <= 0).any():
            raise ValueError(
                f"Paganin pixel_size must be a positive finite scalar or (u, v) pair, "
                f"not {self.pixel_size}"
            )
        normalized = float(pixels) if pixels.ndim == 0 else (float(pixels[0]), float(pixels[1]))
        object.__setattr__(self, "pixel_size", normalized)

    def _pitches(self) -> tuple[float, float]:
        return (
            self.pixel_size
            if isinstance(self.pixel_size, tuple)
            else (self.pixel_size, self.pixel_size)
        )

    def workspace_pixels(self, rows: int, cols: int) -> int:
        # Two padded real arrays and two complex half-spectra: about four
        # padded real arrays per view.
        pv, pu = self._padding(rows, cols)
        return 4 * (rows + sum(pv)) * (cols + sum(pu))

    def _padding(self, rows: int, cols: int) -> tuple[tuple[int, int], tuple[int, int]]:
        wavelength = 12.398419843320026e-10 / self.energy_kev
        scale = self.distance * wavelength * self.delta_beta / (4 * np.pi)
        # The continuous filter's Green function decays over sqrt(scale).
        # Half-image padding can be much too short for a cropped detector;
        # conversely it wastes most of a large image's FFT for a narrow filter.
        pixel_u, pixel_v = self._pitches()

        def axis(n: int, pitch: float) -> tuple[int, int]:
            decay_guard = int(np.ceil(16 * np.sqrt(scale) / pitch))
            # A singleton axis is constant under edge extension, so no FFT
            # samples along it can contribute to the filter.
            guard = max(min(n // 2, 128), decay_guard)
            padded = 1 if n == 1 else cast(int, next_fast_len(n + 2 * guard, real=True))
            before = (padded - n) // 2
            return before, padded - n - before

        return axis(rows, pixel_v), axis(cols, pixel_u)

    def apply(self, batch: jax.Array) -> jax.Array:
        _, rows, cols = batch.shape
        pv, pu = self._padding(rows, cols)
        padded = jnp.pad(batch, ((0, 0), pv, pu), mode="edge")
        wavelength = 12.398419843320026e-10 / self.energy_kev  # metres
        pixel_u, pixel_v = self._pitches()
        ku = 2 * np.pi * np.fft.rfftfreq(padded.shape[2], pixel_u)
        kv = 2 * np.pi * np.fft.fftfreq(padded.shape[1], pixel_v)
        k2 = kv[:, None] ** 2 + ku[None, :] ** 2
        scale = self.distance * wavelength * self.delta_beta / (4 * np.pi)
        response = jnp.asarray(1.0 / (1.0 + scale * k2), jnp.float32)
        # Transmission and the response are real: keep only the Hermitian half.
        # Odd detector extents also give odd padded lengths, so specify the shape
        # rather than letting irfft2 infer an even last axis.
        filtered = jnp.fft.irfft2(jnp.fft.rfft2(padded) * response, s=padded.shape[-2:])
        return filtered[:, pv[0] : pv[0] + rows, pu[0] : pu[0] + cols].astype(batch.dtype)


def _stripe_offsets(row: jax.Array, width: int) -> jax.Array:
    """Each column's offset in one detector row, ``(views, columns)``."""
    ranked = jnp.sort(row, axis=0)
    return jnp.median(ranked - _sliding_median(ranked, width), axis=0)


def _sliding_median(values: jax.Array, width: int) -> jax.Array:
    """The median over ``width`` neighbouring columns, edges repeated.

    An odd-even transposition sort of the shifted copies: elementwise minima
    and maxima, which fuse into one pass (a sort along a window axis is ten
    times slower on a GPU).
    """
    half, columns = width // 2, values.shape[1]
    even = width % 2 == 0
    padded = jnp.pad(values, ((0, 0), (half, half)), mode="edge")
    # An even window is centred between pixels: take one more window and
    # average neighbours onto the pixels, so reversing the detector does not
    # change the correction.
    median = _median_of([padded[:, k : k + columns + even] for k in range(width)])
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
