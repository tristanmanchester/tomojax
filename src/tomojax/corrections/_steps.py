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
from typing import ClassVar, Literal

import jax
import jax.numpy as jnp
import numpy as np

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
        raise NotImplementedError

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
    ``width`` for wider stripes.
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
        # One detector row at a time: the sorted row and its windows stay small.
        offsets = jax.lax.map(
            lambda row: _stripe_offsets(row, self.width), jnp.swapaxes(batch, 0, 1)
        )
        return batch - offsets[None]


@dataclass(frozen=True)
class RejectViews(Step):
    """Drop views whose median line integral is an outlier among the views'.

    A view taken with the beam off, a shutter closed or the sample moved out
    differs from its neighbours as a whole; a view is rejected when its median
    is more than ``z`` robust standard deviations (1.4826 times the median
    absolute deviation) from the median of all views'. The scan's geometry
    keeps the same views; the record lists those rejected.
    """

    domain: ClassVar[Domain] = "line integrals"
    selects_views: ClassVar[bool] = True
    z: float = 6.0

    def __post_init__(self) -> None:
        if not self.z > 0:
            raise ValueError(f"RejectViews z must be positive, not {self.z}")
        object.__setattr__(self, "z", float(self.z))

    def statistic(self, batch: jax.Array) -> jax.Array:
        return jnp.median(batch.reshape(batch.shape[0], -1), axis=1)

    def select(
        self, statistics: np.ndarray, views: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Json]]:
        centre = float(np.median(statistics))
        scale = 1.4826 * float(np.median(np.abs(statistics - centre)))
        if not scale > 0:  # most views alike (a simulation): no scale to judge by
            return np.ones(len(statistics), bool), {"rejected": [], "skipped": "views alike"}
        keep = np.abs(statistics - centre) <= self.z * scale
        found: dict[str, Json] = {
            "rejected": [int(v) for v in views[~keep]],
            "median": centre,
            "robust_scale": scale,
        }
        return keep, found


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
    padded = jnp.pad(values, ((0, 0), (half, width - 1 - half)), mode="edge")
    window = [padded[:, k : k + columns] for k in range(width)]
    for sweep in range(width):
        for i in range(sweep % 2, width - 1, 2):
            low, high = window[i], window[i + 1]
            window[i], window[i + 1] = jnp.minimum(low, high), jnp.maximum(low, high)
    return window[half] if width % 2 else 0.5 * (window[half - 1] + window[half])


def _snake(name: str) -> str:
    return "".join(f"_{c.lower()}" if c.isupper() else c for c in name).lstrip("_")


def _json(value: object) -> Json:
    if isinstance(value, tuple | list):
        return [_json(v) for v in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"step settings are recorded as JSON: {type(value).__name__} is not")
