"""Correction steps: small settings, each acting on one kind of data.

A step acts on ``counts`` (as recorded, before the flat and dark fields),
``transmission`` (after them, before the log) or ``line integrals`` (after it),
which :meth:`tomojax.Frames.corrected` runs in that order. A step that needs
every view of a detector row at once (``whole_rows``) runs on slabs of rows
after the steps before it; the others run on batches of views on the device.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import jax.numpy as jnp

from ._records import Correction, Json

if TYPE_CHECKING:
    import jax

type Domain = Literal["counts", "transmission", "line integrals"]
DOMAINS: tuple[Domain, ...] = ("counts", "transmission", "line integrals")


@dataclass(frozen=True)
class Step:
    """A correction: the data it acts on, and what it does to a batch of it.

    Subclasses are frozen dataclasses of their settings with a ``domain`` and an
    :meth:`apply` taking and returning a ``(views, rows, columns)`` float32 JAX
    array (every view of a slab of rows when ``whole_rows``).
    """

    domain: ClassVar[Domain]
    whole_rows: ClassVar[bool] = False

    def apply(self, batch: jax.Array) -> jax.Array:
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


def _snake(name: str) -> str:
    return "".join(f"_{c.lower()}" if c.isupper() else c for c in name).lstrip("_")


def _json(value: object) -> Json:
    if isinstance(value, tuple | list):
        return [_json(v) for v in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"step settings are recorded as JSON: {type(value).__name__} is not")
