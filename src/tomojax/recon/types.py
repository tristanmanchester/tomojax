"""Shared reconstruction types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import jax
    import numpy as np

Regulariser = Literal["tv", "huber_tv"]


@dataclass
class ReconstructionSlab[T]:
    """Owned host output, measured source interval and retained producer context.

    Parameters
    ----------
    start : int
        First output z index, inclusive.
    stop : int
        Final output z index, exclusive.
    rows : slice
        Measured detector-row interval used by this slab.
    projections : numpy.ndarray or jax.Array
        Retained source projections for the measured row interval.
    volume : numpy.ndarray
        Owned float32 reconstruction with shape ``(nx, ny, stop - start)``.
    context : object
        Opaque producer context retained until output completes.
    """

    start: int
    stop: int
    rows: slice
    projections: np.ndarray | jax.Array
    volume: np.ndarray
    context: T


__all__ = ["ReconstructionSlab", "Regulariser"]
