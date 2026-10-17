"""Shared reconstruction types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import jax
    import numpy as np

Regulariser = Literal["tv", "huber_tv"]


@dataclass
class ReconstructionSlab[T]:
    """Owned host output, measured source interval and retained producer context."""

    start: int
    stop: int
    rows: slice
    projections: np.ndarray | jax.Array
    volume: np.ndarray
    context: T
    info: dict[str, object] = field(default_factory=dict)


__all__ = ["ReconstructionSlab", "Regulariser"]
