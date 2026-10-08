"""Solver-independent fixture containers for isolated comparison workers.

Loading a scan must not initialize a competing solver or import JAX/Pallas in
an ASTRA/TIGRE worker. Only NumPy and lightweight physical metadata belong here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from tomojax.core.geometry.base import Detector, Grid


@dataclass(frozen=True)
class Case:
    """One fully specified physical scan and its independent reference volume."""

    name: str
    grid: Grid
    detector: Detector
    poses: np.ndarray
    angles: np.ndarray
    volume: np.ndarray
    analytic: np.ndarray
