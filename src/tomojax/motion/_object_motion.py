"""Object-frame motion trace containers and CSV IO."""
# pyright: reportAny=false

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

OBJECT_MOTION_FIELDS = ("view", "tx_obj_px", "ty_obj_px", "tz_obj_px", "rot_obj_z_deg")


@dataclass(frozen=True)
class ObjectMotionTrace:
    """Per-view object-frame rigid motion truth or estimates."""

    tx_obj_px: NDArray[np.float64]
    ty_obj_px: NDArray[np.float64]
    tz_obj_px: NDArray[np.float64]
    rot_obj_z_deg: NDArray[np.float64]

    @classmethod
    def zeros(cls, n_views: int) -> ObjectMotionTrace:
        values = np.zeros(int(n_views), dtype=np.float64)
        return cls(
            tx_obj_px=values.copy(),
            ty_obj_px=values.copy(),
            tz_obj_px=values.copy(),
            rot_obj_z_deg=values.copy(),
        )

    @property
    def n_views(self) -> int:
        return int(self.tx_obj_px.shape[0])

    def __post_init__(self) -> None:
        shapes = {
            self.tx_obj_px.shape,
            self.ty_obj_px.shape,
            self.tz_obj_px.shape,
            self.rot_obj_z_deg.shape,
        }
        if len(shapes) != 1:
            raise ValueError("all object-motion arrays must have the same shape")
        if len(next(iter(shapes))) != 1:
            raise ValueError("object-motion arrays must be one-dimensional")

    def tx_rmse_px(self, reference: ObjectMotionTrace) -> float:
        """Return object-frame tx RMSE against a reference trace."""
        if self.n_views != reference.n_views:
            raise ValueError("object-motion traces must have matching view counts")
        error = self.tx_obj_px - reference.tx_obj_px
        return float(np.sqrt(np.mean(error**2)))


def read_object_motion_csv(path: Path) -> ObjectMotionTrace:
    """Read a `true_motion.csv` object-frame motion sidecar."""
    rows: list[dict[str, str]] = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = set(OBJECT_MOTION_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"object-motion CSV missing columns: {sorted(missing)}")
        rows = list(reader)
    return ObjectMotionTrace(
        tx_obj_px=_column(rows, "tx_obj_px"),
        ty_obj_px=_column(rows, "ty_obj_px"),
        tz_obj_px=_column(rows, "tz_obj_px"),
        rot_obj_z_deg=_column(rows, "rot_obj_z_deg"),
    )


def write_object_motion_csv(path: Path, trace: ObjectMotionTrace) -> None:
    """Write an object-frame motion sidecar."""
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OBJECT_MOTION_FIELDS)
        writer.writeheader()
        for idx in range(trace.n_views):
            writer.writerow(
                {
                    "view": idx,
                    "tx_obj_px": float(trace.tx_obj_px[idx]),
                    "ty_obj_px": float(trace.ty_obj_px[idx]),
                    "tz_obj_px": float(trace.tz_obj_px[idx]),
                    "rot_obj_z_deg": float(trace.rot_obj_z_deg[idx]),
                }
            )


def _column(rows: list[dict[str, str]], name: str) -> NDArray[np.float64]:
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)
