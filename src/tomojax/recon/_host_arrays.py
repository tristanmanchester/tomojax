"""Validation shared by reconstructions that keep inputs and outputs in host memory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ProjectionRows:
    """A zero-extended detector-row window, read only for requested view batches."""

    projections: np.ndarray
    start: int
    rows: int

    @property
    def shape(self) -> tuple[int, int, int]:
        """The window's projection shape, without reading the input storage."""
        return self.projections.shape[0], self.rows, self.projections.shape[2]

    def __getitem__(self, views: slice) -> np.ndarray:
        batch = self.projections[views]
        lo, hi = max(0, self.start), min(batch.shape[1], self.start + self.rows)
        if lo == self.start and hi == self.start + self.rows:
            return np.ascontiguousarray(batch[:, lo:hi], dtype=np.float32)
        data = np.zeros((batch.shape[0], self.rows, batch.shape[2]), dtype=np.float32)
        if hi > lo:
            data[:, lo - self.start : hi - self.start] = batch[:, lo:hi]
        return data


def _mapped_storage(array: np.ndarray) -> np.memmap | None:
    current: object = array
    while isinstance(current, np.ndarray):
        if isinstance(current, np.memmap):
            return current
        current = current.base
    return None


def validate_host_arrays(
    projections: np.ndarray,
    out: np.ndarray | None,
    shape: tuple[int, int, int],
    context: str = "fbp_host",
) -> np.ndarray:
    """Check host input/output arrays and allocate the output volume if needed."""
    if not isinstance(projections, np.ndarray) or projections.dtype.kind not in "buif":
        raise TypeError(f"{context}: projections must be a real NumPy array or memmap")
    if out is None:
        return np.empty(shape, dtype=np.float32)
    if not isinstance(out, np.ndarray) or out.shape != shape or out.dtype != np.float32:
        raise ValueError(f"{context}: out must be a float32 NumPy array with the volume shape")
    if not out.flags.writeable:
        raise ValueError(f"{context}: out must be writable")
    if np.may_share_memory(projections, out):
        raise ValueError(f"{context}: input and output storage must not overlap")
    input_map, output_map = _mapped_storage(projections), _mapped_storage(out)
    if input_map is not None and output_map is not None:
        source, target = input_map.filename, output_map.filename
        if source is None or target is None or Path(source).samefile(target):
            raise ValueError(f"{context}: memory-mapped input and output require separate files")
    return out
