"""Validation shared by reconstructions that keep inputs and outputs in host memory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ProjectionRows:
    """Zero-extended, optionally interpolated rows, read for requested views only."""

    projections: np.ndarray
    start: int | float
    rows: int
    stride: int | float = 1

    @property
    def shape(self) -> tuple[int, int, int]:
        """The window's projection shape, without reading the input storage."""
        return self.projections.shape[0], self.rows, self.projections.shape[2]

    def __getitem__(self, views: slice) -> np.ndarray:
        positions = self.start + np.arange(self.rows) * self.stride
        indices = np.floor(positions).astype(np.int64)
        weights = (positions - indices).astype(np.float32)
        if weights.any():
            return self._interpolated(views, indices, weights)
        valid = (indices >= 0) & (indices < self.projections.shape[1])
        if valid.all() and self.stride > 0 and float(self.stride).is_integer():
            start, stride = int(self.start), int(self.stride)
            stop = start + self.rows * stride
            return np.ascontiguousarray(
                self.projections[views, start:stop:stride], dtype=np.float32
            )
        count = len(range(*views.indices(self.projections.shape[0])))
        data = np.zeros((count, self.rows, self.projections.shape[2]), dtype=np.float32)
        if valid.any():
            data[:, valid] = self.projections[views, indices[valid], :]
        return data

    def _interpolated(self, views: slice, indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Blend the two supporting source rows before their row-linear u filter."""
        count = len(range(*views.indices(self.projections.shape[0])))
        data = np.zeros((count, self.rows, self.projections.shape[2]), dtype=np.float32)
        for index, weight in ((indices, 1 - weights), (indices + 1, weights)):
            valid = (index >= 0) & (index < self.projections.shape[1])
            if valid.any():
                values = np.asarray(self.projections[views, index[valid], :], dtype=np.float32)
                data[:, valid, :] += values * weight[None, valid, None]
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
