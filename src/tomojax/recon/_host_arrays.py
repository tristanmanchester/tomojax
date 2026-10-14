"""Validation shared by reconstructions that keep inputs and outputs in host memory."""

from __future__ import annotations

from dataclasses import dataclass
import math
import mmap
import os
from pathlib import Path
import sys
from typing import Literal

import numpy as np

from tomojax._typed_arrays import mapped_storage


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


def _hugepage_capacity() -> int | None:
    """Linux buddy blocks large enough for THP without synchronous compaction."""
    if sys.platform != "linux":
        return None
    try:
        root = Path("/sys/kernel/mm/transparent_hugepage")
        if "[never]" in (root / "enabled").read_text():
            return None
        defrag = (root / "defrag").read_text().split()
        if not any(mode in defrag for mode in ("[always]", "[madvise]", "[defer+madvise]")):
            return None
        page = os.sysconf("SC_PAGE_SIZE")
        huge = int((root / "hpage_pmd_size").read_text())
        minimum = (huge // page).bit_length() - 1
        return sum(
            int(count) * (1 << order) * page
            for line in Path("/proc/buddyinfo").read_text().splitlines()
            for order, count in enumerate(line.split()[4:])
            if order >= minimum
        )
    except (OSError, ValueError):
        return None


def _empty_host_output(
    shape: tuple[int, int, int], *, order: Literal["C", "F"] = "C"
) -> np.ndarray:
    """Avoid direct THP defragmentation for outputs exceeding contiguous RAM.

    Advice applies only to this owned mapping, never caller storage or NumPy's
    process-wide allocation policy. Small outputs retain ordinary NumPy pages.
    """
    size = math.prod(shape) * np.dtype(np.float32).itemsize
    capacity = _hugepage_capacity()
    if capacity is not None and size > capacity:
        storage = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        storage.madvise(mmap.MADV_NOHUGEPAGE)
        return np.ndarray(shape, dtype=np.float32, buffer=storage, order=order)
    return np.empty(shape, dtype=np.float32, order=order)


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
        return _empty_host_output(shape)
    if not isinstance(out, np.ndarray) or out.shape != shape or out.dtype != np.float32:
        raise ValueError(f"{context}: out must be a float32 NumPy array with the volume shape")
    if not out.flags.writeable:
        raise ValueError(f"{context}: out must be writable")
    if np.may_share_memory(projections, out):
        raise ValueError(f"{context}: input and output storage must not overlap")
    input_map, output_map = mapped_storage(projections), mapped_storage(out)
    if input_map is not None and output_map is not None:
        source, target = input_map.filename, output_map.filename
        if source is None or target is None or Path(source).samefile(target):
            raise ValueError(f"{context}: memory-mapped input and output require separate files")
    return out
