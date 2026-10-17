"""Shared lazy detector selection; storage readers own only batch I/O."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Self

import numpy as np


@dataclass(frozen=True, kw_only=True)
class FrameStack(ABC):
    """A logical detector window and frame map, with no open storage resources.

    Selections are metadata-only. A reader implements ``_read`` for source-frame
    indices and physical row/column slices, retaining its batching, ordering and
    storage-layout policy. Reads return owned NumPy storage in the source dtype.
    """

    frames: np.ndarray
    rows: int
    cols: int
    dtype: np.dtype
    first_row: int = 0
    first_col: int = 0

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self.frames), self.rows, self.cols

    @property
    def ndim(self) -> int:
        return 3

    def __len__(self) -> int:
        return len(self.frames)

    def frames_at(self, which: np.ndarray | slice) -> Self:
        """Select frames by indices, mask or slice without reading pixels."""
        return replace(self, frames=np.asarray(self.frames[which], np.int64))

    def window(self, rows: slice, cols: slice) -> Self:
        """Select a contiguous nonempty detector window, relative to this window."""
        r0, r1, rs = rows.indices(self.rows)
        c0, c1, cs = cols.indices(self.cols)
        if rs != 1 or cs != 1 or r1 <= r0 or c1 <= c0:
            raise ValueError("a window of frames is a non-empty block of rows and columns")
        return replace(
            self,
            rows=r1 - r0,
            cols=c1 - c0,
            first_row=self.first_row + r0,
            first_col=self.first_col + c0,
        )

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        if copy is False:
            raise ValueError("lazy frames cannot be converted to an array without copying")
        return np.asarray(self[:], dtype=dtype)

    def __getitem__(self, which: slice | int | np.ndarray) -> np.ndarray:
        if isinstance(which, int | np.integer):
            # Array indexing preserves negative-index and bounds semantics.
            return self[np.asarray([which], np.int64)][0]
        return self._read(
            np.asarray(self.frames[which], np.int64),
            slice(self.first_row, self.first_row + self.rows),
            slice(self.first_col, self.first_col + self.cols),
        )

    @abstractmethod
    def _read(self, indices: np.ndarray, rows: slice, cols: slice) -> np.ndarray:
        """Read one requested batch; all resources are opened and closed here."""
