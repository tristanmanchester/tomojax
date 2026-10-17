"""Lazy concatenation of detector-frame blocks in acquisition order."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrameBlocks:
    blocks: tuple[Any, ...]
    indices: np.ndarray

    @classmethod
    def join(cls, blocks: list[Any]) -> FrameBlocks:
        if not blocks or len({b.shape[1:] for b in blocks}) != 1:
            raise ValueError("acquisition blocks must have one detector image size")
        indices = np.concatenate(
            [np.column_stack((np.full(len(b), i), np.arange(len(b)))) for i, b in enumerate(blocks)]
        )
        return cls(tuple(blocks), indices)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.indices), *self.blocks[0].shape[1:])

    @property
    def dtype(self) -> np.dtype:
        return np.result_type(*(b.dtype for b in self.blocks))

    @property
    def ndim(self) -> int:
        return 3

    @property
    def _efficient_row_windows(self) -> bool:
        return all(getattr(b, "_efficient_row_windows", False) for b in self.blocks)

    def __len__(self) -> int:
        return len(self.indices)

    def frames_at(self, which: np.ndarray | slice) -> FrameBlocks:
        return replace(self, indices=self.indices[which])

    def window(self, rows: slice, cols: slice) -> FrameBlocks:
        return replace(self, blocks=tuple(b.window(rows, cols) for b in self.blocks))

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        if copy is False:
            raise ValueError("frame blocks cannot be converted to an array without copying")
        return np.asarray(self[:], dtype=dtype)

    def __getitem__(self, which: slice | int | np.ndarray) -> np.ndarray:
        if isinstance(which, int | np.integer):
            return self[np.asarray([which], np.int64)][0]
        indices = self.indices[which]
        result = np.empty((len(indices), *self.shape[1:]), self.dtype)
        for block in np.unique(indices[:, 0]):
            selected = indices[:, 0] == block
            frames, inverse = np.unique(indices[selected, 1], return_inverse=True)
            result[selected] = self.blocks[block][frames][inverse]
        return result
