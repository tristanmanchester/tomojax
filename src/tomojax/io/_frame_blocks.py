"""Lazy concatenation of detector-frame blocks in acquisition order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tomojax._data import FrameStack


@dataclass(frozen=True)
class FrameBlocks(FrameStack):
    blocks: tuple[Any, ...]
    indices: np.ndarray

    @classmethod
    def join(cls, blocks: list[Any]) -> FrameBlocks:
        if not blocks or len({b.shape[1:] for b in blocks}) != 1:
            raise ValueError("acquisition blocks must have one detector image size")
        indices = np.concatenate(
            [np.column_stack((np.full(len(b), i), np.arange(len(b)))) for i, b in enumerate(blocks)]
        )
        return cls(
            tuple(blocks),
            indices,
            frames=np.arange(len(indices)),
            rows=blocks[0].shape[1],
            cols=blocks[0].shape[2],
            dtype=np.result_type(*(b.dtype for b in blocks)),
        )

    @property
    def _efficient_row_windows(self) -> bool:
        return all(getattr(b, "_efficient_row_windows", False) for b in self.blocks)

    def _read(self, indices: np.ndarray, rows: slice, cols: slice) -> np.ndarray:
        indices = self.indices[indices]
        result = np.empty((len(indices), *self.shape[1:]), self.dtype)
        for block in np.unique(indices[:, 0]):
            selected = indices[:, 0] == block
            frames, inverse = np.unique(indices[selected, 1], return_inverse=True)
            result[selected] = self.blocks[block].window(rows, cols)[frames][inverse]
        return result
