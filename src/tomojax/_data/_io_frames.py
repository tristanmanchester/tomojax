"""A stack of frames in an HDF5 file, read only when, and as far as, it is indexed."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True)
class Hdf5Frames:
    """Frames ``frames`` of the 3-D dataset ``dataset`` in ``path``, read on indexing.

    It holds no open file: each read opens the file, reads the frames asked for
    and closes it, so it outlives the call that made it and is safe to read
    from several threads. Indexing by a slice or index array selects frames
    (``stack[a:b]`` reads frames a to b of this stack) and returns NumPy arrays
    in the file's dtype; :meth:`frames_at` narrows the stack without reading.
    """

    path: str
    dataset: str
    frames: np.ndarray  # indices into the dataset's first axis, increasing
    rows: int
    cols: int
    dtype: np.dtype

    @classmethod
    def open(cls, path: str, dataset: str) -> Hdf5Frames:
        """Every frame of ``dataset`` in ``path``."""
        with h5py.File(path, "r") as file:
            data = file[dataset]
            if not isinstance(data, h5py.Dataset) or data.ndim != 3:
                raise ValueError(f"{path}:{dataset} is not a 3-D stack of frames")
            n, rows, cols = (int(s) for s in data.shape)
            return cls(path, dataset, np.arange(n), rows, cols, np.dtype(data.dtype))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.frames), self.rows, self.cols)

    @property
    def ndim(self) -> int:
        return 3

    def __len__(self) -> int:
        return len(self.frames)

    def frames_at(self, which: np.ndarray | slice) -> Hdf5Frames:
        """The stack of these frames only (a boolean mask, indices or slice), unread."""
        return replace(self, frames=np.asarray(self.frames[which], np.int64))

    def __getitem__(self, which: slice | int | np.ndarray) -> np.ndarray:
        if isinstance(which, int | np.integer):
            return self[int(which) : int(which) + 1][0]
        indices = np.asarray(self.frames[which], np.int64)
        if indices.size and np.any(np.diff(indices) <= 0):
            raise ValueError("Hdf5Frames reads frames in increasing order only")
        with h5py.File(self.path, "r") as file:
            data: Any = file[self.dataset]
            if indices.size == 0:
                return np.empty((0, self.rows, self.cols), self.dtype)
            first, last = int(indices[0]), int(indices[-1]) + 1
            if last - first == indices.size:  # contiguous: one hyperslab
                return np.asarray(data[first:last])
            return np.asarray(data[indices])
