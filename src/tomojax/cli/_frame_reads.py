"""Bounded detector reads for the worker's logical (view, row, column) selections."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Protocol, Self, TypeGuard

import h5py
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator


class _FrameWindow(Protocol):
    def __getitem__(self, key: slice, /) -> np.ndarray: ...


class WindowedFrames(Protocol):
    @property
    def dtype(self) -> np.dtype: ...

    def window(self, rows: slice, cols: slice) -> _FrameWindow: ...


class _FrameBlock(WindowedFrames, Protocol):
    def __len__(self) -> int: ...

    def frames_at(self, which: np.ndarray | slice) -> Self: ...


class _Hdf5Frames(WindowedFrames, Protocol):
    """A lazy HDF5 stack: read straight from its dataset at the logical selection."""

    path: str
    dataset: str
    frames: np.ndarray
    first_row: int
    first_col: int


class _BlockFrames(WindowedFrames, Protocol):
    blocks: tuple[_FrameBlock, ...]
    indices: np.ndarray
    frames: np.ndarray
    first_row: int
    first_col: int


# Dispatch by attribute with hasattr, which, unlike a runtime Protocol check, sees
# attributes a proxy stack delegates through __getattr__.
def windowed(data: object) -> TypeGuard[WindowedFrames]:
    """Whether ``data`` reads detector windows lazily rather than by indexing."""
    return hasattr(data, "window")


def _in_blocks(data: WindowedFrames) -> TypeGuard[_BlockFrames]:
    return hasattr(data, "blocks")


def _in_hdf5(data: WindowedFrames) -> TypeGuard[_Hdf5Frames]:
    return hasattr(data, "dataset")


def frame_pixels(
    data: np.ndarray | WindowedFrames, ranges: tuple[range, ...], budget: int
) -> np.ndarray:
    views, rows, cols = ranges
    selected = slice(views.start, views.stop, views.step)
    if isinstance(data, np.ndarray):
        return np.asarray(data[tuple(slice(r.start, r.stop, r.step) for r in ranges)])
    if _in_blocks(data):
        out = np.empty(tuple(map(len, ranges)), data.dtype)
        for mask, block, tile in block_windows(data, ranges):
            out[mask] = frame_pixels(block, tile, budget)
        return out
    if rows.step == cols.step == 1:
        return data.window(slice(rows.start, rows.stop), slice(cols.start, cols.stop))[selected]
    if _in_hdf5(data):
        return hdf5_pixels(data, ranges)

    # Other frame stacks expose contiguous detector windows. Account for all
    # pixels between requested rows and columns when sizing each source read.
    capacity = budget // np.dtype(data.dtype).itemsize
    block = [1, 1, 1]
    for axis, step in ((2, cols.step), (1, rows.step), (0, 1)):
        block[axis] = (
            1 if axis == 1 and rows.step > 1 else min(len(ranges[axis]), 1 + (capacity - 1) // step)
        )
        capacity //= 1 + (block[axis] - 1) * step
    out = np.empty(tuple(map(len, ranges)), data.dtype)
    starts = (range(0, n, b) for n, b in zip(out.shape, block, strict=True))
    for origin in product(*starts):
        target = tuple(slice(i, i + n) for i, n in zip(origin, block, strict=True))
        v, r, c = (axis[key] for axis, key in zip(ranges, target, strict=True))
        detector = data.window(slice(r.start, r[-1] + 1), slice(c.start, c[-1] + 1))
        out[target] = detector[slice(v.start, v.stop, v.step)][:, :: r.step, :: c.step]
    return out


def block_windows(
    stack: _BlockFrames, ranges: tuple[range, ...]
) -> Iterator[tuple[np.ndarray, _FrameBlock, tuple[range, ...]]]:
    """Map a logical window of a multi-file acquisition into its contributing blocks."""
    views, rows, cols = ranges
    indices = stack.indices[stack.frames[views.start : views.stop : views.step]]
    detector = tuple(
        range(r.start + offset, r.stop + offset, r.step)
        for r, offset in ((rows, stack.first_row), (cols, stack.first_col))
    )
    for index in np.unique(indices[:, 0]):
        mask = indices[:, 0] == index
        block = stack.blocks[index].frames_at(indices[mask, 1])
        yield mask, block, (range(len(block)), *detector)


def hdf5_pixels(data: _Hdf5Frames, ranges: tuple[range, ...]) -> np.ndarray:
    views, rows, cols = ranges
    selected = slice(views.start, views.stop, views.step)
    # Map the logical view indices and cropped detector origin back to disk.
    indices = data.frames[selected]
    detector = tuple(
        slice(offset + r.start, offset + r[-1] + 1, r.step)
        for r, offset in ((rows, data.first_row), (cols, data.first_col))
    )
    with h5py.File(data.path, "r", rdcc_nbytes=0) as file:
        dataset = file[data.dataset]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"{data.path}:{data.dataset} must be an HDF5 dataset")
        if dataset.ndim == 2:
            detector = detector[1:]
        shape = tuple(map(len, ranges))
        steps = np.diff(indices)
        if not steps.size or (steps[0] > 0 and np.all(steps == steps[0])):
            frames = slice(indices[0], indices[-1] + 1, steps[0] if steps.size else 1)
            return np.asarray(dataset[(frames, *detector)]).reshape(shape)
        out = np.empty(shape, data.dtype)
        for i, frame in enumerate(indices):
            out[i] = np.asarray(dataset[(frame, *detector)])
        return out
