"""Bounded detector reads for the worker's logical (view, row, column) selections."""

from itertools import product
from math import prod
from typing import cast

import h5py
import numpy as np

from tomojax.core.files import read_hdf5
from tomojax.corrections.tiles import corrected_tile
from tomojax.errors import InvalidParameter


def frame_pixels(data, ranges, budget):
    views, rows, cols = ranges
    selected = slice(views.start, views.stop, views.step)
    if hasattr(data, "sampled"):
        return data.sampled(ranges)
    if not hasattr(data, "window"):
        return np.asarray(data[tuple(slice(r.start, r.stop, r.step) for r in ranges)])
    if hasattr(data, "blocks"):
        out = np.empty(tuple(map(len, ranges)), data.dtype)
        for mask, block, tile in block_windows(data, ranges):
            out[mask] = frame_pixels(block, tile, budget)
        return out
    if rows.step == cols.step == 1:
        return data.window(slice(rows.start, rows.stop), slice(cols.start, cols.stop))[selected]
    if hasattr(data, "dataset"):
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


def block_windows(stack, ranges):
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


def hdf5_pixels(data, ranges):
    views, rows, cols = ranges
    selected = slice(views.start, views.stop, views.step)
    # Map the logical view indices and cropped detector origin back to disk.
    indices = data.frames[selected]
    detector = tuple(
        slice(offset + r.start, offset + r[-1] + 1, r.step)
        for r, offset in ((rows, data.first_row), (cols, data.first_col))
    )
    with read_hdf5(data.path, rdcc_nbytes=0) as file:
        dataset = cast("h5py.Dataset", file[data.dataset])
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


def source_bytes(stack, ranges):
    views, rows, cols = ranges
    if hasattr(stack, "read_bytes"):
        return stack.read_bytes(ranges)
    if hasattr(stack, "blocks"):
        return sum(source_bytes(block, tile) for _, block, tile in block_windows(stack, ranges))
    if hasattr(stack, "window") and not hasattr(stack, "dataset"):
        # Some detectors expose complete rows even for a narrow column window.
        width = getattr(stack, "read_width", cols[-1] - cols[0] + 1)
        return len(views) * len(rows) * width * stack.dtype.itemsize
    return prod(map(len, ranges)) * getattr(stack, "storage_dtype", stack.dtype).itemsize


class ReadLimit(InvalidParameter):
    """The requested slice needs more detector input than a single read permits."""

    def __init__(self) -> None:
        super().__init__("roi", "read exceeds 16 MiB; use a stride or a smaller ROI")


class SampleTiles:
    def __init__(self, samples, calibration=None) -> None:
        self.samples = samples
        self.calibration = calibration
        self.dtype = np.dtype(np.float32) if calibration is not None else samples.dtype
        self.shape = samples.shape

    def cost(self, ranges):
        n = source_bytes(self.samples, ranges)
        if self.calibration is not None:
            for field in (self.calibration.flats, self.calibration.darks):
                if field is not None:
                    n += source_bytes(field, (range(len(field)), *ranges[1:]))
        return n

    def read(self, ranges, budget):
        spent = 0

        def check(n):
            if n > budget:
                raise ReadLimit

        def read(stack, tile):
            nonlocal spent
            spent += source_bytes(stack, tile)
            check(spent)
            return frame_pixels(stack, tile, budget)

        check(self.cost(ranges))
        if self.calibration is not None:
            return corrected_tile(self.calibration, ranges, read)
        return read(self.samples, ranges)

    def chunks(self, tile, budget):
        """Split statistical slabs until each sample and its calibration fit together."""
        try:
            yield self.read(tile, budget)
            return
        except ReadLimit:
            # Rows partition both sample and calibration reads, including gap interpolation.
            axis = next((i for i in (1, 0, 2) if len(tile[i]) > 1), None)
            if axis is None:
                raise
        r = tile[axis]
        middle = len(r) // 2
        for half in (r[:middle], r[middle:]):
            yield from self.chunks((*tile[:axis], half, *tile[axis + 1 :]), budget)
