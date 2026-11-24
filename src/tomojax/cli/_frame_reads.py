"""Bounded detector reads for the worker's logical (view, row, column) selections."""

from itertools import product
from typing import cast

import h5py
import numpy as np

from tomojax.core.files import read_hdf5


def frame_pixels(data, ranges, budget):
    views, rows, cols = ranges
    selected = slice(views.start, views.stop, views.step)
    if rows.step == cols.step == 1:
        return data.window(slice(rows.start, rows.stop), slice(cols.start, cols.stop))[selected]
    if hasattr(data, "dataset"):
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

    # Other frame stacks expose contiguous detector windows. Account for all
    # pixels between requested rows and columns when sizing each source read.
    capacity = budget // np.dtype(data.dtype).itemsize
    block = [1, 1, 1]
    for axis, step in ((2, cols.step), (1, rows.step), (0, 1)):
        block[axis] = min(len(ranges[axis]), 1 + (capacity - 1) // step)
        capacity //= 1 + (block[axis] - 1) * step
    out = np.empty(tuple(map(len, ranges)), data.dtype)
    starts = (range(0, n, b) for n, b in zip(out.shape, block, strict=True))
    for origin in product(*starts):
        target = tuple(slice(i, i + n) for i, n in zip(origin, block, strict=True))
        v, r, c = (axis[key] for axis, key in zip(ranges, target, strict=True))
        detector = data.window(slice(r.start, r[-1] + 1), slice(c.start, c[-1] + 1))
        out[target] = detector[slice(v.start, v.stop, v.step)][:, :: r.step, :: c.step]
    return out
