"""Read one plane in storage order and encode it for the worker pipe."""

from __future__ import annotations

from itertools import product
import math
from pathlib import Path

import h5py
import imagecodecs
import numpy as np
import numpy.typing as npt

from tomojax.cli import envelope as e
from tomojax.cli._planes import READ_BYTES, window
from tomojax.core.input_files import open_input_hdf5, reading
from tomojax.wire.arrays import Hdf5


def selection(data: np.ndarray | h5py.Dataset, cmd: e.Fetch) -> np.ndarray:
    ranges = window(cmd.shape, cmd)
    shape = tuple(map(len, ranges))
    if len(shape) not in (2, 3) or (len(shape) == 3 and min(shape) != 1):
        raise ValueError("fetch needs a plane")
    if math.prod(shape) * (4 if cmd.encoding == "float32" else 1) > READ_BYTES:
        raise ValueError("decoded plane exceeds 16 MiB")
    if data.shape != cmd.shape:
        if len(cmd.shape) != 3 or cmd.shape[1] != 1 or data.shape != cmd.shape[::2]:
            raise ValueError("stored shape differs from the descriptor")
        ranges = ranges[::2]
    if data.dtype.kind not in "biuf":
        raise ValueError(f"expected real numeric data, got {data.dtype}")
    if isinstance(data, np.ndarray):
        keys = tuple(slice(r.start, r.stop, r.step) for r in ranges)
        return data[keys].reshape(shape)

    # Read dense rectangles within a bounded scratch size, then stride in RAM.
    capacity: int = READ_BYTES // data.dtype.itemsize
    tile: list[int] = []
    for r in reversed(ranges):
        count = min(len(r), 1 + (capacity - 1) // r.step)
        tile.append(count)
        capacity //= 1 + (count - 1) * r.step
    tile.reverse()
    result = np.empty(tuple(map(len, ranges)), dtype=data.dtype)
    steps = tuple(slice(None, None, r.step) for r in ranges)
    for origin in product(*(range(0, n, k) for n, k in zip(result.shape, tile, strict=True))):
        dest = tuple(slice(i, i + k) for i, k in zip(origin, tile, strict=True))
        block = [r[d] for r, d in zip(ranges, dest, strict=True)]
        keys = tuple(slice(r.start, r[-1] + 1) for r in block)
        result[dest] = data[keys][steps]
    return result.reshape(shape)


def gray_levels(
    samples: npt.NDArray[np.float32 | np.float64], limits: tuple[float, float] | None
) -> tuple[np.ndarray, tuple[float, float]]:
    finite = np.isfinite(samples)
    if limits is None:
        limits = (
            (
                float(np.min(samples, where=finite, initial=np.inf)),
                float(np.max(samples, where=finite, initial=-np.inf)),
            )
            if finite.any()
            else (0.0, 0.0)
        )
    low, high = limits
    if not (math.isfinite(low) and math.isfinite(high) and low <= high):
        raise ValueError("window needs two ordered finite limits")
    if low == high:
        return np.zeros(samples.shape, dtype=np.uint8), limits
    mapped = np.clip(samples, low, high) - low
    mapped *= 255 / (high - low)
    mapped[~finite] = 0
    return np.rint(mapped).astype(np.uint8), limits


def fetch(cmd: e.Fetch) -> tuple[e.Pixels, bytes | bytearray | memoryview]:
    path = Path(cmd.storage.path)
    with reading(path):
        if isinstance(cmd.storage, Hdf5):
            with open_input_hdf5(path) as file:
                dataset: object = file[cmd.storage.dataset]
                if not isinstance(dataset, h5py.Dataset):
                    raise ValueError("storage must name an HDF5 dataset")
                data = selection(dataset, cmd)
        else:
            stored = np.lib.format.open_memmap(path, mode="r")
            data = selection(stored, cmd)
        shape = data.shape
        limits = None
        if cmd.encoding == "float32":
            flat = np.ascontiguousarray(data, dtype="<f4").reshape(-1)
            payload = imagecodecs.zstd_encode(imagecodecs.byteshuffle_encode(flat), level=1)
        else:
            gray, limits = gray_levels(
                np.asarray(data, dtype=np.result_type(data.dtype, np.float32)), cmd.window
            )
            if gray.ndim == 3:
                gray = gray.squeeze(gray.shape.index(1))
            payload = imagecodecs.jpeg8_encode(gray, level=60, optimize=True)
    return e.Pixels(shape, cmd.encoding, len(payload), limits), payload
