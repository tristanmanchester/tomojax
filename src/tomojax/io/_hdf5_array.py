"""Bounded parallel decoding behind an ordinary HDF5 array-read contract."""

from __future__ import annotations

from itertools import product
from math import prod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import h5py


def read_hdf5_array(
    dataset: h5py.Dataset, selection: tuple[slice, ...] | None = None
) -> np.ndarray:
    """Read an array without changing its dtype, axes, filters or file contents.

    ``dataset`` remains owned by the caller and must stay open until this call
    returns. Large, unit-stride slices of read-only, fixed-size gzip chunks use two
    decode workers; HDF5 access itself stays on the calling thread. A bounded
    four-job queue avoids allocating any full-sized temporary stack.
    Other layouts, filters and slices use the ordinary HDF5 reader.

    Parameters
    ----------
    dataset : h5py.Dataset
        Open dataset to read from.
    selection : tuple of slice or None, optional
        Index of the region to read; None reads the whole dataset.

    Returns
    -------
    numpy.ndarray
        The selected values, in the dataset's dtype and axis order.
    """
    key = tuple(slice(None) for _ in dataset.shape) if selection is None else selection
    chunks = dataset.chunks
    dtype = dataset.dtype
    if (
        chunks is None
        or dataset.file.mode != "r"
        or dataset.compression != "gzip"
        or dataset.id.get_create_plist().get_nfilters() != 1
        or dtype.hasobject
        or len(key) != dataset.ndim
        or any(not isinstance(part, slice) or part.indices(1)[2] != 1 for part in key)
    ):
        return np.asarray(dataset[key])
    bounds = [part.indices(size)[:2] for part, size in zip(key, dataset.shape, strict=True)]
    shape = tuple(max(stop - start, 0) for start, stop in bounds)
    axes = [
        range(start // chunk * chunk, stop, chunk)
        for (start, stop), chunk in zip(bounds, chunks, strict=True)
    ]
    if (
        prod(shape) * dtype.itemsize < 1024**2
        or prod(len(axis) for axis in axes) < 2
        or prod(chunks) * dtype.itemsize > 64 * 1024**2
    ):
        return np.asarray(dataset[key])

    from collections import deque
    from concurrent.futures import Future, ThreadPoolExecutor
    import zlib

    result = np.empty(shape, dtype)

    def fill(
        encoded: bytes, mask: int, source: tuple[slice, ...], target: tuple[slice, ...]
    ) -> None:
        decoded = encoded if mask & 1 else zlib.decompress(encoded)
        block = np.frombuffer(decoded, dtype).reshape(chunks)
        result[target] = block[source]

    pending: deque[Future[None]] = deque()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for origin in product(*axes):
            overlap = [
                (max(start, offset), min(stop, offset + chunk))
                for (start, stop), offset, chunk in zip(bounds, origin, chunks, strict=True)
            ]
            source = tuple(
                slice(lo - offset, hi - offset)
                for (lo, hi), offset in zip(overlap, origin, strict=True)
            )
            target = tuple(
                slice(lo - start, hi - start)
                for (lo, hi), (start, _) in zip(overlap, bounds, strict=True)
            )
            try:
                mask, encoded = dataset.id.read_direct_chunk(origin)
            except (MemoryError, OSError, RuntimeError):
                # Raw reads of unallocated chunks can raise even MemoryError.
                # Query allocation only on failure: per-chunk metadata probes
                # are expensive on large chunk indexes. Real read errors escape.
                if dataset.id.get_chunk_info_by_coord(origin).byte_offset is not None:
                    raise
                result[target] = dataset.fillvalue
                continue
            pending.append(pool.submit(fill, encoded, mask, source, target))
            if len(pending) >= 4:
                pending.popleft().result()
        for future in pending:
            future.result()
    return result
