"""Parallel positional reads of unfiltered HDF5 frame stacks, at their stored byte offsets."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import os
from pathlib import Path
import threading

import h5py
import numpy as np

# Parallel file systems (GPFS, Lustre) need many requests in flight; local disks
# saturate with fewer, and the extra threads only wait.
_READERS = 32
# Chunk offsets are listed once per file state for datasets of up to this many chunks.
_LISTED_CHUNKS = 1 << 18
_listed: dict[tuple[str, str, int, int], dict[tuple[int, ...], int]] = {}
_listing = threading.Lock()


@dataclass(frozen=True)
class _Run:
    """One frame's part of one stored chunk, read from ``position`` into ``target``.

    ``position`` is None for a chunk never written. The run spans ``elements``
    stored values from the window's first to its last, with ``stride`` values
    per stored chunk row.
    """

    position: int | None
    elements: int
    stride: int
    target: np.ndarray


def read_stored(
    data: h5py.Dataset, indices: np.ndarray, rows: slice, cols: slice
) -> np.ndarray | None:
    """Frames ``indices`` of ``data`` in a window of rows and columns, or None if not stored raw.

    A dataset whose values lie unfiltered in its own file, chunked or contiguous,
    is read by positional reads at the byte offsets HDF5 reports, many at once,
    outside HDF5's single-threaded path. Each frame's part of a chunk is one read
    from the first to the last byte of the window, so a band of rows of a
    whole-width chunk reads only those rows. Chunks never written hold the fill
    value, as HDF5 returns them. Filtered, virtual, compact or externally stored
    data, and files on drivers other than the default, return None: only HDF5
    can turn their stored bytes into values.

    Parameters
    ----------
    data : h5py.Dataset
        Open 2-D ``(frame, column)`` or 3-D ``(frame, row, column)`` dataset; a 2-D
        dataset has one detector row.
    indices : numpy.ndarray
        Frame indices to read, in any order.
    rows, cols : slice
        Unit-stride detector window in stored coordinates.

    Returns
    -------
    numpy.ndarray or None
        ``(len(indices), rows, columns)`` in the dataset's dtype and byte order,
        or None when the stored bytes are not the values.

    Raises
    ------
    OSError
        If the file is shorter than its HDF5 metadata says.
    """
    plist = data.id.get_create_plist()
    if (
        data.is_virtual
        or plist.get_layout() == h5py.h5d.COMPACT
        or plist.get_nfilters()
        or plist.get_external_count()
        or data.dtype.hasobject
        or data.file.driver != "sec2"
    ):
        return None
    shape = (data.shape[0], 1, data.shape[1]) if data.ndim == 2 else data.shape
    r0, r1, _ = rows.indices(shape[1])
    c0, c1, _ = cols.indices(shape[2])
    out = np.empty((len(indices), r1 - r0, c1 - c0), data.dtype)
    # A memoryview lists typed ints; ndarray.tolist() is untyped.
    frames = memoryview(np.ascontiguousarray(indices, np.int64)).tolist()
    runs = _runs(data, frames, (r0, r1), (c0, c1), out)
    where = f"{data.file.filename}:{data.name}"
    fill: object = data.fillvalue
    descriptor = os.open(data.file.filename, os.O_RDONLY)

    def read(group: list[_Run]) -> None:
        for run in group:
            if run.position is None:  # a chunk never written
                run.target[...] = fill
            else:
                _read_run(descriptor, run, where)

    try:
        workers = max(1, min(_READERS, len(runs)))
        size = max(1, -(-len(runs) // workers))
        groups = [runs[i : i + size] for i in range(0, len(runs), size)]
        if len(groups) <= 1:
            read(runs)
        else:
            with ThreadPoolExecutor(len(groups), thread_name_prefix="tomojax-hdf5") as pool:
                for done in [pool.submit(read, group) for group in groups]:
                    done.result()
    finally:
        os.close(descriptor)
    return out


def _runs(
    data: h5py.Dataset,
    frames: list[int],
    rows: tuple[int, int],
    cols: tuple[int, int],
    out: np.ndarray,
) -> list[_Run]:
    """The stored byte runs that fill ``out`` with ``frames`` in a window of ``rows``, ``cols``."""
    item = data.dtype.itemsize
    starts: dict[tuple[int, int, int], int | None] = {}
    listed = None
    if data.chunks is not None:
        chunk = (data.chunks[0], 1, data.chunks[1]) if data.ndim == 2 else data.chunks
        listed = _chunk_offsets(data)
    else:  # one frame after another: address each frame as a chunk
        chunk = (1, 1, data.shape[1]) if data.ndim == 2 else (1, *data.shape[1:])
        base = data.id.get_offset()  # None until the data are first written
        for frame in frames:
            starts[frame, 0, 0] = (
                None if base is None else base + frame * chunk[1] * chunk[2] * item
            )
    (r0, r1), (c0, c1) = rows, cols
    runs: list[_Run] = []
    for k, frame in enumerate(frames):
        f0 = frame // chunk[0] * chunk[0]
        for top in range(r0 // chunk[1] * chunk[1], r1, chunk[1]):
            for left in range(c0 // chunk[2] * chunk[2], c1, chunk[2]):
                origin = (f0, top, left)
                if origin not in starts:
                    coord = (f0, left) if data.ndim == 2 else origin
                    starts[origin] = (
                        data.id.get_chunk_info_by_coord(coord).byte_offset
                        if listed is None
                        else listed.get(coord)
                    )
                lr0, lr1 = max(r0, top) - top, min(r1, top + chunk[1]) - top
                lc0, lc1 = max(c0, left) - left, min(c1, left + chunk[2]) - left
                first = ((frame - f0) * chunk[1] + lr0) * chunk[2] + lc0
                start = starts[origin]
                target = out[k, top + lr0 - r0 : top + lr1 - r0, left + lc0 - c0 : left + lc1 - c0]
                position = None if start is None else start + first * item
                runs.append(
                    _Run(position, (lr1 - lr0 - 1) * chunk[2] + lc1 - lc0, chunk[2], target)
                )
    return runs


def _chunk_offsets(data: h5py.Dataset) -> dict[tuple[int, ...], int] | None:
    """Byte offset of every written chunk of ``data`` by its origin, or None if too many to list.

    Asking HDF5 for one chunk walks its chunk index; for the thousands of frames of a
    scan that takes seconds on a network file system, at every read. One pass over the
    index lists them all, and the list is kept while the file's size and modification
    time stay the same.
    """
    assert data.chunks is not None
    if math.prod(-(-n // c) for n, c in zip(data.shape, data.chunks, strict=True)) > _LISTED_CHUNKS:
        return None
    status = Path(data.file.filename).stat()
    key = (data.file.filename, str(data.name), status.st_size, status.st_mtime_ns)
    with _listing:
        offsets = _listed.get(key)
        if offsets is None:
            found: dict[tuple[int, ...], int] = {}
            data.id.chunk_iter(lambda chunk: found.update({chunk.chunk_offset: chunk.byte_offset}))
            offsets = found
            while len(_listed) >= 8:
                del _listed[next(iter(_listed))]
            _listed[key] = offsets
    return offsets


def _read_run(descriptor: int, run: _Run, where: str) -> None:
    """Read one stored run of the dataset ``where`` from the open file ``descriptor``."""
    assert run.position is not None
    target = run.target
    # A window as wide as its chunk is one contiguous run of the output.
    direct = bool(target.flags.c_contiguous) and target.size == run.elements
    buffer = target if direct else np.empty(run.elements, target.dtype)
    view, done = memoryview(buffer).cast("B"), 0
    while done < len(view):
        count = os.preadv(descriptor, [view[done:]], run.position + done)
        if count == 0:
            raise OSError(f"{where}: the file ends before byte {run.position + len(view)}")
        done += count
    if not direct:
        strides = (run.stride * target.itemsize, target.itemsize)
        target[...] = np.lib.stride_tricks.as_strided(buffer, target.shape, strides)
