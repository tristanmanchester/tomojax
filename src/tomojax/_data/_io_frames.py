"""A stack of frames in an HDF5 file, read only when, and as far as, it is indexed."""

from __future__ import annotations

from collections.abc import Callable
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
    in the file's dtype; :meth:`frames_at` and :meth:`window` narrow the stack
    without reading.
    """

    path: str
    dataset: str
    frames: np.ndarray  # indices into the dataset's first axis, increasing
    rows: int
    cols: int
    dtype: np.dtype
    first_row: int = 0  # the window of each frame read: rows and columns from these
    first_col: int = 0

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

    def window(self, rows: slice, cols: slice) -> Hdf5Frames:
        """The stack of this block of ``rows`` and ``cols`` of each frame, unread."""
        r0, r1, rstep = rows.indices(self.rows)
        c0, c1, cstep = cols.indices(self.cols)
        if rstep != 1 or cstep != 1 or r1 <= r0 or c1 <= c0:
            raise ValueError("a window of frames is a non-empty block of rows and columns")
        return replace(
            self,
            rows=r1 - r0,
            cols=c1 - c0,
            first_row=self.first_row + r0,
            first_col=self.first_col + c0,
        )

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
            r = slice(self.first_row, self.first_row + self.rows)
            c = slice(self.first_col, self.first_col + self.cols)
            first, last = int(indices[0]), int(indices[-1]) + 1
            if last - first == indices.size:  # contiguous: one hyperslab
                return np.asarray(data[first:last, r, c])
            return np.asarray(data[indices, r, c])


_FRAME_PATHS = ("/entry/instrument/detector/data", "/entry/data/projections", "/entry/projections")
_IMAGE_KEY_PATH = "/entry/instrument/detector/image_key"
_ANGLE_PATH = "/entry/sample/transformations/rotation_angle"


@dataclass(frozen=True)
class LocatedFrames:
    """An HDF5 file's frames, with each frame's ``image_key`` and angle (degrees) when found."""

    stack: Hdf5Frames
    image_key: np.ndarray | None
    angles: np.ndarray | None


def locate_frames(
    path: str,
    *,
    data_path: str | None = None,
    image_key_path: str | None = None,
    angles_path: str | None = None,
) -> LocatedFrames:
    """Find the stack of frames in an HDF5 file, and its ``image_key`` and angles.

    Each is read from the path given, else the NXtomo path, else the file's
    only dataset of that name (``data`` of three dimensions, ``image_key``,
    ``rotation_angle``) with one entry per frame. Angles in radians (a
    ``units`` attribute) are converted to degrees.
    """
    with h5py.File(path, "r") as file:
        where = data_path or next((p for p in _FRAME_PATHS if _dataset(file, p) is not None), None)
        where = where or _only(file, "data", lambda d: len(d.shape) == 3, path)
        stack = None if where is None else _dataset(file, where)
        if data_path is not None and stack is None:
            raise KeyError(f"{path} has no dataset {data_path!r}")
        if where is None or stack is None:
            raise KeyError(f"{path}: no stack of frames found; name its dataset (data_path)")
        frames = int(stack.shape[0])

        def per_frame(item: h5py.Dataset) -> bool:
            return item.shape == (frames,)

        key_at = image_key_path or (_IMAGE_KEY_PATH if _dataset(file, _IMAGE_KEY_PATH) else None)
        key_at = key_at or _only(file, "image_key", per_frame, path)
        angles_at = angles_path or (_ANGLE_PATH if _dataset(file, _ANGLE_PATH) else None)
        angles_at = angles_at or _only(file, "rotation_angle", per_frame, path)
        key = None if key_at is None else np.asarray(_read(file, key_at, path), np.int32)
        angles = None if angles_at is None else _degrees(file, angles_at, path)
    for name, values in (("image_key", key), ("angles", angles)):
        if values is not None and values.shape != (frames,):
            raise ValueError(f"{path}: {name} has {values.shape} entries for {frames} frames")
    return LocatedFrames(Hdf5Frames.open(path, where), key, angles)


def _dataset(file: h5py.File, where: str) -> h5py.Dataset | None:
    found = file.get(where)
    return found if isinstance(found, h5py.Dataset) else None


def _read(file: h5py.File, where: str, path: str) -> np.ndarray:
    found = _dataset(file, where)
    if found is None:
        raise KeyError(f"{path} has no dataset {where!r}")
    return np.asarray(found[()])


def _degrees(file: h5py.File, where: str, path: str) -> np.ndarray:
    values = np.asarray(_read(file, where, path), np.float64).reshape(-1)
    units = file[where].attrs.get("units", b"")
    units = units.decode() if isinstance(units, bytes) else str(units)
    return np.degrees(values) if units.strip().lower() in {"rad", "radian", "radians"} else values


def _only(
    file: h5py.File, name: str, fits: Callable[[h5py.Dataset], bool], path: str
) -> str | None:
    """The path of the file's only dataset called ``name`` that ``fits``, if any."""
    found: list[str] = []

    def visit(where: str, item: object) -> None:
        if isinstance(item, h5py.Dataset) and where.rsplit("/", 1)[-1] == name and fits(item):
            found.append("/" + where)

    visitable: Any = file
    visitable.visititems(visit)
    if len(found) > 1:
        raise KeyError(f"{path} has several {name!r} datasets ({', '.join(found)}): name one")
    return found[0] if found else None
