"""A stack of frames in an HDF5 file, read only when, and as far as, it is indexed."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import logging
from pathlib import Path
from typing import Any, ClassVar, cast

import h5py
import numpy as np

from ._io_acquisition import ROTATION_ANGLE_PATHS, angles_in_degrees, checked_image_key


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
    _efficient_row_windows: bool = False  # a window of rows reads only those rows
    _bulk_row_reads: ClassVar[bool] = True  # many views in one HDF5 request

    @classmethod
    def open(cls, path: str, dataset: str) -> Hdf5Frames:
        """Every frame of ``dataset`` in ``path``."""
        path = str(Path(path).absolute())  # later reads must not depend on the working directory
        with h5py.File(path, "r") as file:
            data = file[dataset]
            if not isinstance(data, h5py.Dataset) or data.ndim != 3:
                raise ValueError(f"{path}:{dataset} is not a 3-D stack of frames")
            n, rows, cols = (int(s) for s in data.shape)
            layout = cast(Any, data)  # h5py's typing omits the storage layout
            # Unfiltered data supports direct reads of partial chunks. __getitem__
            # bypasses the chunk cache so narrow windows do not fetch whole frames.
            windows = not layout.is_virtual and layout.id.get_create_plist().get_nfilters() == 0
            return cls(path, dataset, np.arange(n), rows, cols, np.dtype(data.dtype), 0, 0, windows)

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

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        """Read this stack explicitly with ``np.asarray`` (always allocates host storage)."""
        if copy is False:
            raise ValueError("Hdf5Frames cannot be converted to an array without copying")
        return np.asarray(self[:], dtype=dtype)

    def __getitem__(self, which: slice | int | np.ndarray) -> np.ndarray:
        if isinstance(which, int | np.integer):
            # Array indexing preserves NumPy's negative-index and bounds rules;
            # a slice(-1, 0) would silently select no frames instead of the last.
            return self[np.asarray([which], np.int64)][0]
        indices = np.asarray(self.frames[which], np.int64)
        if indices.size and np.any(np.diff(indices) <= 0):
            raise ValueError("Hdf5Frames reads frames in increasing order only")
        # Chunk caching can amplify a narrow read to the entire chunk even with
        # no compression. Filtered/virtual datasets retain their normal cache.
        cache = 0 if self._efficient_row_windows else None
        with h5py.File(self.path, "r", rdcc_nbytes=cache) as file:
            data: Any = file[self.dataset]
            if indices.size == 0:
                return np.empty((0, self.rows, self.cols), self.dtype)
            r = slice(self.first_row, self.first_row + self.rows)
            c = slice(self.first_col, self.first_col + self.cols)
            first, last = int(indices[0]), int(indices[-1]) + 1
            if last - first == indices.size:  # contiguous: one hyperslab
                return np.asarray(data[first:last, r, c])
            # h5py fancy indexing is far slower than reading each run of frames.
            out = np.empty((indices.size, self.rows, self.cols), self.dtype)
            starts = np.concatenate(([0], np.flatnonzero(np.diff(indices) > 1) + 1))
            stops = np.concatenate((starts[1:], [indices.size]))
            for start, stop in zip(starts, stops, strict=True):
                source = (slice(int(indices[start]), int(indices[stop - 1]) + 1), r, c)
                destination = (slice(int(start), int(stop)), slice(None), slice(None))
                data.read_direct(out, source_sel=source, dest_sel=destination)
            return out


_FRAME_PATHS = (
    "/entry/instrument/detector/data",
    "/entry/data/projections",
    "/entry/projections",
    "/exchange/data",  # APS Data Exchange
)
_IMAGE_KEY_PATH = "/entry/instrument/detector/image_key"


def _gda_paths(file: h5py.File) -> tuple[str, str, str] | None:
    """A coherent GDA imaging acquisition, including stripped NeXus exports.

    The rotation command, not the encoder's raw counts, supplies the angles.
    Matching the stack, key and command together avoids unrelated NXdata
    signals and duplicate instrument links. Entry names are not prescribed.
    """
    found: list[tuple[str, str, str]] = []
    entries: Any = file  # h5py's File stub omits inherited Group methods.
    for name, entry in entries.items():
        if not isinstance(entry, h5py.Group):
            continue
        prefix = f"/{name}"
        data = f"{prefix}/imaging/data"
        stack = _dataset(file, data)
        if stack is None or stack.ndim != 3:
            continue
        key = next(
            (
                f"{prefix}/{p}"
                for p in (
                    "tomo_entry/instrument/detector/image_key",
                    "instrument/EtherCAT/image_key",
                )
                if _dataset(file, f"{prefix}/{p}") is not None
            ),
            None,
        )
        angle = next(
            (
                f"{prefix}/{p}"
                for p in (
                    "imaging/smaract_zrot_value_set",
                    "imaging_sum/smaract_zrot_value_set",
                    "imaging_sum/smaract_zrot",
                )
                if _dataset(file, f"{prefix}/{p}") is not None
            ),
            None,
        )
        if key is not None and angle is not None:
            found.append((data, key, angle))
    if len(found) > 1:
        raise KeyError("Several GDA imaging acquisitions found: name their data/key/angle paths")
    return found[0] if found else None


def _gda_metadata(
    file: h5py.File,
    gda: tuple[str, str, str] | None,
    where: str,
    key_at: str | None,
    angles_at: str | None,
) -> tuple[dict[str, str] | None, str | None]:
    """Record discovery and the known GDA angular-command unit correction."""
    acquisition = None
    units_override = None
    if gda is not None and key_at is not None and angles_at is not None:
        acquisition = {
            "layout": "diamond_gda",
            "data_path": where,
            "image_key_path": key_at,
            "angles_path": angles_at,
        }
        units = file[angles_at].attrs.get("units", "")
        units = units.decode() if isinstance(units, bytes) else str(units)
        # GDA's SmarAct angular command can inherit the positioner's linear
        # unit label. Only this identified rotation command has that quirk.
        if angles_at == gda[2] and angles_at.endswith("/smaract_zrot_value_set") and units == "mm":
            units_override = "degree"
            acquisition.update(angle_units_recorded=units, angle_units="degree")
            logging.getLogger(__name__).warning(
                "%s:%s: GDA angular command is labelled mm; interpreting it as degrees",
                cast(Any, file).filename,
                angles_at,
            )
    return acquisition, units_override


@dataclass(frozen=True)
class LocatedFrames:
    """Located HDF5 frames with optional frame labels and rotation angles.

    ``flats`` and ``darks`` are stacks of their own beside the frames, as in
    APS Data Exchange (``data_white``, ``data_dark``), when the file keeps
    them apart rather than marking them in ``image_key``.

    Parameters
    ----------
    stack : lazy HDF5 stack
        Located detector-frame stack, read only as indexed.
    image_key : numpy.ndarray or None
        Frame labels when present: sample, flat, dark or other.
    angles : numpy.ndarray or None
        One angle per frame, in degrees, when present.
    flats : lazy HDF5 stack or None, optional
        Separate open-beam frames, when not interleaved with the sample.
    darks : lazy HDF5 stack or None, optional
        Separate beam-off frames.
    acquisition : dict or None, optional
        Discovered acquisition paths and unit corrections.
    """

    stack: Hdf5Frames
    image_key: np.ndarray | None
    angles: np.ndarray | None
    flats: Hdf5Frames | None = None
    darks: Hdf5Frames | None = None
    acquisition: dict[str, str] | None = None
    image_key_path: str | None = None
    angles_path: str | None = None

    @property
    def raw(self) -> bool:
        """Whether the file holds flat or dark frames: its frames are detector counts.

        Returns
        -------
        bool
            True when flat or dark frames mark the data as detector counts.
        """
        marked = self.image_key is not None and bool(np.isin(self.image_key, (1, 2)).any())
        return marked or self.flats is not None or self.darks is not None


def locate_frames(
    path: str,
    *,
    data_path: str | None = None,
    image_key_path: str | None = None,
    angles_path: str | None = None,
) -> LocatedFrames:
    """Find the stack of frames in an HDF5 file, and its ``image_key`` and angles.

    Each is read from the path given, else the NXtomo (or Data Exchange) path,
    else the file's only dataset of that name (``data`` of three dimensions,
    ``image_key``, ``rotation_angle``) with one entry per frame. A uniquely
    named key or angle dataset with the wrong shape is rejected, not
    treated as missing. Flats and darks kept beside the frames
    (``data_white``, ``data_dark``) are found when ``image_key`` marks none of that
    kind, and so are angles there (``theta``). Angles in radians (a ``units``
    attribute) are converted to degrees.

    Parameters
    ----------
    path : str
        HDF5 acquisition file.
    data_path : str or None, optional
        Explicit frame dataset path; None uses discovery.
    image_key_path : str or None, optional
        Explicit frame-label dataset path; None uses discovery.
    angles_path : str or None, optional
        Explicit rotation-angle dataset path; None uses discovery.

    Returns
    -------
    LocatedFrames
        Lazy frame stacks plus frame labels, angles and acquisition metadata.

    Raises
    ------
    KeyError
        If frames or an explicitly requested dataset cannot be found.
    ValueError
        If discovery is ambiguous or per-frame metadata are malformed.
    OSError
        If the file cannot be opened.
    """
    with h5py.File(path, "r") as file:
        where = data_path or next((p for p in _FRAME_PATHS if _dataset(file, p) is not None), None)
        gda = _gda_paths(file) if where is None else None
        where = where or (gda[0] if gda is not None else None)
        gda = gda if gda is not None and where == gda[0] else None
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
        key_at = key_at or (gda[1] if gda is not None else None)
        key_at = key_at or _only(file, "image_key", per_frame, path, validate_unmatched=True)
        angles_at = angles_path or next(
            (p for p in ROTATION_ANGLE_PATHS if _dataset(file, p) is not None), None
        )
        angles_at = angles_at or (gda[2] if gda is not None else None)
        angles_at = angles_at or _only(file, "rotation_angle", per_frame, path)
        parent = where.rsplit("/", 1)[0] or "/"
        beside = {n: f"{parent.rstrip('/')}/{n}" for n in ("data_white", "data_dark", "theta")}
        angles_at = angles_at or (beside["theta"] if _dataset(file, beside["theta"]) else None)
        # A known Data Exchange angle path takes priority over unmatched names.
        angles_at = angles_at or _only(
            file, "rotation_angle", per_frame, path, validate_unmatched=True
        )
        key = (
            None
            if key_at is None
            else checked_image_key(_read(file, key_at, path), frames=frames, path=path)
        )
        acquisition, units_override = _gda_metadata(file, gda, where, key_at, angles_at)
        angles = (
            None
            if angles_at is None
            else _degrees(
                file, angles_at, path, frames=frames, key=key, units_override=units_override
            )
        )
        flats, darks = (
            Hdf5Frames.open(path, beside[n])
            if (key is None or not np.any(key == label)) and _dataset(file, beside[n])
            else None
            for n, label in (("data_white", 1), ("data_dark", 2))
        )
    return LocatedFrames(
        Hdf5Frames.open(path, where), key, angles, flats, darks, acquisition, key_at, angles_at
    )


def _dataset(file: h5py.File, where: str) -> h5py.Dataset | None:
    found = file.get(where)
    return found if isinstance(found, h5py.Dataset) else None


def _read(file: h5py.File, where: str, path: str) -> np.ndarray:
    found = _dataset(file, where)
    if found is None:
        raise KeyError(f"{path} has no dataset {where!r}")
    return np.asarray(found[()])


def _degrees(
    file: h5py.File,
    where: str,
    path: str,
    *,
    frames: int,
    key: np.ndarray | None,
    units_override: str | None = None,
) -> np.ndarray:
    values = _read(file, where, path)
    units = file[where].attrs.get("units", b"")
    units = units.decode() if isinstance(units, bytes) else str(units)
    return angles_in_degrees(
        values, units_override or units, frames=frames, path=path, image_key=key
    )


def _only(
    file: h5py.File,
    name: str,
    fits: Callable[[h5py.Dataset], bool],
    path: str,
    *,
    validate_unmatched: bool = False,
) -> str | None:
    """Find a fitting named dataset, or one unmatched candidate to validate.

    Shape still disambiguates different detectors. With ``validate_unmatched``,
    a malformed uniquely named acquisition dataset is not mistaken for absent
    metadata; its caller validates it. More than one candidate is ambiguous.
    """
    found: list[str] = []
    named: list[str] = []

    def visit(where: str, item: object) -> None:
        if isinstance(item, h5py.Dataset) and where.rsplit("/", 1)[-1] == name:
            named.append("/" + where)
            if fits(item):
                found.append("/" + where)

    visitable: Any = file
    visitable.visititems(visit)
    if validate_unmatched and not found:
        found = named
    if len(found) > 1:
        raise KeyError(f"{path} has several {name!r} datasets ({', '.join(found)}): name one")
    return found[0] if found else None
