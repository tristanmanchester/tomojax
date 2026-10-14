"""A stack of frames in an HDF5 file, read only when, and as far as, it is indexed."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import glob
import os
from pathlib import Path
import re
from typing import Any, cast

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
    chunk_views: int | None = None  # acquisition storage layout, not a caller cache setting
    is_window: bool = False
    _efficient_row_windows: bool = False  # safe to revisit each frame in successive row windows

    @classmethod
    def open(cls, path: str, dataset: str) -> Hdf5Frames:
        """Every frame of ``dataset`` in ``path``."""
        path = str(Path(path).absolute())  # later reads must not depend on the process cwd
        with h5py.File(path, "r") as file:
            data = file[dataset]
            if not isinstance(data, h5py.Dataset) or data.ndim != 3:
                raise ValueError(f"{path}:{dataset} is not a 3-D stack of frames")
            n, rows, cols = (int(s) for s in data.shape)
            layout = cast(Any, data)  # h5py's typing omits acquisition layout metadata
            chunks = layout.chunks
            efficient = (
                not layout.is_virtual
                and layout.id.get_create_plist().get_nfilters() == 0
                and (chunks is None or chunks[0] == 1)
            )
            return cls(
                path, dataset, np.arange(n), rows, cols, np.dtype(data.dtype),
                chunk_views=None if chunks is None else int(chunks[0]),
                _efficient_row_windows=efficient,
            )  # fmt: skip

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.frames), self.rows, self.cols)

    @property
    def ndim(self) -> int:
        return 3

    def __len__(self) -> int:
        return len(self.frames)

    def _shares_file(self, path: Path) -> bool:
        """Check physical containers/storage without reading acquisition pixels."""
        if Path(self.path).samefile(path):
            return True
        with h5py.File(self.path, "r") as file:
            return _dataset_shares_file(cast(Any, file[self.dataset]), path, set())

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
            is_window=self.is_window or (r0, r1, c0, c1) != (0, self.rows, 0, self.cols),
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
        # Cropped reads never reuse single-view chunks. Caching an unfiltered
        # chunk forces the whole frame to be read even for a tiny window. Keep
        # the established bulk path, and multi-view reuse across sparse runs.
        cache = {"rdcc_nbytes": 0} if self.is_window and self.chunk_views == 1 else {}
        with h5py.File(self.path, "r", **cache) as file:
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
            if _read_regular(data, indices, out, r, c):
                return out
            starts = np.concatenate(([0], np.flatnonzero(np.diff(indices) > 1) + 1))
            stops = np.concatenate((starts[1:], [indices.size]))
            for start, stop in zip(starts, stops, strict=True):
                source = (slice(int(indices[start]), int(indices[stop - 1]) + 1), r, c)
                destination = (slice(int(start), int(stop)), slice(None), slice(None))
                data.read_direct(out, source_sel=source, dest_sel=destination)
            return out


def _read_regular(
    data: Any, indices: np.ndarray, out: np.ndarray, rows: slice, cols: slice
) -> bool:
    """Match regular file and memory strides, then compact a bounded workspace."""
    stride = int(indices[1] - indices[0])
    if not np.all(np.diff(indices) == stride):
        return False
    # A strided file selection into dense memory is expensive inside HDF5.
    # Matching its memory stride avoids that cost without reading skipped views.
    # Keep at most one extra workspace, no larger than the result or 16 MiB.
    frame_bytes = out[0].nbytes
    group = min(len(indices), min(16 * 2**20, out.nbytes) // (stride * frame_bytes))
    if group < 2:
        return False
    padded = np.empty(((group - 1) * stride + 1, *out.shape[1:]), out.dtype)
    for first in range(0, len(indices), group):
        stop = min(first + group, len(indices))
        extent = (stop - first - 1) * stride + 1
        source = (slice(int(indices[first]), int(indices[stop - 1]) + 1, stride), rows, cols)
        destination = (slice(0, extent, stride), slice(None), slice(None))
        data.read_direct(padded, source_sel=source, dest_sel=destination)
        out[first:stop] = padded[:extent:stride]
    return True


def _dataset_shares_file(data: Any, target: Path, visited: set[tuple[Path, str]]) -> bool:
    """Follow linked/virtual datasets and external raw storage using header metadata."""
    container = Path(data.file.filename).resolve()
    if container.samefile(target):
        return True
    identity = (container, data.name)
    if identity in visited:
        return False
    visited.add(identity)
    access = data.id.get_access_plist()
    prefix = os.environ.get("HDF5_EXTFILE_PREFIX", os.fsdecode(access.get_efile_prefix()))
    prefix = prefix.replace("${ORIGIN}", str(container.parent))
    if any(
        raw.is_file() and raw.samefile(target)
        for raw in (Path(prefix) / filename for filename, _, _ in data.external or ())
    ):
        return True
    return data.is_virtual and any(
        _virtual_source_shares_file(source, container, access, target, visited)
        for source in data.virtual_sources()
    )


def _virtual_source_shares_file(
    source: Any, container: Path, access: Any, target: Path, visited: set[tuple[Path, str]]
) -> bool:
    for path, blocks in _virtual_source_locations(os.fsdecode(source.file_name), container, access):
        if path.samefile(target):
            return True
        try:
            with h5py.File(path, "r") as file:
                if any(
                    _dataset_shares_file(dataset, target, visited)
                    for dataset in _virtual_source_datasets(
                        file, os.fsdecode(source.dset_name), blocks
                    )
                ):
                    return True
        except OSError:
            # Leave source availability/read errors to the native reader.
            continue
    return False


def _virtual_source_locations(
    filename: str, container: Path, access: Any
) -> list[tuple[Path, dict[str, str]]]:
    """Resolve each materialized VDS block in HDF5's documented search order."""
    if filename == ".":
        return [(container, {})]
    name = Path(filename)
    searches = []
    if name.is_absolute():
        anchor = Path(name.anchor)
        searches.append((anchor, name.relative_to(anchor)))
        name = Path(name.name)
    prefixes = [Path(p) for p in os.environ.get("HDF5_VDS_PREFIX", "").split(os.pathsep) if p]
    access_prefix = os.fsdecode(access.get_virtual_prefix())
    if access_prefix:
        prefixes.append(Path(access_prefix))
    prefixes.extend([container.parent, Path.cwd()])
    searches.extend((base, name) for base in prefixes)
    files: dict[tuple[tuple[str, str], ...], tuple[Path, dict[str, str]]] = {}
    for base, pattern in searches:
        fields = _virtual_fields(re.split(r"(%%|%(?:\d+)?b)", str(pattern)))
        for block, path in _materialized_virtual_files(base, pattern).items():
            bindings = dict(zip(fields, block, strict=True))
            files.setdefault(tuple(bindings.items()), (path, bindings))
    return list(files.values())


def _virtual_source_datasets(file: Any, name: str, blocks: dict[str, str]) -> list[Any]:
    """Resolve dataset-name substitutions, retaining repeated/cross-file block bindings."""
    locations = [(file, blocks)]
    for component in (part for part in name.split("/") if part not in {"", "."}):
        children = []
        for group, bindings in locations:
            if not isinstance(group, h5py.Group):
                continue
            parts = [bindings.get(p, p) for p in re.split(r"(%%|%(?:\d+)?b)", component)]
            _, pattern = _virtual_name_patterns(parts)
            fields = _virtual_fields(parts)
            for key in cast(Any, group):  # h5py's stubs omit native group iteration
                match = re.fullmatch(pattern, key)
                if match is not None:
                    child = group.get(key)
                    values = dict(zip(fields, match.groups(), strict=True))
                    children.append((child, bindings | values))
        locations = children
    return [value for value, _ in locations if isinstance(value, h5py.Dataset)]


def _virtual_fields(parts: list[str]) -> tuple[str, ...]:
    """Distinct substitution fields in the same order as the pattern's capture groups."""
    return tuple(dict.fromkeys(p for p in parts if re.fullmatch(r"%(?:\d+)?b", p)))


def _materialized_virtual_files(base: Path, name: Path) -> dict[tuple[str, ...], Path]:
    """Match source-name blocks without formatting/globbing a literal search prefix."""
    parts = re.split(r"(%%|%(?:\d+)?b)", str(name))
    if len(parts) == 1:
        path = base / name
        return {(): path} if path.is_file() else {}
    wildcard, pattern = _virtual_name_patterns(parts)
    files = {}
    for path in base.glob(wildcard):
        match = re.fullmatch(pattern, str(path.relative_to(base)))
        if match is not None and path.is_file():
            files[match.groups()] = path
    return files


def _virtual_name_patterns(parts: list[str]) -> tuple[str, str]:
    """Capture distinct integer substitutions; repeated placeholders must agree."""
    wildcard, pattern = "", ""
    groups = {}
    for part in parts:
        if re.fullmatch(r"%(?:\d+)?b", part):
            wildcard += "*"
            if part in groups:
                pattern += f"(?P={groups[part]})"
            else:
                group = f"block{len(groups)}"
                groups[part] = group
                pattern += rf"(?P<{group}>0|[1-9]\d*)"
        else:
            literal = part.replace("%%", "%")
            wildcard += glob.escape(literal)
            pattern += re.escape(literal)
    return wildcard, pattern


_FRAME_PATHS = (
    "/entry/instrument/detector/data",
    "/entry/data/projections",
    "/entry/projections",
    "/exchange/data",  # APS Data Exchange
)
_IMAGE_KEY_PATH = "/entry/instrument/detector/image_key"


@dataclass(frozen=True)
class LocatedFrames:
    """An HDF5 file's frames, with each frame's ``image_key`` and angle (degrees) when found.

    ``flats`` and ``darks`` are stacks of their own beside the frames, as in
    APS Data Exchange (``data_white``, ``data_dark``), when the file keeps
    them apart rather than marking them in ``image_key``.
    """

    stack: Hdf5Frames
    image_key: np.ndarray | None
    angles: np.ndarray | None
    flats: Hdf5Frames | None = None
    darks: Hdf5Frames | None = None

    @property
    def raw(self) -> bool:
        """Whether the file holds flat or dark frames: its frames are detector counts."""
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
        key_at = key_at or _only(file, "image_key", per_frame, path, validate_unmatched=True)
        angles_at = angles_path or next(
            (p for p in ROTATION_ANGLE_PATHS if _dataset(file, p) is not None), None
        )
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
        angles = (
            None if angles_at is None else _degrees(file, angles_at, path, frames=frames, key=key)
        )
        flats, darks = (
            Hdf5Frames.open(path, beside[n])
            if (key is None or not np.any(key == label)) and _dataset(file, beside[n])
            else None
            for n, label in (("data_white", 1), ("data_dark", 2))
        )
    return LocatedFrames(Hdf5Frames.open(path, where), key, angles, flats, darks)


def _dataset(file: h5py.File, where: str) -> h5py.Dataset | None:
    found = file.get(where)
    return found if isinstance(found, h5py.Dataset) else None


def _read(file: h5py.File, where: str, path: str) -> np.ndarray:
    found = _dataset(file, where)
    if found is None:
        raise KeyError(f"{path} has no dataset {where!r}")
    return np.asarray(found[()])


def _degrees(
    file: h5py.File, where: str, path: str, *, frames: int, key: np.ndarray | None
) -> np.ndarray:
    values = _read(file, where, path)
    units = file[where].attrs.get("units", b"")
    units = units.decode() if isinstance(units, bytes) else str(units)
    return angles_in_degrees(values, units, frames=frames, path=path, image_key=key)


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
