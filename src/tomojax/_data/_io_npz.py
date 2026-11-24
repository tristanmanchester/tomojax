from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
import json
from math import prod
from pathlib import Path
import shutil
import tempfile
from zipfile import ZIP_STORED, ZipFile

import numpy as np

from tomojax import progress
from tomojax.core.files import read_path, replaced_when_complete

from ._frame_stack import FrameStack
from ._io_helpers import sizes_of
from ._io_types import LoadedDataset, LoadedNXTomo, NXTomoMetadata

# The file's key for the view angles (``NXTomoMetadata.angles``); earlier
# versions wrote it, so it stays.
_ANGLES_KEY = "thetas_deg"
_AXES_KEY = "volume_axes_order"
_JSON_FIELDS = {
    "grid",
    "detector",
    "source",
    "geometry_meta",
    "align_gauge",
    "geometry_calibration",
    "misalign_spec",
    "simulation_artefacts",
}


def save_npz(
    path: str,
    projections: np.ndarray | None,
    *,
    metadata: NXTomoMetadata,
) -> None:
    """Write a typed TomoJAX payload to compressed NPZ.

    ``metadata`` mirrors the required ``save_nxtomo`` persistence contract; None
    projections save a reconstruction without its scan.
    """
    payload: LoadedDataset = LoadedNXTomo(
        projections=None if projections is None else np.asarray(projections),
        metadata=metadata,
    ).to_dataset_dict()
    if "angles" in payload:
        payload[_ANGLES_KEY] = payload.pop("angles")
    if "volume" in payload:
        payload[_AXES_KEY] = "zyx"
    parts = sizes_of(projections, metadata.volume)
    with replaced_when_complete(path, parts) as temporary, temporary.open("wb") as stream:
        arrays = {k: json.dumps(v) if k in _JSON_FIELDS else v for k, v in payload.items()}
        np.savez_compressed(stream, allow_pickle=False, **arrays)


def array_header(stream):
    version = np.lib.format.read_magic(stream)
    reader = (
        np.lib.format.read_array_header_1_0
        if version == (1, 0)
        else np.lib.format.read_array_header_2_0
    )
    return reader(stream)


@dataclass(frozen=True)
class NpzFrames(FrameStack):
    """A zipped NPY stream. Random selections must inflate the prefix up to the last pixel."""

    path: Path
    strides: tuple[int, ...]
    offset: int
    compressed: bool

    def read_bytes(self, ranges: tuple[range, ...], *, streaming: bool) -> int:
        if streaming or not self.compressed:
            return prod(map(len, ranges)) * self.dtype.itemsize
        view, row, col = (r[-1] for r in ranges)
        last = (int(self.frames[view]), self.first_row + row, self.first_col + col)
        return sum(i * s for i, s in zip(last, self.strides, strict=True)) + self.dtype.itemsize

    def _read(self, indices, rows, cols):
        return self.read_indices(
            indices, np.arange(rows.start, rows.stop), np.arange(cols.start, cols.stop)
        )

    def sampled(self, ranges):
        views, rows, cols = ranges
        return self.read_indices(
            self.frames[views.start : views.stop : views.step],
            self.first_row + np.asarray(rows),
            self.first_col + np.asarray(cols),
        )

    def read_indices(self, indices, rows, cols):
        if np.any(np.diff(indices) < 0):
            ordered, inverse = np.unique(indices, return_inverse=True)
            return self.read_indices(ordered, rows, cols)[inverse]
        strides = self.strides
        offsets = (
            indices[:, None, None] * strides[0]
            + rows[None, :, None] * strides[1]
            + cols[None, None, :] * strides[2]
        )
        order = "C" if strides[0] >= strides[-1] else "F"
        addresses = offsets.ravel(order=order)
        result = np.empty(len(addresses), self.dtype)
        with ZipFile(read_path(self.path)) as archive, archive.open("projections.npy") as stream:
            # Stored members seek directly. Deflated members must decode gaps, so
            # coalesce nearby samples into bounded reads while moving forward once.
            boundaries = (
                np.flatnonzero(np.diff(addresses) != self.dtype.itemsize) + 1
                if not self.compressed
                else ()
            )
            start = 0
            for end in chain(boundaries, (len(addresses),)):
                while start < end:
                    first = int(addresses[start])
                    stop = min(end, int(np.searchsorted(addresses, first + (1 << 20))))
                    last = int(addresses[stop - 1]) + self.dtype.itemsize
                    stream.seek(self.offset + first)
                    block = np.frombuffer(stream.read(last - first), self.dtype)
                    result[start:stop] = block[
                        (addresses[start:stop] - first) // self.dtype.itemsize
                    ]
                    start = stop
        return result.reshape(offsets.shape, order=order)


def npz_projection_layout(path: Path):
    with ZipFile(read_path(path)) as archive, archive.open("projections.npy") as stream:
        shape, fortran, dtype = array_header(stream)
        if dtype.hasobject:
            raise ValueError(f"{path}: object arrays are unsupported")
        placeholder = np.broadcast_to(np.zeros((), dtype), shape)
        payload = LoadedNXTomo.from_dataset(
            _load_npz_dataset(archive, mapped={"projections": placeholder})
        )
        strides = np.cumprod(
            (dtype.itemsize, *shape[:-1]) if fortran else (dtype.itemsize, *shape[:0:-1])
        )
        if len(shape) == 2:
            shape = (shape[0], 1, shape[1])
            strides = np.insert(strides, 1, 0)
        stack = NpzFrames(
            path,
            tuple(map(int, strides if fortran else strides[::-1])),
            stream.tell(),
            archive.getinfo("projections.npy").compress_type != ZIP_STORED,
            frames=np.arange(shape[0]),
            rows=shape[1],
            cols=shape[2],
            dtype=dtype,
        )
        return payload, stack


def _load_npz_dataset(
    archive: ZipFile, *, mapped: dict[str, np.ndarray] | None = None
) -> LoadedDataset:
    out: LoadedDataset = dict(mapped or {})
    for member in archive.infolist():
        k = member.filename.removesuffix(".npy")
        with archive.open(member) as stream:
            if k in ("projections", "volume"):
                shape, _, dtype = array_header(stream)
                if dtype.hasobject:
                    raise ValueError(f"{archive.filename}: object arrays are unsupported")
                if mapped is not None:
                    continue
                stream.seek(0)
                if k == "projections" and progress.watching():
                    views = shape[0] if shape else 0
                    with progress.task("load", Path(str(archive.filename)).name, views, "views"):
                        out[k] = np.lib.format.read_array(stream, allow_pickle=False)
                        if active := progress.for_unit("views"):
                            active.advance(views, {})
                        progress.check_cancelled()
                    continue
            val = np.lib.format.read_array(stream, allow_pickle=False)
            out["angles" if k == _ANGLES_KEY else k] = (
                json.loads(val.item()) if k in _JSON_FIELDS else val
            )
    if "volume" in out and str(out.pop(_AXES_KEY, "")) != "zyx":
        raise ValueError(
            f"{archive.filename}: volume has no zyx axes tag; declare it or re-export the file"
        )
    return out


def load_npz(path: str) -> LoadedNXTomo:
    """Load a compressed NPZ payload using the same typed shape as NXtomo."""
    with ZipFile(read_path(path)) as archive:
        return LoadedNXTomo.from_dataset(_load_npz_dataset(archive))


@contextmanager
def mapped_npz(path: str, *, member: str | None = None) -> Iterator[LoadedNXTomo]:
    """Map the primary array of a TomoJAX NPZ file for bounded random reads.

    Parameters
    ----------
    path : str
        Saved TomoJAX NPZ dataset.
    member : str or None, optional
        Select volume or projections. By default prefer the stored volume.

    Yields
    ------
    LoadedNXTomo
        The stored volume (z, y, x), or projections (view, v, u) for a scan,
        with metadata. The temporary pixel file lasts for the context's lifetime.

    Raises
    ------
    ValueError
        If the array or its saved volume axis declaration is invalid.
    """
    with ZipFile(read_path(path)) as archive, tempfile.TemporaryDirectory() as directory:
        key = member or ("volume" if "volume.npy" in archive.namelist() else "projections")
        if key + ".npy" not in archive.namelist():
            raise ValueError(f"{path}: no {key} array")
        filename = Path(directory) / "pixels.npy"
        with archive.open(key + ".npy") as src, filename.open("wb") as dest:
            shutil.copyfileobj(src, dest, length=1024 * 1024)
        pixels = np.load(filename, mmap_mode="r", allow_pickle=False)
        fields = _load_npz_dataset(archive, mapped={key: pixels})
        fields.pop("volume", None)
        dataset = LoadedNXTomo.from_dataset(fields)
        dataset.metadata.volume = pixels if key == "volume" else None
        yield dataset
