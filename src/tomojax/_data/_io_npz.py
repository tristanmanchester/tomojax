from __future__ import annotations

from dataclasses import dataclass
import json
from math import prod
from pathlib import Path
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
    member: str

    def read_bytes(self, ranges: tuple[range, ...]) -> int:
        # Include axis indices, buffered offsets and the zip decoder's working space.
        return prod(map(len, ranges)) * self.dtype.itemsize + 16 * sum(map(len, ranges)) + (1 << 20)

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
        if np.any(np.diff(indices) <= 0):
            ordered, inverse = np.unique(indices, return_inverse=True)
            return self.read_indices(ordered, rows, cols)[inverse]
        axes = (indices[:, None, None], rows[None, :, None], cols[None, None, :])
        order = "C" if self.strides[0] >= self.strides[-1] else "F"
        result = np.empty(tuple(map(len, (indices, rows, cols))), self.dtype, order=order)
        iterator = np.nditer(
            [*axes, result],
            flags=["external_loop", "buffered", "zerosize_ok"],
            op_flags=[["readonly"], ["readonly"], ["readonly"], ["writeonly"]],
            buffersize=4096,
            order=order,
        )
        with ZipFile(read_path(self.path)) as archive, archive.open(self.member) as stream:
            for v, r, c, target in iterator:
                addresses = (
                    self.offset + v * self.strides[0] + r * self.strides[1] + c * self.strides[2]
                )
                start = 0
                while start < len(target):
                    progress.check_cancelled()
                    first = int(addresses[start])
                    # ZipExtFile.seek inflates the entire skipped interval in one read.
                    if self.compressed:
                        while (gap := first - stream.tell()) > 0:
                            if not stream.read(min(gap, 65536)):
                                raise EOFError(f"truncated {self.member}")
                            progress.check_cancelled()
                    else:
                        stream.seek(first)
                    stop = addresses.searchsorted(first + 65536)
                    last = int(addresses[stop - 1]) + self.dtype.itemsize
                    block = np.frombuffer(stream.read(last - first), self.dtype)
                    target[start:stop] = block[
                        (addresses[start:stop] - first) // self.dtype.itemsize
                    ]
                    start = stop
        return result


def npz_layout(path: Path):
    """Describe saved arrays from NPY headers and decode only their metadata."""
    stacks = {}
    placeholders = {}
    with ZipFile(read_path(path)) as archive:
        for key in ("projections", "volume"):
            member = key + ".npy"
            if member not in archive.namelist():
                continue
            with archive.open(member) as stream:
                shape, fortran, dtype = array_header(stream)
                if len(shape) == 2 and key == "projections":
                    shape = (shape[0], 1, shape[1])
                if len(shape) != 3 or dtype.hasobject:
                    raise ValueError(f"{path}: invalid {key} shape or dtype")
                strides = np.cumprod(
                    (dtype.itemsize, *shape[:-1]) if fortran else (dtype.itemsize, *shape[:0:-1])
                )
                stacks[key] = NpzFrames(
                    path,
                    tuple(map(int, strides if fortran else strides[::-1])),
                    stream.tell(),
                    archive.getinfo(member).compress_type != ZIP_STORED,
                    member,
                    frames=np.arange(shape[0]),
                    rows=shape[1],
                    cols=shape[2],
                    dtype=dtype,
                )
                placeholders[key] = np.broadcast_to(np.zeros((), dtype), shape)
        fields = _load_npz_dataset(archive, mapped=placeholders)
        fields.pop("volume", None)
        payload = LoadedNXTomo.from_dataset(fields)
        payload.metadata.volume = placeholders.get("volume")
        return payload, stacks


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
