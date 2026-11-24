from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from heapq import merge
from itertools import groupby
import json
from math import prod
from operator import itemgetter
from pathlib import Path
from tempfile import TemporaryFile
from typing import BinaryIO
from zipfile import ZIP_STORED, ZipFile

import numpy as np

from tomojax import progress
from tomojax.core.files import read_path, replaced_when_complete
from tomojax.core.host_volume import release

from ._frame_stack import FrameStack
from ._io_helpers import sizes_of
from ._io_types import LoadedDataset, LoadedNXTomo, NXTomoMetadata

_BLOCK_BYTES = 64 << 10

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


class MemberReader:
    """Read bounded spans, retaining inflated bytes only for nonmonotonic access."""

    def __init__(self, stream, compressed) -> None:
        self.stream = stream
        self.compressed = compressed
        self.cache: BinaryIO | None = None

    def read(self, offset, size):
        progress.check_cancelled()
        if self.cache is not None:
            self.cache.seek(offset)
            return self.cache.read(size)
        if self.compressed:
            # ZipExtFile.seek inflates the skipped interval in a single allocation.
            while (gap := offset - self.stream.tell()) > 0:
                if not self.stream.read(min(gap, _BLOCK_BYTES)):
                    raise EOFError("truncated NPZ member")
                progress.check_cancelled()
            assert offset == self.stream.tell()
        else:
            self.stream.seek(offset)
        return self.stream.read(size)


@dataclass(frozen=True)
class NpzFrames(FrameStack):
    """A zipped NPY stream. Random selections must inflate the prefix up to the last pixel."""

    path: Path
    strides: tuple[int, ...]
    offset: int
    compressed: bool
    member: str
    end: int

    reader: MemberReader | None = None

    def read_bytes(self, ranges: tuple[range, ...]) -> int:
        # Include axis indices, buffered offsets and the zip decoder's working space.
        return prod(map(len, ranges)) * self.dtype.itemsize + 16 * sum(map(len, ranges)) + (1 << 20)

    def _read(self, indices, rows, cols):
        return self.read_windows(
            [(indices, range(rows.start, rows.stop), range(cols.start, cols.stop))]
        )[0]

    def selection(self, ranges):
        views, rows, cols = ranges
        return (
            self.frames[views.start : views.stop : views.step],
            range(self.first_row + rows.start, self.first_row + rows.stop, rows.step),
            range(self.first_col + cols.start, self.first_col + cols.stop, cols.step),
        )

    def sampled(self, ranges):
        return self.sampled_windows([ranges])[0]

    def sampled_windows(self, windows):
        return self.read_windows([self.selection(ranges) for ranges in windows])

    @property
    def order(self):
        return "C" if self.strides[0] >= self.strides[-1] else "F"

    def copy_views(self, ids, out):
        """Inflate this partition once in file order, including Fortran-ordered members."""
        flat = (out if self.order == "C" else out.T).flat
        selected = self.selection((range(len(self)), range(self.rows), range(self.cols)))
        written = 0
        for _, target, values in self.chunks([(selected[0][ids], *selected[1:])]):
            flat[target] = values
            written += values.nbytes
            # Fortran input scatters one decoder block over many mapped pages.
            if written >= (8 << 20) or self.order == "F":
                release(out)
                written = 0
        release(out)

    def read_windows(self, selections):
        outputs = [
            np.empty(tuple(map(len, axes)), self.dtype, order=self.order) for axes in selections
        ]
        for i, target, chunk in self.chunks(selections):
            outputs[i].ravel(order=self.order)[target] = chunk
        return outputs

    @contextmanager
    def open_reader(self):
        if self.reader is not None:
            yield self.reader
        else:
            with ZipFile(read_path(self.path)) as archive, archive.open(self.member) as stream:
                yield MemberReader(stream, self.compressed)

    def sample_chunks(self, ranges):
        for _, _, values in self.chunks([self.selection(ranges)]):
            yield values

    def chunks(self, selections):
        # Merge plane selections by physical block. Overlapping planes share the
        # same decoder buffer, including where their sampled pixels intersect.
        with self.open_reader() as reader:
            for offset, size, spans in self.blocks(selections):
                block = np.frombuffer(reader.read(offset, size), self.dtype)
                for _, i, target, indices in spans:
                    yield i, target, block[indices]

    def blocks(self, selections):
        spans = (self.spans(i, *axes) for i, axes in enumerate(selections))
        ordered = merge(*spans, key=itemgetter(0))
        for offset, group in groupby(ordered, key=itemgetter(0)):
            yield offset, min(self.block_bytes, self.end - offset), group

    @contextmanager
    def cached_rows(self, ranges, fields):
        """Retain selected sample and calibration rows for nonmonotonic correction reads."""
        reader = self.reader
        if reader is None or not self.compressed or reader.cache is not None:
            yield
            return
        views, rows, _ = ranges
        # A detector gap may need neighbours anywhere on the selected row.
        columns = range(self.cols)
        selections = [self.selection((views, rows, columns))]
        selections.extend(
            field.selection((range(len(field)), rows, columns))
            for field in fields
            if isinstance(field, NpzFrames) and field.reader is reader
        )
        with TemporaryFile() as cache:
            for offset, size, _ in self.blocks(selections):
                cache.seek(offset)
                cache.write(reader.read(offset, size))
            reader.cache = cache
            try:
                yield
            finally:
                reader.cache = None

    @property
    def block_bytes(self):
        return max(1, _BLOCK_BYTES // self.dtype.itemsize) * self.dtype.itemsize

    def spans(self, window, indices, rows, cols):
        logical_shape = (len(indices), len(rows), len(cols))
        permutation = None
        if np.any(np.diff(indices) <= 0):
            # File order may differ from the requested view map, which can repeat views.
            permutation = np.argsort(indices, kind="stable")
            indices, starts, counts = np.unique(
                indices[permutation], return_index=True, return_counts=True
            )
        shape = (len(indices), len(rows), len(cols))
        total = prod(shape)
        for start in range(0, total, 16384):
            v, r, c = np.unravel_index(
                np.arange(start, min(start + 16384, total)), shape, order=self.order
            )
            addresses = (
                indices[v] * self.strides[0]
                + (rows.start + r * rows.step) * self.strides[1]
                + (cols.start + c * cols.step) * self.strides[2]
            )
            first = 0
            while first < len(addresses):
                offset = int(addresses[first]) // self.block_bytes * self.block_bytes
                stop = addresses.searchsorted(offset + self.block_bytes)
                source = (addresses[first:stop] - offset) // self.dtype.itemsize
                if permutation is None:
                    yield self.offset + offset, window, slice(start + first, start + stop), source
                else:
                    views = v[first:stop]
                    for copy in range(counts[views].max()):
                        keep = counts[views] > copy
                        logical = permutation[starts[views[keep]] + copy]
                        target = np.ravel_multi_index(
                            (logical, r[first:stop][keep], c[first:stop][keep]),
                            logical_shape,
                            order=self.order,
                        )
                        yield self.offset + offset, window, target, source[keep]
                first = stop


def npz_layout(path: Path, *, resources: ExitStack | None = None):
    """Describe arrays from NPY headers. An optional request scope keeps their decoders open."""
    stacks = {}
    placeholders = {}
    with ZipFile(read_path(path)) as archive, ExitStack() as local:
        owner = local if resources is None else resources
        for key in ("projections", "volume"):
            member = key + ".npy"
            if member not in archive.namelist():
                continue
            stream = owner.enter_context(archive.open(member))
            shape, fortran, dtype = array_header(stream)
            if len(shape) == 2 and key == "projections":
                shape = (shape[0], 1, shape[1])
            if len(shape) != 3 or dtype.hasobject:
                raise ValueError(f"{path}: invalid {key} shape or dtype")
            strides = np.cumprod(
                (dtype.itemsize, *shape[:-1]) if fortran else (dtype.itemsize, *shape[:0:-1])
            )
            compressed = archive.getinfo(member).compress_type != ZIP_STORED
            stacks[key] = NpzFrames(
                path,
                tuple(map(int, strides if fortran else strides[::-1])),
                stream.tell(),
                compressed,
                member,
                stream.tell() + prod(shape) * dtype.itemsize,
                reader=MemberReader(stream, compressed) if resources is not None else None,
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
    known = {
        *NXTomoMetadata.__dataclass_fields__,
        _ANGLES_KEY,
        _AXES_KEY,
        "source",
        "projections",
        "disk_volume_axes_order",
    }
    for member in archive.infolist():
        k = member.filename.removesuffix(".npy")
        if mapped is not None and k in mapped:
            continue
        with archive.open(member) as stream:
            if k not in known:
                if array_header(stream)[2].hasobject:
                    raise ValueError(f"{archive.filename}: object arrays are unsupported")
                continue
            if k == "projections" and progress.watching():
                shape, _, _ = array_header(stream)
                stream.seek(0)
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
