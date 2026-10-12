"""Shared TIFF stack discovery helpers."""
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import TYPE_CHECKING, Literal, NamedTuple, overload

import numpy as np
from tifffile import TIFF, TiffFile

if TYPE_CHECKING:
    from tifffile import TiffFrame, TiffPage

TIFF_SUFFIXES = frozenset({".tif", ".tiff"})
_NATURAL_SORT_PARTS = re.compile(r"(\d+)")

type _NaturalSortPart = tuple[int, str] | tuple[int, int, str]


def _natural_sort_key(path: Path) -> tuple[_NaturalSortPart, ...]:
    parts: list[_NaturalSortPart] = []
    for text in _NATURAL_SORT_PARTS.split(path.name.lower()):
        if text.isdigit():
            parts.append((1, int(text), text))
        else:
            parts.append((0, text))
    return tuple(parts)


def tiff_files(path: Path) -> list[Path]:
    """Return a deterministic list of TIFF files for a file or directory path."""
    if path.is_file():
        if path.suffix.lower() not in TIFF_SUFFIXES:
            raise ValueError(f"{path} is not a TIFF file")
        return [path]
    if path.is_dir():
        return sorted(
            (
                file
                for file in path.iterdir()
                if file.is_file() and file.suffix.lower() in TIFF_SUFFIXES
            ),
            key=_natural_sort_key,
        )
    raise FileNotFoundError(path)


def _page_window(page: TiffPage | TiffFrame, rows: slice, cols: slice) -> np.ndarray:
    """Read intersecting grayscale segments through tifffile's own decoder."""
    key = page.keyframe
    if key is not None and key.is_memmappable:
        return page.asarray(out="memmap")[rows, cols]
    # Preserve native handling of special layouts (including NDPI/JPEG).
    # Canonical scalar segments have one separate/contiguous sample and depth.
    if (
        key is None
        or key.shaped != (1, 1, *page.shape, 1)
        or key.compression in {6, 7, 34892, 33007}
        or (key.is_tiled and 324 not in key.tags)
    ):
        return page.asarray()[rows, cols]
    r0, r1, _ = rows.indices(page.shape[0])
    c0, c1, _ = cols.indices(page.shape[1])
    if key.is_tiled:
        height, width = key.tilelength, key.tilewidth
        across = (page.shape[1] + width - 1) // width
        indices = [
            y * across + x
            for y in range(r0 // height, (r1 - 1) // height + 1)
            for x in range(c0 // width, (c1 - 1) // width + 1)
        ]
    else:
        indices = list(range(r0 // key.rowsperstrip, (r1 - 1) // key.rowsperstrip + 1))
    out = np.empty((r1 - r0, c1 - c0), key.dtype)
    decode = key.decode
    handle = page.parent.filehandle  # this read owns its TiffFile, never shared across threads
    for index in indices:
        size, offset = page.databytecounts[index], page.dataoffsets[index]
        raw = None
        if size and offset:
            _ = handle.seek(offset)
            raw = handle.read(size)
        pixels, (_, _, y, x, _), shape = decode(raw, index)
        ya, yb = max(r0, y), min(r1, y + shape[1])
        xa, xb = max(c0, x), min(c1, x + shape[2])
        target = out[ya - r0 : yb - r0, xa - c0 : xb - c0]
        if pixels is None:
            target[:] = key.nodata
        else:
            target[:] = pixels[0, ya - y : yb - y, xa - x : xb - x, 0]
    return out


def _compatible_pages(pages: list[TiffPage | TiffFrame], shape: tuple[int, int]) -> bool:
    """Whether native stacking can reuse one scalar page's decoder and fill value."""
    keys = [page.keyframe for page in pages]
    first = keys[0]
    if first is None:
        return False
    return (
        first.shaped == (1, 1, *shape, 1)
        and first.compression not in {6, 7, 34892, 33007}
        and not first.parent.is_ndpi
        # tifffile's decoder-compatibility hash excludes the no-data fill.
        and all(
            key is not None and key.hash == first.hash and key.nodata == first.nodata
            for key in keys[1:]
        )  # fmt: skip
    )


def _read_page_group(
    tiff: TiffFile,
    indices: list[int],
    out: np.ndarray,
    rows: slice,
    cols: slice,
    *,
    maxworkers: int | None = None,
) -> None:
    """Fill one file's output, retaining streaming headers for cropped reads."""
    first = tiff.pages[indices[0]]
    full = rows.start == cols.start == 0 and first.shape == out.shape[1:]
    if full and len(indices) > 1:
        # Bound retained page metadata even for an eager read of tiny images.
        for start in range(0, len(indices), 256):
            selected = indices[start : start + 256]
            pages = [tiff.pages[index] for index in selected]
            if _compatible_pages(pages, out.shape[1:]):
                # Native stacking bounds its workers and writes into this
                # buffer; it can parallelize decoding without extra stacks.
                _ = tiff.asarray(
                    key=selected, out=out[start : start + len(selected)], maxworkers=maxworkers
                )
            else:
                for index, page in enumerate(pages, start):
                    _ = page.asarray(out=out[index], maxworkers=maxworkers)
        return
    for index, selected in enumerate(indices):
        page = first if index == 0 else tiff.pages[selected]
        if full:
            _ = page.asarray(out=out[index], maxworkers=maxworkers)
        else:
            out[index] = _page_window(page, rows, cols)


def _read_file_group(
    path: Path,
    indices: list[int],
    out: np.ndarray,
    rows: slice,
    cols: slice,
    *,
    maxworkers: int | None = None,
) -> None:
    with TiffFile(path) as tiff:
        _read_page_group(tiff, indices, out, rows, cols, maxworkers=maxworkers)


def _file_read_groups(selected: list[tuple[Path, int]]) -> list[tuple[Path, list[int], slice]]:
    groups = []
    start = 0
    while start < len(selected):
        path = selected[start][0]
        stop = start + 1
        while stop < len(selected) and selected[stop][0] == path:
            stop += 1
        groups.append((path, [index for _, index in selected[start:stop]], slice(start, stop)))
        start = stop
    return groups


def _read_compressed_files(
    groups: list[tuple[Path, list[int], slice]], out: np.ndarray, rows: slice, cols: slice
) -> None:
    """Transfer native decoder concurrency to files, with bounded pending metadata."""
    workers = min(len(groups), TIFF.MAXWORKERS)
    pending: set[Future[None]] = set()
    with ThreadPoolExecutor(max_workers=workers) as reader:
        for path, indices, destination in groups:
            pending.add(
                reader.submit(
                    _read_file_group, path, indices, out[destination], rows, cols, maxworkers=1
                )
            )
            if len(pending) == workers:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
        for future in pending:
            future.result()


def _read_file_sequence(
    selected: list[tuple[Path, int]],
    compressed: tuple[bool, ...],
    out: np.ndarray,
    rows: slice,
    cols: slice,
    *,
    parallel: bool,
) -> None:
    groups = _file_read_groups(selected)
    decodable = [group for group in groups if any(compressed[group[2]])]
    if parallel and len(decodable) > 1 and TIFF.MAXWORKERS > 1:
        _read_compressed_files(decodable, out, rows, cols)
        groups = [group for group in groups if not any(compressed[group[2]])]
    for path, indices, destination in groups:
        _read_file_group(path, indices, out[destination], rows, cols)


class _MappedPage(NamedTuple):
    """Native readonly page-map metadata; never a decoded image or live file."""

    offset: int
    shape: tuple[int, ...]
    dtype: np.dtype


@dataclass(frozen=True)
class _TiffFrames:
    """Header-validated TIFF pages, with no open file or decoded pixels retained."""

    pages: tuple[tuple[Path, int], ...]
    frames: np.ndarray
    rows: int
    cols: int
    dtype: np.dtype
    compressed: tuple[bool, ...]
    first_row: int = 0
    first_col: int = 0
    is_window: bool = False
    _mapped_pages: tuple[_MappedPage | None, ...] = ()

    @classmethod
    def open(cls, path: Path | str) -> _TiffFrames:
        files = tiff_files(Path(path).absolute())  # capture the source before later cwd changes
        if not files:
            raise ValueError(f"no TIFF files found under {path}")
        pages: list[tuple[Path, int]] = []
        layouts: list[tuple[tuple[int, ...], np.dtype | None]] = []
        compressed: list[bool] = []
        mapped_pages: list[_MappedPage | None] = []
        for file in files:
            with TiffFile(file) as tiff:
                for index, page in enumerate(tiff.pages):
                    pages.append((file, index))
                    layouts.append((page.shape, page.dtype))
                    compressed.append(page.compression != 1)
                    key = page.keyframe
                    mapped_pages.append(
                        _MappedPage(page.dataoffsets[0], page.shape,
                                    np.dtype(key.dtype).newbyteorder(tiff.byteorder))
                        if key is not None and key.is_memmappable else None
                    )  # fmt: skip
        if not layouts:
            raise ValueError(f"no TIFF frames found under {path}")
        shape, dtype = layouts[0]
        if len(shape) != 2 or len(set(layouts)) != 1:
            found = sorted({f"{shape} {dtype}" for shape, dtype in layouts})
            raise ValueError(
                f"{path}: TIFF frames must be 2-D images of one size and dtype: {found}"
            )
        return cls(
            tuple(pages),
            np.arange(len(pages)),
            shape[0],
            shape[1],
            np.dtype(dtype),
            tuple(compressed),
            _mapped_pages=tuple(mapped_pages),
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.frames), self.rows, self.cols)

    @property
    def ndim(self) -> int:
        return 3

    @property
    def _efficient_row_windows(self) -> bool:
        """Selected frames can be revisited without repeatedly decoding whole frames."""
        return bool(self._mapped_pages) and all(
            self._mapped_pages[int(i)] is not None for i in self.frames
        )

    def __len__(self) -> int:
        return len(self.frames)

    def _shares_file(self, path: Path) -> bool:
        """Whether caller-owned mapped output would overwrite an acquisition file."""
        return any(file.samefile(path) for file in {self.pages[int(i)][0] for i in self.frames})

    def frames_at(self, which: np.ndarray | slice) -> _TiffFrames:
        """Narrow the views without decoding any pixels."""
        return replace(self, frames=np.asarray(self.frames[which], np.int64))

    def window(self, rows: slice, cols: slice) -> _TiffFrames:
        """Narrow the detector window without decoding any pixels."""
        r0, r1, rs = rows.indices(self.rows)
        c0, c1, cs = cols.indices(self.cols)
        if rs != 1 or cs != 1 or r1 <= r0 or c1 <= c0:
            raise ValueError("a window of frames is a non-empty block of rows and columns")
        return replace(
            self, rows=r1 - r0, cols=c1 - c0,
            first_row=self.first_row + r0, first_col=self.first_col + c0,
            is_window=self.is_window or (r0, r1, c0, c1) != (0, self.rows, 0, self.cols),
        )  # fmt: skip

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        if copy is False:
            raise ValueError("TIFF frames cannot be converted to an array without copying")
        return np.asarray(self[:], dtype=dtype)

    def __getitem__(self, which: slice | int | np.ndarray) -> np.ndarray:
        if isinstance(which, int | np.integer):
            return self[np.asarray([which], np.int64)][0]
        indices = np.asarray(self.frames[which], np.int64)
        selected = [self.pages[int(index)] for index in indices]
        out = np.empty((len(selected), self.rows, self.cols), self.dtype)
        r = slice(self.first_row, self.first_row + self.rows)
        c = slice(self.first_col, self.first_col + self.cols)
        if self.is_window and self._efficient_row_windows:
            for destination, index in enumerate(indices):
                layout = self._mapped_pages[int(index)]
                assert layout is not None
                pixels = np.memmap(
                    self.pages[int(index)][0], mode="r", dtype=layout.dtype,
                    offset=layout.offset, shape=layout.shape,
                )  # fmt: skip
                out[destination] = pixels[r, c]
            return out
        compressed = tuple(self.compressed[int(index)] for index in indices)
        _read_file_sequence(selected, compressed, out, r, c, parallel=not self.is_window)
        return out


@overload
def read_tiff_frames(path: Path | str, *, lazy: Literal[False] = False) -> np.ndarray: ...


@overload
def read_tiff_frames(path: Path | str, *, lazy: Literal[True]) -> _TiffFrames: ...


@overload
def read_tiff_frames(path: Path | str, *, lazy: bool) -> np.ndarray | _TiffFrames: ...


def read_tiff_frames(path: Path | str, *, lazy: bool = False) -> np.ndarray | _TiffFrames:
    """Grayscale TIFF pages as ``(frames, rows, columns)`` in the files' dtype.

    Files in a folder are naturally sorted, then every top-level page is read
    in file order. Every page must be a 2-D image of the first page's size and
    dtype; colour images are rejected rather than read as extra view axes.
    ``lazy=True`` validates headers but reads only when indexed: a detector
    window maps uncompressed contiguous pages, or decodes intersecting scalar
    strips/tiles; special layouts use native page decoding. The returned stack
    keeps no open files and is safe to read from
    several threads; ``np.asarray`` explicitly materializes it.
    """
    stack = _TiffFrames.open(path)
    return stack if lazy else np.asarray(stack)
