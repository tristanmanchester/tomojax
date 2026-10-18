"""Shared TIFF stack discovery helpers."""
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
import re
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np
from tifffile import TiffFile

from tomojax._data import FrameStack

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

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


def _decoded_window(page: TiffPage | TiffFrame, rows: slice, cols: slice) -> np.ndarray:
    """Decode only strips/tiles intersecting a contiguous grayscale window."""
    key = cast("TiffPage", page.keyframe)
    height, width = key.imagelength, key.imagewidth
    r0, r1, _ = rows.indices(height)
    c0, c1, _ = cols.indices(width)
    if r0 == c0 == 0 and r1 == height and c1 == width:
        return page.asarray()  # tifffile batches/parallelizes full-page decoding.
    tile_rows = key.tilelength if key.is_tiled else key.rowsperstrip
    tile_cols = key.tilewidth if key.is_tiled else width
    across = (width + tile_cols - 1) // tile_cols
    indices = [
        r * across + c
        for r in range(r0 // tile_rows, (r1 - 1) // tile_rows + 1)
        for c in range(c0 // tile_cols, (c1 - 1) // tile_cols + 1)
    ]
    offsets = [page.dataoffsets[i] for i in indices]
    counts = [page.databytecounts[i] for i in indices]
    encoded = cast(
        "Iterator[tuple[bytes | None, int]]",
        page.parent.filehandle.read_segments(offsets, counts, indices, flat=True),
    )
    options = {}
    if key.compression in {6, 7, 34892, 33007}:
        options = {"jpegtables": page.jpegtables, "jpegheader": key.jpegheader}
    result = np.zeros((r1 - r0, c1 - c0), key.dtype)
    for payload, index in encoded:
        pixels, origin, _ = key.decode(payload, index, **options)
        if pixels is None:  # TIFF permits omitted, zero-filled segments.
            continue
        _, _, row, col, _ = origin
        block = pixels[0, :, :, 0]
        ra, rb = max(r0, row), min(r1, row + block.shape[0])
        ca, cb = max(c0, col), min(c1, col + block.shape[1])
        result[ra - r0 : rb - r0, ca - c0 : cb - c0] = block[
            ra - row : rb - row, ca - col : cb - col
        ]
    return result


@dataclass(frozen=True)
class TiffFrames(FrameStack):
    """The pages of a TIFF stack, checked from their headers and read only as indexed.

    Holds no open file. A window of uncompressed pages is read through a memory map,
    so only its rows are touched; other windows decode only intersecting strips/tiles.
    """

    pages: tuple[tuple[Path, int], ...]  # file and page index of each frame
    maps: tuple[tuple[int, np.dtype] | None, ...]  # data offset and dtype of mappable pages
    page_shape: tuple[int, int]
    segment_rows: tuple[int, ...]
    flip_rows: bool = False

    @classmethod
    def open(cls, path: Path | str) -> TiffFrames:
        return cls.from_files(tiff_files(Path(path)))

    @classmethod
    def from_files(cls, files: Sequence[Path], *, flip_rows: bool = False) -> TiffFrames:
        """Read only the selected headers, in metadata order, optionally with v up."""
        files = [file.absolute() for file in files]  # independent of later cwd changes
        if not files:
            raise ValueError("no TIFF files selected")
        pages: list[tuple[Path, int]] = []
        layouts = []
        maps = []
        segment_rows = []
        for file in files:
            with TiffFile(file) as tiff:
                for index, page in enumerate(tiff.pages):
                    pages.append((file, index))
                    layouts.append((page.shape, page.dtype))
                    key = cast("TiffPage", page.keyframe)
                    segment_rows.append(key.tilelength if key.is_tiled else key.rowsperstrip)
                    if key.is_memmappable:
                        dtype = np.dtype(key.dtype).newbyteorder(tiff.byteorder)
                        maps.append((page.dataoffsets[0], dtype))
                    else:
                        maps.append(None)
        if not layouts:
            raise ValueError("no TIFF frames found in the selected files")
        shape, dtype = layouts[0]
        if len(shape) != 2 or len(set(layouts)) != 1:
            found = sorted({f"{shape} {dtype}" for shape, dtype in layouts})
            raise ValueError(f"TIFF frames must be 2-D images of one size and dtype: {found}")
        return cls(
            tuple(pages),
            tuple(maps),
            shape,
            tuple(segment_rows),
            frames=np.arange(len(pages)),
            dtype=np.dtype(dtype),
            rows=shape[0],
            cols=shape[1],
            flip_rows=flip_rows,
        )

    @property
    def _efficient_row_windows(self) -> bool:
        """Whether row windows avoid decoding a whole detector panel."""
        return all(
            self.maps[i] is not None or self.segment_rows[i] < self.page_shape[0]
            for i in self.frames
        )

    def _read(self, indices: np.ndarray, rows: slice, cols: slice) -> np.ndarray:
        out = np.empty((len(indices), self.rows, self.cols), self.dtype)
        if self.flip_rows:
            start, stop, _ = rows.indices(self.page_shape[0])
            rows = slice(self.page_shape[0] - stop, self.page_shape[0] - start)
        window = (rows, cols)
        for file, group in groupby(enumerate(indices), key=lambda item: self.pages[item[1]][0]):
            with TiffFile(file) as tiff:  # one open per run of pages in a file
                for destination, index in group:
                    mapped = self.maps[index]
                    if mapped is None:
                        image = _decoded_window(tiff.pages[self.pages[index][1]], rows, cols)
                    else:
                        offset, dtype = mapped
                        pixels = np.memmap(
                            file, mode="r", dtype=dtype, offset=offset, shape=self.page_shape
                        )
                        image = pixels[window]
                    out[destination] = image[::-1] if self.flip_rows else image
        return out


@overload
def read_tiff_frames(path: Path | str, *, lazy: Literal[False] = False) -> np.ndarray: ...


@overload
def read_tiff_frames(path: Path | str, *, lazy: Literal[True]) -> TiffFrames: ...


def read_tiff_frames(path: Path | str, *, lazy: bool = False) -> np.ndarray | TiffFrames:
    """Grayscale TIFF pages as ``(frames, rows, columns)`` in the files' dtype.

    Files in a folder are naturally sorted, then every top-level page is read
    in file order. Every page must be a 2-D image of the first page's size and
    dtype; colour images are rejected rather than read as extra view axes.
    ``lazy=True`` checks the headers and returns a stack read only as indexed
    (``np.asarray`` reads it all); windows read only their mapped pixels or
    intersecting compressed strips/tiles.

    Parameters
    ----------
    path : str or pathlib.Path
        Grayscale multipage TIFF or folder of TIFF files.
    lazy : bool, optional
        Inspect headers now and read pixels only when indexed; defaults to False.

    Returns
    -------
    numpy.ndarray or lazy TIFF stack
        Frames in (frame, row, column) order and the files' dtype.

    Raises
    ------
    ValueError
        If pages are not consistent two-dimensional grayscale frames.
    FileNotFoundError
        If the input contains no TIFF frames.
    """
    stack = TiffFrames.open(path)
    return stack if lazy else np.asarray(stack)
