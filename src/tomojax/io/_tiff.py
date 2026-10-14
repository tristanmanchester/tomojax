"""Shared TIFF stack discovery helpers."""
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import groupby
from pathlib import Path
import re
from typing import Literal, overload

import numpy as np
from tifffile import TiffFile

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


@dataclass(frozen=True)
class _TiffFrames:
    """The pages of a TIFF stack, checked from their headers and read only as indexed.

    Holds no open file. A window of uncompressed pages is read through a memory map,
    so only its rows are touched; other pages are decoded whole and then cropped.
    """

    pages: tuple[tuple[Path, int], ...]  # file and page index of each frame
    maps: tuple[tuple[int, np.dtype] | None, ...]  # data offset and dtype of mappable pages
    page_shape: tuple[int, int]
    dtype: np.dtype
    rows: int  # the window of each page read: rows and columns from these
    cols: int
    first_row: int = 0
    first_col: int = 0

    @classmethod
    def open(cls, path: Path | str) -> _TiffFrames:
        files = tiff_files(Path(path).absolute())  # capture the source before later cwd changes
        if not files:
            raise ValueError(f"no TIFF files found under {path}")
        pages: list[tuple[Path, int]] = []
        layouts = []
        maps = []
        for file in files:
            with TiffFile(file) as tiff:
                for index, page in enumerate(tiff.pages):
                    pages.append((file, index))
                    layouts.append((page.shape, page.dtype))
                    key = page.keyframe
                    if key is not None and key.is_memmappable:
                        dtype = np.dtype(key.dtype).newbyteorder(tiff.byteorder)
                        maps.append((page.dataoffsets[0], dtype))
                    else:
                        maps.append(None)
        if not layouts:
            raise ValueError(f"no TIFF frames found under {path}")
        shape, dtype = layouts[0]
        if len(shape) != 2 or len(set(layouts)) != 1:
            found = sorted({f"{shape} {dtype}" for shape, dtype in layouts})
            raise ValueError(
                f"{path}: TIFF frames must be 2-D images of one size and dtype: {found}"
            )
        return cls(tuple(pages), tuple(maps), shape, np.dtype(dtype), shape[0], shape[1])

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.pages), self.rows, self.cols)

    @property
    def ndim(self) -> int:
        return 3

    @property
    def _efficient_row_windows(self) -> bool:
        """Whether a window of rows reads only those rows (every page is mappable)."""
        return all(m is not None for m in self.maps)

    def __len__(self) -> int:
        return len(self.pages)

    def frames_at(self, which: np.ndarray | slice) -> _TiffFrames:
        """The stack of these frames only (a boolean mask, indices or slice), unread."""
        chosen = np.arange(len(self))[which]
        return replace(
            self,
            pages=tuple(self.pages[i] for i in chosen),
            maps=tuple(self.maps[i] for i in chosen),
        )

    def window(self, rows: slice, cols: slice) -> _TiffFrames:
        """The stack of this block of every frame only, unread."""
        r0, r1, rs = rows.indices(self.rows)
        c0, c1, cs = cols.indices(self.cols)
        if rs != 1 or cs != 1 or r1 <= r0 or c1 <= c0:
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
            raise ValueError("TIFF frames cannot be converted to an array without copying")
        return np.asarray(self[:], dtype=dtype)

    def __getitem__(self, which: slice | int | np.ndarray) -> np.ndarray:
        if isinstance(which, int | np.integer):
            return self[np.asarray([which], np.int64)][0]
        chosen = np.arange(len(self))[which]
        out = np.empty((len(chosen), self.rows, self.cols), self.dtype)
        window = (
            slice(self.first_row, self.first_row + self.rows),
            slice(self.first_col, self.first_col + self.cols),
        )
        for file, group in groupby(enumerate(chosen), key=lambda item: self.pages[item[1]][0]):
            with TiffFile(file) as tiff:  # one open per run of pages in a file
                for destination, index in group:
                    mapped = self.maps[index]
                    if mapped is None:
                        pixels = tiff.pages[self.pages[index][1]].asarray()
                    else:
                        offset, dtype = mapped
                        pixels = np.memmap(
                            file, mode="r", dtype=dtype, offset=offset, shape=self.page_shape
                        )
                    out[destination] = pixels[window]
        return out


@overload
def read_tiff_frames(path: Path | str, *, lazy: Literal[False] = False) -> np.ndarray: ...


@overload
def read_tiff_frames(path: Path | str, *, lazy: Literal[True]) -> _TiffFrames: ...


def read_tiff_frames(path: Path | str, *, lazy: bool = False) -> np.ndarray | _TiffFrames:
    """Grayscale TIFF pages as ``(frames, rows, columns)`` in the files' dtype.

    Files in a folder are naturally sorted, then every top-level page is read
    in file order. Every page must be a 2-D image of the first page's size and
    dtype; colour images are rejected rather than read as extra view axes.
    ``lazy=True`` checks the headers and returns a stack read only as indexed
    (``np.asarray`` reads it all); a window of uncompressed pages reads only
    that window.
    """
    stack = _TiffFrames.open(path)
    return stack if lazy else np.asarray(stack)
