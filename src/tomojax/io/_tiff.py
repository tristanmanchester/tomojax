"""Shared TIFF stack discovery helpers."""
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
import re
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
from tifffile import TiffFile

from tomojax._data import FrameStack

if TYPE_CHECKING:
    from collections.abc import Sequence

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
class TiffFrames(FrameStack):
    """The pages of a TIFF stack, checked from their headers and read only as indexed.

    Holds no open file. A window of uncompressed pages is read through a memory map,
    so only its rows are touched; other pages are decoded whole and then cropped.
    """

    pages: tuple[tuple[Path, int], ...]  # file and page index of each frame
    maps: tuple[tuple[int, np.dtype] | None, ...]  # data offset and dtype of mappable pages
    page_shape: tuple[int, int]
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
            raise ValueError("no TIFF frames found in the selected files")
        shape, dtype = layouts[0]
        if len(shape) != 2 or len(set(layouts)) != 1:
            found = sorted({f"{shape} {dtype}" for shape, dtype in layouts})
            raise ValueError(f"TIFF frames must be 2-D images of one size and dtype: {found}")
        return cls(
            tuple(pages),
            tuple(maps),
            shape,
            frames=np.arange(len(pages)),
            dtype=np.dtype(dtype),
            rows=shape[0],
            cols=shape[1],
            flip_rows=flip_rows,
        )

    @property
    def _efficient_row_windows(self) -> bool:
        """Whether a window of rows reads only those rows (every page is mappable)."""
        return all(self.maps[i] is not None for i in self.frames)

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
                        pixels = tiff.pages[self.pages[index][1]].asarray()
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
    (``np.asarray`` reads it all); a window of uncompressed pages reads only
    that window.

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
