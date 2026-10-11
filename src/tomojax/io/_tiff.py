"""Shared TIFF stack discovery helpers."""
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from pathlib import Path
import re
from typing import TYPE_CHECKING

import numpy as np
from tifffile import TiffFile

if TYPE_CHECKING:
    from collections.abc import Iterator

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


def read_tiff_frames(path: Path | str) -> np.ndarray:
    """Grayscale TIFF pages as ``(frames, rows, columns)`` in the files' dtype.

    Files in a folder are naturally sorted, then every top-level page is read
    in file order. Every page must be a 2-D image of the first page's size and
    dtype; colour images are rejected rather than read as extra view axes.
    """
    files = tiff_files(Path(path))
    if not files:
        raise ValueError(f"no TIFF files found under {path}")

    def pages() -> Iterator[TiffPage | TiffFrame]:
        for file in files:
            with TiffFile(file) as tiff:
                yield from tiff.pages

    layouts = [(page.shape, page.dtype) for page in pages()]  # headers only
    if not layouts:
        raise ValueError(f"no TIFF frames found under {path}")
    shape, dtype = layouts[0]
    if len(shape) != 2 or len(set(layouts)) != 1:
        found = sorted({f"{shape} {dtype}" for shape, dtype in layouts})
        raise ValueError(f"{path}: TIFF frames must be 2-D images of one size and dtype: {found}")
    # Decode straight into one stack, not a list of pages and then a copy.
    frames = np.empty((len(layouts), *shape), dtype=dtype)
    for view, page in enumerate(pages()):
        page.asarray(out=frames[view])
    return frames
