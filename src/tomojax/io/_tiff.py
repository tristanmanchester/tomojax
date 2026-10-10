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
    in file order, including pages in separate TIFF series. Colour images and
    unequal detector sizes are rejected rather than interpreted as view axes.
    """
    files = tiff_files(Path(path))
    if not files:
        raise ValueError(f"no TIFF files found under {path}")
    # Keep the first file open so its page headers can be reused. Only one
    # additional file is open at a time, even for a large acquisition folder.
    with TiffFile(files[0]) as first:
        first.pages.cache = True  # headers only, not decoded pixel arrays

        def pages() -> Iterator[TiffPage | TiffFrame]:
            yield from first.pages
            for file in files[1:]:
                with TiffFile(file) as tiff:
                    yield from tiff.pages

        metadata = [(page.shape, page.dtype) for page in pages()]
        if not metadata:
            raise ValueError(f"no TIFF frames found under {path}")
        shapes = {shape for shape, _ in metadata}
        if len(shapes) != 1 or len(next(iter(shapes))) != 2:
            raise ValueError(
                f"{path}: TIFF frames must be 2-D images of one size, found {sorted(shapes)}"
            )
        dtype = np.result_type(*(dtype for _, dtype in metadata))
        # Retain one decoded page at a time, instead of a full list plus stack.
        frames = np.empty((len(metadata), *next(iter(shapes))), dtype=dtype)
        view = 0
        for page in pages():
            if view >= len(metadata) or (page.shape, page.dtype) != metadata[view]:
                raise ValueError(f"{path}: TIFF frame metadata changed while reading")
            if page.dtype == dtype:
                _ = page.asarray(out=frames[view])
            else:  # mixed TIFF dtypes still promote like np.stack
                frames[view] = page.asarray()
            view += 1
        if view != len(metadata):
            raise ValueError(f"{path}: TIFF frame metadata changed while reading")
        return frames
