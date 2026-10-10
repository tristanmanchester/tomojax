"""Shared TIFF stack discovery helpers."""
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from pathlib import Path
import re

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


def read_tiff_frames(path: Path | str) -> np.ndarray:
    """Grayscale TIFF pages as ``(frames, rows, columns)`` in the files' dtype.

    Files in a folder are naturally sorted, then every top-level page is read
    in file order, including pages in separate TIFF series. Colour images and
    unequal detector sizes are rejected rather than interpreted as view axes.
    """
    files = tiff_files(Path(path))
    if not files:
        raise ValueError(f"no TIFF files found under {path}")
    frames: list[np.ndarray] = []
    for file in files:
        with TiffFile(file) as tiff:
            frames.extend(np.asarray(page.asarray()) for page in tiff.pages)
    if not frames:
        raise ValueError(f"no TIFF frames found under {path}")
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError(
            f"{path}: TIFF frames must be 2-D images of one size, found {sorted(shapes)}"
        )
    return np.stack(frames)
