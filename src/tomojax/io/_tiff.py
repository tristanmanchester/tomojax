"""Shared TIFF stack discovery helpers."""
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from pathlib import Path
import re
from typing import cast

import imageio.v3 as iio
import numpy as np

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
    """The frames of a TIFF file or folder, ``(frames, rows, columns)`` in the files' dtype."""
    files = tiff_files(Path(path))
    if not files:
        raise ValueError(f"no TIFF files found under {path}")
    frames = [np.asarray(cast("object", iio.imread(file))) for file in files]
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError(
            f"{path}: TIFF frames must be 2-D images of one size, found {sorted(shapes)}"
        )
    return np.stack(frames)
