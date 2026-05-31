"""Shared TIFF stack discovery helpers."""

from __future__ import annotations

from pathlib import Path

TIFF_SUFFIXES = frozenset({".tif", ".tiff"})


def tiff_files(path: Path) -> list[Path]:
    """Return a deterministic list of TIFF files for a file or directory path."""
    if path.is_file():
        if path.suffix.lower() not in TIFF_SUFFIXES:
            raise ValueError(f"{path} is not a TIFF file")
        return [path]
    if path.is_dir():
        return sorted(
            file
            for file in path.iterdir()
            if file.is_file() and file.suffix.lower() in TIFF_SUFFIXES
        )
    raise FileNotFoundError(path)
