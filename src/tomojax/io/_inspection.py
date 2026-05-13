"""Public dataset inspection facade."""

from __future__ import annotations

from pathlib import Path

from tomojax.data.inspection import (
    InspectionReport,
    format_inspection_report,
    inspect_nxtomo,
    save_projection_quicklook,
)

type PathLike = str | Path


def inspect_dataset(path: PathLike) -> InspectionReport:
    """Inspect a measured projection dataset before reconstruction."""
    return inspect_nxtomo(Path(path))


__all__ = [
    "InspectionReport",
    "format_inspection_report",
    "inspect_dataset",
    "save_projection_quicklook",
]
