"""Content detection for ESRF SPEC EDF and raw Bliss acquisitions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import h5py

from ._esrf_edf import read_edf_scan

if TYPE_CHECKING:
    from ._acquisition import RawAcquisition


def read_esrf(path: Path, *, data_path: str | None = None) -> RawAcquisition | None:
    """Read recognized acquisitions, returning None only for unrelated content."""
    if path.is_file() and h5py.is_hdf5(path):
        from ._bliss import read_bliss

        return read_bliss(path, data_path=data_path)
    return read_edf_scan(path)
