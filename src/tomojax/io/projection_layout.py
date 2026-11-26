"""Saved scan metadata and lazy sample storage in logical detector order."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from tomojax._data import Hdf5Frames, npz_layout, read_nxtomo
from tomojax.io._datasets import ProjectionDataset, psi_record

if TYPE_CHECKING:
    from pathlib import Path


def saved_npz_layout(
    path: Path, *, volume_only: bool = False
) -> tuple[ProjectionDataset, dict[str, np.ndarray]]:
    """Read the archive's record and its saved arrays."""
    payload, arrays = npz_layout(path, volume_only=volume_only)
    return ProjectionDataset.from_nxtomo(payload, source_path=path), arrays


def saved_projection_layout(
    path: Path,
) -> tuple[ProjectionDataset, Hdf5Frames | MatlabFrames | np.ndarray | None]:
    """Read a geometry record with shape-only projections and the matching pixel source."""
    if path.suffix.lower() == ".mat":
        record = psi_record(path, layout_only=True)
        stack = Hdf5Frames.open(str(path), "aligned_phase_sinogram")
        return record, MatlabFrames(stack)
    if path.suffix.lower() == ".npz":
        record, members = saved_npz_layout(path)
        return record, members.get("projections")
    payload = read_nxtomo(str(path), lazy=True, layout_only=True)
    stack = payload.projections
    if stack is not None:
        shape = np.broadcast_to(np.zeros((), stack.dtype), stack.shape)
        payload = replace(payload, projections=shape)
    return ProjectionDataset.from_nxtomo(payload, source_path=path), stack


class MatlabFrames:
    """MATLAB stores phase columns before rows. The scan loader exposes float32."""

    ndim = 3
    dtype = np.dtype(np.float32)

    def __init__(self, stored: Hdf5Frames) -> None:
        self.stored = stored
        self.shape = (len(stored), stored.cols, stored.rows)

    def __getitem__(self, views: int | slice) -> np.ndarray:
        return self.stored[views].swapaxes(-1, -2).astype(self.dtype, copy=False)
