"""Saved scan metadata and lazy sample storage in logical detector order."""

from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from tomojax._data import Hdf5Frames, npz_layout, read_nxtomo
from tomojax.core.files import read_hdf5
from tomojax.io._datasets import ProjectionDataset, psi_record
from tomojax.io._hdf5_array import read_hdf5_array


def saved_npz_layout(
    path: Path, *, resources: ExitStack | None = None
) -> tuple[ProjectionDataset, dict[str, Any]]:
    """Read the archive metadata with sliceable sources for its saved arrays."""
    payload, arrays = npz_layout(path, resources=resources)
    return ProjectionDataset.from_nxtomo(payload, source_path=path), arrays


def saved_projection_layout(path: Path) -> tuple[ProjectionDataset, Any]:
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
        self.storage_dtype = stored.dtype
        self.shape = (len(stored), stored.cols, stored.rows)

    def __getitem__(self, views: int | slice) -> np.ndarray:
        return self.stored[views].swapaxes(-1, -2).astype(self.dtype, copy=False)

    def sampled(self, ranges: tuple[range, range, range]) -> np.ndarray:
        """Read logical (view, row, column) indices from the MATLAB-ordered phase stack."""
        views, rows, cols = ranges
        selection = tuple(slice(r.start, r.stop, r.step) for r in (views, cols, rows))
        with read_hdf5(self.stored.path, rdcc_nbytes=0) as file:
            phase = file[self.stored.dataset]
            assert isinstance(phase, h5py.Dataset)
            return (
                read_hdf5_array(phase, selection).transpose(0, 2, 1).astype(self.dtype, copy=False)
            )
