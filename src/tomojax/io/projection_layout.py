"""Saved scan metadata and lazy sample storage in logical detector order."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import h5py
import numpy as np

from tomojax._data import (
    Hdf5Frames,
    LocatedFrames,
    locate_frames,
    npz_layout,
    read_nxtomo,
    saved_without_frames,
)
from tomojax.io._acquisition import RawAcquisition, read_acquisition
from tomojax.io._datasets import ProjectionDataset, psi_record
from tomojax.io.dataset_paths import check_dataset_paths

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
    *,
    located: LocatedFrames | None = None,
    layout_only: bool = True,
) -> tuple[ProjectionDataset, Hdf5Frames | MatlabFrames | np.ndarray | None]:
    """Read a geometry record with shape-only projections and the matching pixel source."""
    if path.suffix.lower() == ".mat":
        record = psi_record(path, layout_only=layout_only)
        if not layout_only:
            return record, record.projections
        stack = Hdf5Frames.open(str(path), "aligned_phase_sinogram")
        return record, MatlabFrames(stack)
    if path.suffix.lower() == ".npz":
        record, members = saved_npz_layout(path)
        return record, members.get("projections")
    payload = read_nxtomo(str(path), lazy=True, layout_only=layout_only, located=located)
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


type ProjectionPixels = Hdf5Frames | MatlabFrames | np.ndarray


@dataclass(frozen=True)
class ProjectionStorage:
    """One selection of samples and per-frame metadata, before resolving geometry."""

    record: ProjectionDataset
    stack: ProjectionPixels
    key: np.ndarray
    counts: bool
    located: LocatedFrames | None

    def frames_at(self, mask: np.ndarray) -> ProjectionPixels:
        """Restrict the selected storage without reading lazy pixels."""
        if isinstance(self.stack, np.ndarray):
            return self.stack[mask]
        if isinstance(self.stack, MatlabFrames):
            return MatlabFrames(self.stack.stored.frames_at(mask))
        return self.stack.frames_at(mask)


def projection_storage(path: Path, *, layout_only: bool, **paths: str | None) -> ProjectionStorage:
    """Resolve HDF5 paths once. NPZ members and MATLAB phase arrays have fixed meanings."""
    located = None
    suffix = path.suffix.lower()
    counts = False
    if suffix in {".nx", ".nxs", ".h5", ".hdf", ".hdf5"}:
        try:
            located = locate_frames(str(path), **paths)
        except KeyError:
            if any(p is not None for p in paths.values()) or not saved_without_frames(path):
                raise
            raise ValueError(
                f"{path} holds a reconstruction saved without its scan: "
                "tj.load_reconstruction reads it"
            ) from None
        record, stack, counts = hdf_projection_storage(path, located, layout_only=layout_only)
    elif suffix in {".npz", ".mat"}:
        check_dataset_paths("saved.npz" if suffix == ".npz" else "phase", paths)
        record, stack = saved_projection_layout(path, layout_only=layout_only)
    else:
        raise ValueError(f"unsupported dataset format for {path}")
    if stack is None:
        raise ValueError(
            f"{path} holds a reconstruction saved without its scan: tj.load_reconstruction reads it"
        )
    if isinstance(stack, np.ndarray) and stack.ndim == 2:
        stack = stack[:, None, :]
        record.projections = stack
    if layout_only:
        record.volume = None
    key = record.image_key if located is None else located.image_key
    key = np.zeros(stack.shape[0], np.int32) if key is None else key
    return ProjectionStorage(
        record, stack, key, counts or bool(np.isin(key, (1, 2)).any()), located
    )


def hdf_projection_storage(
    path: Path,
    located: LocatedFrames,
    *,
    layout_only: bool,
) -> tuple[ProjectionDataset, ProjectionPixels | None, bool]:
    """Read geometry from a saved layout or from the selected acquisition's metadata."""
    with h5py.File(path, "r") as file:
        projections = any(p in file for p in ("/entry/data/projections", "/entry/projections"))
        saved_layout = projections or "/entry/instrument/detector/data" in file
    if saved_layout:
        record, stack = saved_projection_layout(path, located=located, layout_only=layout_only)
        return record, stack, not projections or located.raw
    meta = {"detector_center_status": "not_recorded"}
    record = ProjectionDataset(
        np.broadcast_to(np.zeros((), located.stack.dtype), located.stack.shape),
        np.empty(0, np.float32) if located.angles is None else located.angles,
        geometry_metadata=meta,
        source_path=str(path),
    )
    if located.acquisition is not None:
        record.geometry_metadata["acquisition"] = {
            **located.acquisition,
            "geometry_status": "not_recorded",
            "geometry_note": "Source has no calibrated rotation axis or pixel scale.",
        }
    return record, located.stack, True


type SelectedStorage = RawAcquisition | ProjectionStorage | Path


def select_storage(path: Path, *, layout_only: bool, **paths: str | None) -> SelectedStorage:
    """Select the acquisition, TIFF stack or stored projection dataset named by the caller."""
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    for name, value in paths.items():
        if value == "":
            raise ValueError(f"{name} needs a nonempty dataset path")
    acquisition = read_acquisition(path, **paths)
    if acquisition is not None:
        return acquisition
    if path.is_dir() or path.suffix.lower() in {".tif", ".tiff"}:
        check_dataset_paths("tiff", paths)
        return path
    return projection_storage(path, layout_only=layout_only, **paths)
