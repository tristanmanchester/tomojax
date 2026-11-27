"""Saved scan metadata and lazy sample storage in logical detector order."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import h5py
import numpy as np

from tomojax._data import (
    Hdf5Frames,
    LoadedNXTomo,
    LocatedFrames,
    PinnedFrames,
    locate_frames,
    native_payload,
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
) -> tuple[ProjectionDataset, Hdf5Frames | PinnedFrames | MatlabFrames | np.ndarray | None]:
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

    def frames_at(self, which: np.ndarray | slice) -> MatlabFrames:
        """Select phase images in acquisition order."""
        return MatlabFrames(self.stored.frames_at(which))

    def window(self, rows: slice, cols: slice) -> MatlabFrames:
        """Select detector rows and columns in logical image order."""
        return MatlabFrames(self.stored.window(cols, rows))

    def __getitem__(self, views: int | slice) -> np.ndarray:
        return self.stored[views].swapaxes(-1, -2).astype(self.dtype, copy=False)


type ProjectionPixels = Hdf5Frames | PinnedFrames | MatlabFrames | np.ndarray


@dataclass(frozen=True)
class ProjectionStorage:
    """One selection of samples and per-frame metadata, before resolving geometry."""

    record: ProjectionDataset
    stack: ProjectionPixels
    key: np.ndarray
    domain: str | None
    located: LocatedFrames | None

    @classmethod
    def from_payload(cls, path: Path, saved: LoadedNXTomo) -> ProjectionStorage:
        """Keep selected pixels lazy while rebuilding their geometry record."""
        stack = saved.projections
        if stack is None:
            raise ValueError(
                f"{path} holds a reconstruction saved without its scan: "
                "tj.load_reconstruction reads it"
            )
        shape = np.broadcast_to(np.zeros((), stack.dtype), stack.shape)
        record = ProjectionDataset.from_nxtomo(replace(saved, projections=shape), source_path=path)
        return cls.from_native(record, stack)

    @classmethod
    def from_native(cls, record: ProjectionDataset, stack: ProjectionPixels) -> ProjectionStorage:
        """Pair a format-2 record with the pixels selected in the same open file."""
        key = record.image_key
        if key is None:
            key = np.zeros(stack.shape[0], np.int32)
        return cls(record, stack, key, record.projection_domain, None)

    def frames_at(self, mask: np.ndarray) -> ProjectionPixels:
        """Restrict the selected storage without reading lazy pixels."""
        if isinstance(self.stack, np.ndarray):
            return self.stack[mask]
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
        record, stack, key, "counts" if counts or np.isin(key, (1, 2)).any() else None, located
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
    saved = native_payload(
        path, include_volume=False, layout_only=layout_only, lazy_projections=layout_only, **paths
    )
    if saved is not None:
        return ProjectionStorage.from_payload(path, saved)
    acquisition = read_acquisition(path, **paths)
    if acquisition is not None:
        return acquisition
    if path.is_dir() or path.suffix.lower() in {".tif", ".tiff"}:
        check_dataset_paths("tiff", paths)
        return path
    return projection_storage(path, layout_only=layout_only, **paths)
