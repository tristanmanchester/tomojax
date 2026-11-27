"""Owner facade for private persistence, preprocessing, phantom, and simulation helpers."""

from __future__ import annotations

from tomojax._data._format2 import nxdata, read_format_2, registered_members, saved_volume_grid
from tomojax._data._frame_stack import FrameStack
from tomojax._data._io_acquisition import (
    ROTATION_ANGLE_PATHS,
    angle_scale_degrees,
    calibration_issues,
)
from tomojax._data._io_frames import Hdf5Frames, LocatedFrames, PinnedFrames, locate_frames
from tomojax._data._io_helpers import child_dataset
from tomojax._data._io_npz import load_npz, npz_layout, save_npz
from tomojax._data._io_nxtomo import (
    load_nxtomo,
    native_payload,
    read_nxtomo,
    read_projections,
    save_nxtomo,
    saved_volume_axes,
    saved_without_frames,
    slab_volume,
    validate_nxtomo,
    volume_entry,
)
from tomojax._data._io_types import LoadedNXTomo, NXTomoMetadata
from tomojax._data._record import array_record, metadata_record, provenance
from tomojax._data.geometry_meta import LoadedGeometryMeta, build_geometry_from_meta, views_of

__all__ = [
    "ROTATION_ANGLE_PATHS",
    "FrameStack",
    "Hdf5Frames",
    "LoadedGeometryMeta",
    "LoadedNXTomo",
    "LocatedFrames",
    "NXTomoMetadata",
    "PinnedFrames",
    "angle_scale_degrees",
    "array_record",
    "build_geometry_from_meta",
    "calibration_issues",
    "child_dataset",
    "load_npz",
    "load_nxtomo",
    "locate_frames",
    "metadata_record",
    "native_payload",
    "npz_layout",
    "nxdata",
    "provenance",
    "read_format_2",
    "read_nxtomo",
    "read_projections",
    "registered_members",
    "save_npz",
    "save_nxtomo",
    "saved_volume_axes",
    "saved_volume_grid",
    "saved_without_frames",
    "slab_volume",
    "validate_nxtomo",
    "views_of",
    "volume_entry",
]
