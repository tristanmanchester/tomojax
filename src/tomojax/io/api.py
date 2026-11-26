"""Expert IO helpers: acquisition readers, NXtomo containers, inspection and JSON.

The dataset entry points users call are exported by :mod:`tomojax.io`; raw
frames are read with :func:`tomojax.load_frames`.
"""

from tomojax._data import slab_volume
from tomojax.core.json_values import JsonValue, normalize_json, read_json_object
from tomojax.io._acquisition import read_acquisition
from tomojax.io._angles import load_angles
from tomojax.io._calibration import read_calibration_frames
from tomojax.io._datasets import (
    LoadedNXTomo,
    LocatedFrames,
    NXTomoMetadata,
    convert_dataset,
    dataset_output_format,
    load_nxtomo,
    locate_frames,
    replace_dataset,
    save_nxtomo,
    saved_volume_axes,
    saved_without_frames,
    validate_nxtomo,
)
from tomojax.io._hdf5_array import read_hdf5_array
from tomojax.io._inspection import inspect_dataset, projection_stats
from tomojax.io._inspection_format import format_inspection_report
from tomojax.io._inspection_types import Inspection
from tomojax.io._quicklook import save_projection_quicklook
from tomojax.io._tiff import read_tiff_frames
from tomojax.io._view_offsets import (
    dropped_offsets_message,
    with_view_offsets,
    without_view_offsets,
)

__all__ = [
    "Inspection",
    "JsonValue",
    "LoadedNXTomo",
    "LocatedFrames",
    "NXTomoMetadata",
    "convert_dataset",
    "dataset_output_format",
    "dropped_offsets_message",
    "format_inspection_report",
    "inspect_dataset",
    "load_angles",
    "load_nxtomo",
    "locate_frames",
    "normalize_json",
    "projection_stats",
    "read_acquisition",
    "read_calibration_frames",
    "read_hdf5_array",
    "read_json_object",
    "read_tiff_frames",
    "replace_dataset",
    "save_nxtomo",
    "save_projection_quicklook",
    "saved_volume_axes",
    "saved_without_frames",
    "slab_volume",
    "validate_nxtomo",
    "with_view_offsets",
    "without_view_offsets",
]
