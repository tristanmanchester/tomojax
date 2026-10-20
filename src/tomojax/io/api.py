"""Expert IO helpers: acquisition readers, NXtomo containers, inspection and JSON.

The dataset entry points users call are exported by :mod:`tomojax.io`; raw
frames are read with :func:`tomojax.load_frames`.
"""

from tomojax._json import JsonValue, normalize_json, read_json_object
from tomojax.io._acquisition import read_acquisition
from tomojax.io._angles import load_angles
from tomojax.io._calibration import read_calibration_frames
from tomojax.io._datasets import (
    LoadedNXTomo,
    LocatedFrames,
    NXTomoMetadata,
    convert_dataset,
    load_nxtomo,
    locate_frames,
    save_nxtomo,
    validate_nxtomo,
)
from tomojax.io._hdf5_array import read_hdf5_array
from tomojax.io._inspection import inspect_dataset, projection_stats
from tomojax.io._inspection_format import format_inspection_report
from tomojax.io._quicklook import save_projection_quicklook
from tomojax.io._tiff import read_tiff_frames

__all__ = [
    "JsonValue",
    "LoadedNXTomo",
    "LocatedFrames",
    "NXTomoMetadata",
    "convert_dataset",
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
    "save_nxtomo",
    "save_projection_quicklook",
    "validate_nxtomo",
]
