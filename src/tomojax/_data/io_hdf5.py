"""HDF5/NXtomo IO facade."""

from __future__ import annotations

from ._io_npz import convert, load_npz, save_npz
from ._io_nxtomo import load_nxtomo, save_nxtomo, validate_nxtomo
from ._io_types import (
    DatasetValue,
    JsonObject,
    JsonValue,
    LoadedDataset,
    LoadedNXTomo,
    NXTomoMetadata,
    SourceInfo,
    ValidationReport,
)

__all__ = [
    "DatasetValue",
    "JsonObject",
    "JsonValue",
    "LoadedDataset",
    "LoadedNXTomo",
    "NXTomoMetadata",
    "SourceInfo",
    "ValidationReport",
    "convert",
    "load_npz",
    "load_nxtomo",
    "save_npz",
    "save_nxtomo",
    "validate_nxtomo",
]
