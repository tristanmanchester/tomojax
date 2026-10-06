"""Public API for dataset IO and metadata normalization."""

from tomojax.io._angles import load_angles
from tomojax.io._contrast import absorption_to_transmission, flat_dark_to_absorption
from tomojax.io._datasets import (
    LoadedNXTomo,
    NXTomoMetadata,
    ProjectionDataset,
    ValidationReport,
    build_geometry_from_dataset_metadata,
    convert_dataset,
    load_dataset,
    load_nxtomo,
    load_projection_payload,
    load_tiff_stack,
    save_dataset,
    save_nxtomo,
    save_projection_payload,
    validate_dataset,
    validate_nxtomo,
)
from tomojax.io._inspection import inspect_dataset, projection_stats
from tomojax.io._inspection_format import format_inspection_report
from tomojax.io._inspection_types import (
    InspectionReport,
)
from tomojax.io._json import JsonValue, drop_none, normalize_json, read_json_object
from tomojax.io._nikon import load_nikon_xtekct
from tomojax.io._preprocess import (
    PreprocessConfig,
    PreprocessResult,
    preprocess_nxtomo,
    preprocess_tiff_stack,
)
from tomojax.io._quicklook import save_projection_quicklook
from tomojax.io._real_laminography import RealLaminographyInput, load_real_laminography_input

__all__ = [
    "InspectionReport",
    "JsonValue",
    "LoadedNXTomo",
    "NXTomoMetadata",
    "PreprocessConfig",
    "PreprocessResult",
    "ProjectionDataset",
    "RealLaminographyInput",
    "ValidationReport",
    "absorption_to_transmission",
    "build_geometry_from_dataset_metadata",
    "convert_dataset",
    "drop_none",
    "flat_dark_to_absorption",
    "format_inspection_report",
    "inspect_dataset",
    "load_angles",
    "load_dataset",
    "load_nikon_xtekct",
    "load_nxtomo",
    "load_projection_payload",
    "load_real_laminography_input",
    "load_tiff_stack",
    "normalize_json",
    "preprocess_nxtomo",
    "preprocess_tiff_stack",
    "projection_stats",
    "read_json_object",
    "save_dataset",
    "save_nxtomo",
    "save_projection_payload",
    "save_projection_quicklook",
    "validate_dataset",
    "validate_nxtomo",
]
