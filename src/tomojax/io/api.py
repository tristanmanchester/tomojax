"""Public API for dataset IO and metadata normalization."""

from tomojax.io._datasets import (
    ProjectionDataset,
    ValidationReport,
    build_geometry_from_dataset_metadata,
    convert_dataset,
    load_dataset,
    load_projection_payload,
    load_tiff_stack,
    save_dataset,
    save_projection_payload,
    validate_dataset,
)
from tomojax.io._inspection import (
    InspectionReport,
    format_inspection_report,
    inspect_dataset,
    save_projection_quicklook,
)
from tomojax.io._json import JsonValue, drop_none, normalize_json
from tomojax.io._preprocess import PreprocessConfig, PreprocessResult, preprocess_nxtomo

__all__ = [
    "InspectionReport",
    "JsonValue",
    "PreprocessConfig",
    "PreprocessResult",
    "ProjectionDataset",
    "ValidationReport",
    "build_geometry_from_dataset_metadata",
    "convert_dataset",
    "drop_none",
    "format_inspection_report",
    "inspect_dataset",
    "load_dataset",
    "load_projection_payload",
    "load_tiff_stack",
    "normalize_json",
    "preprocess_nxtomo",
    "save_dataset",
    "save_projection_payload",
    "save_projection_quicklook",
    "validate_dataset",
]
