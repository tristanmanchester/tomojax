"""Public IO entry points for TomoJAX datasets and preprocessing.

Use :mod:`tomojax.io.api` for inspection, contrast, JSON, and Nexus-wrangling
helpers that are useful but not part of the package-root surface.
"""

from tomojax.io.api import (
    JsonValue,
    LoadedNXTomo,
    NXTomoMetadata,
    PreprocessConfig,
    PreprocessResult,
    ProjectionDataset,
    ValidationReport,
    build_geometry_from_dataset_metadata,
    convert_dataset,
    drop_none,
    load_dataset,
    load_nxtomo,
    load_projection_payload,
    load_tiff_stack,
    normalize_json,
    preprocess_nxtomo,
    preprocess_tiff_stack,
    read_json_object,
    save_dataset,
    save_nxtomo,
    save_projection_payload,
    validate_dataset,
    validate_nxtomo,
    write_json_object,
)

__all__ = [
    "JsonValue",
    "LoadedNXTomo",
    "NXTomoMetadata",
    "PreprocessConfig",
    "PreprocessResult",
    "ProjectionDataset",
    "ValidationReport",
    "build_geometry_from_dataset_metadata",
    "convert_dataset",
    "drop_none",
    "load_dataset",
    "load_nxtomo",
    "load_projection_payload",
    "load_tiff_stack",
    "normalize_json",
    "preprocess_nxtomo",
    "preprocess_tiff_stack",
    "read_json_object",
    "save_dataset",
    "save_nxtomo",
    "save_projection_payload",
    "validate_dataset",
    "validate_nxtomo",
    "write_json_object",
]
