"""Public IO entry points for TomoJAX datasets and preprocessing.

Use :mod:`tomojax.io.api` for inspection, contrast, JSON, and Nexus-wrangling
helpers that are useful but not part of the package-root surface.
"""

from tomojax.io.api import (
    PreprocessConfig,
    PreprocessResult,
    ProjectionDataset,
    ValidationReport,
    build_geometry_from_dataset_metadata,
    load_dataset,
    load_nikon_xtekct,
    load_tiff_stack,
    preprocess_nxtomo,
    preprocess_tiff_stack,
    save_dataset,
    validate_dataset,
)

__all__ = [
    "PreprocessConfig",
    "PreprocessResult",
    "ProjectionDataset",
    "ValidationReport",
    "build_geometry_from_dataset_metadata",
    "load_dataset",
    "load_nikon_xtekct",
    "load_tiff_stack",
    "preprocess_nxtomo",
    "preprocess_tiff_stack",
    "save_dataset",
    "validate_dataset",
]
