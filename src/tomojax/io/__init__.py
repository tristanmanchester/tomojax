"""Public IO entry points for TomoJAX datasets.

Use :mod:`tomojax.io.api` for inspection, JSON, and Nexus-wrangling helpers
that are useful but not part of the package-root surface. Raw detector frames
are read with :func:`tomojax.load_frames` and corrected with
:mod:`tomojax.corrections`.
"""

from tomojax.io.api import (
    ProjectionDataset,
    ValidationReport,
    build_geometry_from_dataset_metadata,
    load_dataset,
    load_nikon_xtekct,
    load_tiff_stack,
    save_dataset,
    validate_dataset,
)

__all__ = [
    "ProjectionDataset",
    "ValidationReport",
    "build_geometry_from_dataset_metadata",
    "load_dataset",
    "load_nikon_xtekct",
    "load_tiff_stack",
    "save_dataset",
    "validate_dataset",
]
