"""Projection datasets on disk: load, save, validate and convert them.

Raw detector frames are read with :func:`tomojax.load_frames` and corrected
with :mod:`tomojax.corrections`. Expert helpers (acquisition readers, NXtomo
containers, inspection, JSON) are in :mod:`tomojax.io.api`.
"""

from tomojax.io._datasets import (
    ProjectionDataset,
    ValidationReport,
    build_geometry_from_dataset_metadata,
    load_dataset,
    load_tiff_stack,
    save_dataset,
    validate_dataset,
)

__all__ = [
    "ProjectionDataset",
    "ValidationReport",
    "build_geometry_from_dataset_metadata",
    "load_dataset",
    "load_tiff_stack",
    "save_dataset",
    "validate_dataset",
]
