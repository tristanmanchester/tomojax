"""Compatibility exports for dataset inspection.

The implementation moved to `tomojax.io._inspection`; new code should import
the public `tomojax.io` facade.
"""

from tomojax.io._inspection import (
    AlignmentReport,
    DetectorMetadataReport,
    DetectorShapeReport,
    InspectionReport,
    NonfiniteReport,
    ProjectionReport,
    ProjectionStatsReport,
    _attr_to_str,
    _dataset_at,
    format_inspection_report,
    inspect_dataset,
    inspect_nxtomo,
    save_projection_quicklook,
)

__all__ = [
    "AlignmentReport",
    "DetectorMetadataReport",
    "DetectorShapeReport",
    "InspectionReport",
    "NonfiniteReport",
    "ProjectionReport",
    "ProjectionStatsReport",
    "_attr_to_str",
    "_dataset_at",
    "format_inspection_report",
    "inspect_dataset",
    "inspect_nxtomo",
    "save_projection_quicklook",
]
