"""Typed inspection report contracts for NXtomo/HDF5 diagnostics."""

from __future__ import annotations

from typing import TypedDict


class DetectorShapeReport(TypedDict):
    """Detector pixel shape in v/u order."""

    nv: int
    nu: int


class ProjectionStatsReport(TypedDict):
    """Summary statistics for a projection stack."""

    min: float | None
    p01: float | None
    mean: float | None
    p50: float | None
    p99: float | None
    max: float | None


class NonfiniteReport(TypedDict):
    """Non-finite value counts for a projection stack."""

    nan_count: int | None
    posinf_count: int | None
    neginf_count: int | None
    inf_count: int | None


class ProjectionReport(TypedDict):
    """Projection dataset discovery and summary report."""

    found: bool
    path: str | None
    shape: list[int] | None
    dtype: str | None
    n_views: int | None
    detector_shape: DetectorShapeReport | None
    storage_bytes: int | None
    stats: ProjectionStatsReport
    nonfinite: NonfiniteReport


class DetectorMetadataReport(TypedDict):
    """Detector metadata discovery report."""

    found: bool
    nu: object
    nv: object
    du: object
    dv: object
    det_center: object


class AlignmentReport(TypedDict):
    """Persisted alignment metadata discovery report."""

    found: bool
    params_found: bool
    params_shape: list[int] | None
    angle_offset_found: bool
    angle_offset_shape: list[int] | None
    misalign_spec_found: bool
    gauge_fix_found: bool


class AnglesReport(TypedDict):
    """Angle metadata discovery report."""

    found: bool
    path: str | None
    count: int | None
    units: str | None
    min_deg: float | None
    max_deg: float | None
    coverage_deg: float | None


class GeometryReport(TypedDict):
    """Persisted geometry metadata discovery report."""

    found: bool
    type: str | None
    meta_found: bool
    meta_keys: list[str]


class FlatsDarksReport(TypedDict):
    """Flat/dark frame discovery report."""

    image_key_found: bool
    image_key_path: str | None
    image_key_counts: dict[str, int]
    sample_count: int
    flats_present: bool
    darks_present: bool
    flat_count: int
    dark_count: int


class PreprocessReport(TypedDict):
    """Preprocessing metadata discovery report."""

    found: bool
    output_domain: str | None
    formula: str | None
    epsilon: str | None
    clip_min: str | None
    paths: dict[str, str]
    overrides: dict[str, str | None]
    crop_bounds: dict[str, object] | None


class WorkingSetEstimate(TypedDict):
    """Memory estimate for a reconstruction mode."""

    estimated_working_set_bytes: int


class MemoryEstimatesReport(TypedDict):
    """Heuristic memory estimate report."""

    feasible: bool
    reconstruction_grid_shape: list[int] | None
    input_projection_bytes: int | None
    modes: dict[str, WorkingSetEstimate]
    notes: str


class InspectionReport(TypedDict):
    """Top-level dataset inspection report."""

    schema_version: int
    input_path: str
    projection: ProjectionReport
    angles: AnglesReport
    geometry: GeometryReport
    detector_metadata: DetectorMetadataReport
    flats_darks: FlatsDarksReport
    preprocess: PreprocessReport
    alignment: AlignmentReport
    memory_estimates: MemoryEstimatesReport


__all__ = [
    "AlignmentReport",
    "AnglesReport",
    "DetectorMetadataReport",
    "DetectorShapeReport",
    "FlatsDarksReport",
    "GeometryReport",
    "InspectionReport",
    "MemoryEstimatesReport",
    "NonfiniteReport",
    "PreprocessReport",
    "ProjectionReport",
    "ProjectionStatsReport",
    "WorkingSetEstimate",
]
