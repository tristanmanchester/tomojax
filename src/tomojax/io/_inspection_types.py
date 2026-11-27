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
    center: object


class AlignmentReport(TypedDict):
    """Persisted alignment metadata discovery report."""

    found: bool
    params_found: bool
    params_shape: list[int] | None
    angle_offset_found: bool
    angle_offset_shape: list[int] | None
    misalign_spec_found: bool
    gauge_fix_found: bool


class VolumeReport(TypedDict):
    """Saved reconstruction discovery report."""

    found: bool
    shape: list[int] | None
    axes: str | None
    method: str | None


class AnglesReport(TypedDict):
    """Angle metadata discovery report."""

    found: bool
    path: str | None
    count: int | None
    units: str | None
    min: float | None
    max: float | None
    coverage: float | None


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
    """Top-level dataset inspection report.

    Attributes
    ----------
    schema_version : int
        Version of the inspection report schema.
    input_path : str
        Inspected input path.
    projection : dict
        Projection location, shape, dtype, statistics and nonfinite counts.
    angles : dict
        Rotation-angle discovery, units and angular coverage.
    geometry : dict
        Geometry family and recorded metadata keys.
    detector_metadata : dict
        Detector dimensions, sampling and center.
    flats_darks : dict
        Calibration frame discovery and frame-label counts.
    corrections : list of str
        Recorded steps converting counts to line integrals.
    alignment : dict
        Saved alignment parameter and gauge information.
    memory_estimates : dict
        Heuristic reconstruction working-set estimates, not allocation guarantees.
    volume : dict
        Saved reconstruction shape, axis order and method.
    """

    schema_version: int
    input_path: str
    projection: ProjectionReport
    angles: AnglesReport
    geometry: GeometryReport
    detector_metadata: DetectorMetadataReport
    flats_darks: FlatsDarksReport
    corrections: list[str]  # how the projections were made from detector frames
    alignment: AlignmentReport
    memory_estimates: MemoryEstimatesReport
    volume: VolumeReport


class LoadingReport(TypedDict):
    """What loading the file as a scan reads, with no arguments beyond its path.

    Attributes
    ----------
    message : str or None
        Why the file does not load as it is, naming the argument to supply; None
        when it loads.
    length_unit : str or None
        Unit of the geometry's lengths: ``"pixel"`` when the file records no pixel size.
    center : str or None
        Where the rotation center comes from (``detector_center_status``):
        ``"recorded"``, ``"provided"``, ``"inferred"``, ``"nominal"``, or
        ``"not_recorded"``, the detector middle until a center is fitted.
    geometry : str or None
        ``"not_recorded"`` when the file records no rotation axis or pixel scale:
        reconstruction and alignment need ``geometry`` declared when loading.
    frame_records : dict or None
        The lengths of the frames, ``image_key`` and angles when they differ, as
        when a scan stopped early; loading reads the frames all of them cover.
    """

    message: str | None
    length_unit: str | None
    center: str | None
    geometry: str | None
    frame_records: dict[str, int] | None


class Inspection(InspectionReport):
    """A dataset's inspection report and its check (:func:`tomojax.inspect`).

    Attributes
    ----------
    loading : dict or None
        What loading the file as a scan reads (:class:`LoadingReport`); None when
        the file holds no projections.
    checkpoint : dict or None
        For a checkpoint directory (``checkpoint=`` of a stopped run), what it
        holds: the run (operation, method, settings, input file, version), how it
        was requested (a command line, or a worker operation and its inputs'
        provenance), when it was created and last saved, each saved entry's
        scalars, its size in bytes, whether a running process holds it, and how to
        resume. None for a dataset, whose sections are then the ones above.
    valid : bool
        Whether the check found no issue.
    issues : list of str
        What the check found: missing calibration, angles or frames.
    """

    loading: LoadingReport | None
    checkpoint: dict[str, object] | None
    valid: bool
    issues: list[str]


__all__ = [
    "AlignmentReport",
    "AnglesReport",
    "DetectorMetadataReport",
    "DetectorShapeReport",
    "FlatsDarksReport",
    "GeometryReport",
    "Inspection",
    "InspectionReport",
    "LoadingReport",
    "MemoryEstimatesReport",
    "NonfiniteReport",
    "ProjectionReport",
    "ProjectionStatsReport",
    "VolumeReport",
    "WorkingSetEstimate",
]
