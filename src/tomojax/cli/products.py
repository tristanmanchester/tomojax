"""Typed results prepared by adapters and published by the session owner."""

from __future__ import annotations

from typing import Any

import msgspec

from tomojax.cli.wire_options import Grid
from tomojax.wire import resources
from tomojax.wire.arrays import Array, Count, Id, Member, Npy, Record
from tomojax.wire.progress import Info

VOLUME_DATASET = "/entry/processing/tomojax/volume"

Placement = resources.Placement
Snapshot = resources.Snapshot


class Scan(Record, tag="scan"):
    """A reusable scan with its registered projections."""

    handle: Id
    projections: Array


class Frames(Record, tag="frames"):
    """Lazy raw input, retaining calibration."""

    handle: Id
    counts: Array


class Reconstruction(Record, tag="reconstruction"):
    """A volume can outlive or omit its source scan."""

    handle: Id
    volume: Array
    scan: Member | None
    method: Id


class Alignment(Record, tag="alignment"):
    """Reusable corrected geometry and optional working image."""

    handle: Id
    scan: Member
    poses: Array
    image: Array | None


class Simulation(Record, tag="simulation"):
    """Truth and the synthetic acquisition."""

    handle: Id
    scan: Member
    phantom: Array


class Layout(Record, tag="layout"):
    """Dimensions, units and coordinate sampling, without allocating or issuing pixels."""

    array: dict[str, Any]


class ArrayResult(Record, tag="array"):
    """Registered hyperslab or metadata for the original source."""

    array: Array


class StatisticsResult(Record, tag="statistics"):
    """Distribution over finite samples. Empty finite populations have null moments."""

    count: int
    non_finite: int
    mean: float | None
    std: float | None
    percentiles: dict[str, float | None]
    edges: tuple[float, ...]
    histogram: tuple[int, ...]
    unit: str


class PlaneResult(Record, tag="plane"):
    """One registered plane."""

    plane: Array


class Projections(Record, tag="projections"):
    """Forward projection in (view, v, u) order."""

    projections: Array


class Volume(Record, tag="volume"):
    """Backprojected or slab-reconstructed volume."""

    volume: Array


class Sweep(Record, tag="sweep"):
    """Ordered planes with a leading signed centre-offset coordinate."""

    planes: Array


class CorrectionResult(Record, tag="correction_preview"):
    """Registered measurements and retained original acquisition IDs."""

    before: Array
    after: Array
    views: Array


class Region(Record, tag="region"):
    """Estimated support and registered overlay planes."""

    handle: Id
    grid: Grid
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    arrays: dict[str, Array]


class Staged(Record):
    """Destination-local work owned by the committer after terminal acceptance."""

    destination: Id
    staged: Id
    force: bool


class Files(Record, tag="files"):
    """All committed output destinations, including adjacent manifests."""

    paths: tuple[Id, ...]
    plan: tuple[Staged, ...] = ()
    recovery: dict[str, list[str]] = msgspec.field(default_factory=dict)


class PythonResult(Record, tag="python"):
    """Captured streams, last expression, registered products and rendered images."""

    stdout: str
    stderr: str
    value: Any
    handles: dict[str, HandleProduct | ArrayResult]
    images: tuple[Npy, ...]


class Evaluation(Record, tag="evaluation"):
    """Additive held-out scores on a frozen native detector split."""

    split_fingerprint: str
    reference: dict[str, Any]
    scores: tuple[dict[str, Any], ...]


class Inspection(Record, tag="inspection"):
    """The public inspection report."""

    report: dict[str, Any]


class ReconPlan(Record, tag="reconstruction_plan"):
    """Method selection and original working grid."""

    method: Id
    reasons: tuple[str, ...]
    iterations: Count | None
    grid: Grid
    placement: Placement


class Level(Record):
    """A joint-fit level and its iteration budget."""

    grid: Grid
    iterations: Count
    image_iterations: Count


class JointPlan(Record, tag="joint_plan"):
    """Working grids in execution order."""

    reasons: tuple[str, ...]
    levels: tuple[Level, ...]
    placement: Placement


class CentrePlan(Record, tag="opposite_views_plan"):
    """Centre estimation uses paired projections."""

    pairs: Count
    reasons: tuple[str, ...]
    placement: Placement


class Validated(Record, tag="validated"):
    """Successful shared preflight, including resolved inputs and destinations."""

    details: dict[str, Any] = msgspec.field(default_factory=dict)


class Provenance(Record, omit_defaults=True):
    """Ordered input tree, preserving repeated inputs to combine and their histories."""

    operation: Id
    parameters: dict[str, Any]
    inputs: tuple[Provenance, ...]
    files: set[str] = msgspec.field(default_factory=set)


type HandleProduct = Scan | Frames | Reconstruction | Alignment | Simulation | Region
type DataProduct = (
    Layout
    | ArrayResult
    | StatisticsResult
    | PlaneResult
    | Projections
    | Volume
    | Sweep
    | CorrectionResult
    | Files
    | PythonResult
    | Evaluation
    | Inspection
    | ReconPlan
    | JointPlan
    | CentrePlan
    | Validated
)
type Product = HandleProduct | DataProduct


class Output(Record, tag="output"):
    """Published execution result."""

    value: Product
    info: Info
    manifest: Provenance
