"""Typed results prepared by adapters and published by the session owner."""

from __future__ import annotations

from typing import Any, Literal

from tomojax.cli.wire_options import Grid
from tomojax.wire.arrays import Array, Count, Id, Member, Record, Seconds
from tomojax.wire.progress import Info


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


class Files(Record, tag="files"):
    """All committed output destinations, including adjacent manifests."""

    paths: tuple[Id, ...]


class Inspection(Record, tag="inspection"):
    """The public inspection report."""

    report: dict[str, Any]


class Snapshot(Record):
    """Immutable executor sample. Bytes refer to the named allocator and device."""

    time: Seconds
    device: Id
    scope: Id
    used: Count | None
    peak: Count | None
    available: Count | None


class Placement(Record):
    """Resource estimate at planning time."""

    devices: tuple[Id, ...]
    strategy: Literal["resident", "slabbed"]
    host_bytes: Count
    device_bytes: Count
    snapshot: tuple[Snapshot, ...]


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
    """Successful shared preflight."""


class Provenance(Record):
    """Ordered input tree, preserving repeated inputs to combine and their histories."""

    operation: Id
    parameters: dict[str, Any]
    inputs: tuple[Provenance, ...]


type HandleProduct = Scan | Frames | Reconstruction | Alignment | Simulation | Region
type DataProduct = (
    PlaneResult
    | Projections
    | Volume
    | Sweep
    | CorrectionResult
    | Files
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
