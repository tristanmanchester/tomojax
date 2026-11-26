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


class DisplayPlane(Record):
    """A sampled plane and its position on the original axes.

    ``window`` holds the (start, stop, step) of source indices read along each axis.
    """

    array: Array
    dimension: int | None
    index: int
    sampling: tuple[str, ...]
    window: tuple[tuple[int, int, int], ...]


class DisplayPlanes(Record, tag="display_planes"):
    """Requested preview planes from a single source registration."""

    planes: tuple[DisplayPlane, ...]


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
    """Ordered planes with a leading signed center-offset coordinate."""

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


class Feature(Record, tag="feature"):
    """A named mask, its size and extent, and the source values inside it.

    ``bounds`` holds a half-open index interval per axis; ``mean`` and ``std`` are
    over the finite source values inside the mask. ``recipe`` is how it was made,
    with the level, band and noise a seeded region resolved.
    """

    handle: Id
    name: str
    mask: Array
    voxels: int
    bounds: tuple[tuple[int, int], ...]
    mean: float | None
    std: float | None
    recipe: dict[str, Any]


class Files(Record, tag="files"):
    """All committed output destinations, including adjacent manifests."""

    paths: tuple[Id, ...]


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


class CenterPlan(Record, tag="opposite_views_plan"):
    """Center estimation uses paired projections."""

    pairs: Count
    reasons: tuple[str, ...]
    placement: Placement


class FocusPlan(Record, tag="sinogram_focus_plan"):
    """Interior angular samples for the global search and native ray pairs for refinement."""

    views: Count
    binning: Count
    pairs: Count
    reasons: tuple[str, ...]
    placement: Placement


class Validated(Record, tag="validated"):
    """Successful shared preflight, including resolved inputs and destinations."""

    details: dict[str, Any] = msgspec.field(default_factory=dict)


class Provenance(Record):
    """Ordered input tree, preserving repeated inputs to combine and their histories."""

    operation: Id
    parameters: dict[str, Any]
    inputs: tuple[Provenance, ...]


type HandleProduct = Scan | Frames | Reconstruction | Alignment | Simulation | Region | Feature
type DataProduct = (
    Layout
    | DisplayPlanes
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
    | CenterPlan
    | FocusPlan
    | Validated
)
type Product = HandleProduct | DataProduct


class Output(Record, tag="output"):
    """Published execution result."""

    value: Product
    info: Info
    manifest: Provenance
