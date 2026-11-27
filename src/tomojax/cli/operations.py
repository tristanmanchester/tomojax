"""Typed commands admitted by a local worker, including ordered member references."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

import msgspec

from tomojax._parameters import (
    FLAT_WEIGHTS,
    AlignMethod,
    FlatWeighting,
    GeometryName,
    Method,
    Phantom,
    RegionName,
)
from tomojax.cli import wire_options as w
from tomojax.wire.arrays import Count, Hdf5, Id, Member, Npy, Positive, Record, Storage

if TYPE_CHECKING:
    from collections.abc import Iterator

type Step = w.BeamHardening | w.Paganin | w.RadialDistortion | w.RejectViews | w.Rings | w.Zingers
type Config = (
    w.FBPConfig
    | w.FourierConfig
    | w.CGLSConfig
    | w.FISTAConfig
    | w.SPDHGConfig
    | w.SIRTConfig
    | w.MLEMConfig
)
type Region = RegionName | Member


ACQUISITION_ARRAY_ERROR = (
    "An acquisition path cannot supply an array. Use open and select .projections, or use "
    "kind=frames and .counts for raw counts. Use preview with correction_preview for corrected "
    "rows, or inspect with preview=true for a first look."
)


class PathInput(Record, tag="path"):
    """File input opened on the execution queue."""

    path: Id


type Source = PathInput | Member
type RootKind = Literal["frames", "scan", "reconstruction", "alignment", "simulation", "feature"]


class Open(Record, tag="open"):
    """Open a file with the named public loader."""

    path: Id
    kind: RootKind
    flats: Storage | float | str | None = None
    darks: Storage | float | str | None = None
    angles: Storage | None = None
    detector: w.Detector | None = None
    pixel_size: float | tuple[float, float] | None = None
    grid: w.Grid | None = None
    geometry: w.DeclaredGeometry | None = None
    exposure_times: float | Storage | None = None
    flat_exposure_times: float | Storage | None = None
    data_path: str | None = None
    image_key_path: str | None = None
    angles_path: str | None = None
    apply_poses: bool | msgspec.UnsetType = msgspec.UNSET


class Inspect(Record, tag="inspect"):
    """Read a file's inspection report."""

    path: Id


class Correct(Record, tag="correct"):
    """Correct a scan or raw frames."""

    source: Source
    corrections: tuple[Step, ...] = ()
    epsilon: float = 1e-6
    flat_weights: FlatWeighting = FLAT_WEIGHTS
    views_per_batch: int | None = None
    binning: int = 1


class ReconstructSettings(Record, tag="reconstruct", kw_only=True):
    """Reconstruct the supplied scan on its resolved grid."""

    method: Method = "auto"
    grid: w.Grid | None = None
    region: Region | None = None
    config: Config | None = None
    filter: str | None = None
    iterations: int | None = None
    tv_weight: float | None = None
    nonnegative: bool | None = None
    warm_start: bool | None = None
    seed: int | None = None
    devices: tuple[Id, ...] | None = None


class Reconstruct(ReconstructSettings, tag="reconstruct"):
    """Reconstruct the supplied scan on its resolved grid."""

    source: Source
    checkpoint: str | None = None


class ReconstructSlabs(Record, tag="reconstruct_slabs"):
    """Consume each yielded slab into one staged disk-backed volume."""

    source: Source
    path: Id
    method: Literal["fbp", "fourier", "sirt", "mlem"]
    force: bool = False
    grid: w.Grid | None = None
    config: w.FBPConfig | w.FourierConfig | w.SIRTConfig | w.MLEMConfig | None = None
    corrections: tuple[Step, ...] = ()
    warm_start: bool | None = None
    epsilon: float = 1e-6
    flat_weights: FlatWeighting = FLAT_WEIGHTS
    checkpoint: str | None = None


class AlignSettings(Record, tag="align", kw_only=True):
    """Return geometry-corrected scan and fitted pose table."""

    fit: str | tuple[str, ...] | None = None
    grid: w.Grid | None = None
    region: Region | None = None
    method: AlignMethod | None = None
    motion: Literal["independent", "smooth"] | None = None
    iterations: int | None = None
    image_iterations: int | None = None
    tolerance: float | None = None
    interpolation: Literal["linear", "cubic"] | None = None
    resolution: int | None = None
    config: w.AlignConfig | None = None


class Align(AlignSettings, tag="align"):
    """Return geometry-corrected scan and fitted pose table."""

    source: Source
    checkpoint: str | None = None


class Interleaved(Record, tag="interleaved"):
    """Hold out regularly spaced views with optional neighboring guard views."""

    stride: int = 5
    offset: int = 2
    guard: int = 0


class Angular(Record, tag="angular"):
    """Hold out an interior angular block with guards in degrees."""

    start: float
    width: float
    guard: float = 0
    period: float = 360


class Evaluate(Record, tag="evaluate"):
    """Score settings against shared held-out line integrals. Corrections run per partition."""

    source: Source
    candidates: tuple[ReconstructSettings | AlignSettings, ...]
    split: Interleaved | Angular = Interleaved()
    corrections: tuple[Step, ...] = ()
    reference: ReconstructSettings = ReconstructSettings(method="cgls", iterations=30)


class Simulate(Record, tag="simulate"):
    """Produce both a scan and its ground truth."""

    phantom: Phantom = "shepp"
    size: int = 64
    n_views: int | None = None
    geometry: GeometryName = "parallel"
    rotation: float | None = None
    tilt: float | None = None
    source_to_axis: float | None = None
    source_to_detector: float | None = None
    poisson_scale: float = 0.0
    translation_sigma: float = 0.0
    artifacts: w.SimulationArtifacts | None = None
    grid: w.Grid | None = None
    detector: w.Detector | None = None
    seed: int = 0


class Project(Record, tag="project"):
    """Project registered voxels with the source scan geometry."""

    source: Source
    volume: Storage | PathInput
    grid: w.Grid | None = None
    devices: tuple[Id, ...] | None = None


class Backproject(Record, tag="backproject"):
    """Apply the matched transpose with the source scan geometry."""

    source: Source
    projections: Storage | PathInput
    grid: w.Grid | None = None
    devices: tuple[Id, ...] | None = None


class Save(Record, tag="save"):
    """Save the dataset with its provenance."""

    source: Member
    path: Id
    force: bool = False
    include_scan: bool = False


class Export(Record, tag="export"):
    """Export one registered member in the format selected by its destination."""

    source: Member
    path: Id
    force: bool = False
    dtype: Literal["float32", "uint16", "uint8"] | None = None
    range: tuple[float, float] | None = None
    crop: tuple[tuple[int | None, int | None] | None, ...] | None = None
    binning: Positive = 1
    axis: str | None = None
    compress: bool | None = None


class Select(Record, tag="select"):
    """Select ordered view indices."""

    source: Source
    views: tuple[int, ...]


class Crop(Record, tag="crop"):
    """Crop half-open detector intervals."""

    source: Source
    rows: tuple[int, int]
    cols: tuple[int, int]


class Bin(Record, tag="bin"):
    """Mean-bin both detector dimensions."""

    source: Source
    binning: int


class Poses(Record, tag="poses"):
    """Replace poses (alpha, beta, phi in degrees, then x, z, y in length units) or clear them."""

    source: Source
    poses: Storage | None


class Combine(Record, tag="combine"):
    """Combine scans in the given order, retaining all input provenance."""

    sources: tuple[Source, ...]
    name: str | None = None


type Position = tuple[float, ...]
Radius = Annotated[float, msgspec.Meta(gt=0)]
Extent = tuple[float, float]


class Box(Record, tag="box"):
    """Voxels between two opposite corners, inclusive, in array indices."""

    corners: tuple[Position, Position]


class Sphere(Record, tag="sphere"):
    """Voxels within ``radius`` index units of ``at``."""

    at: Position
    radius: Radius


class Cylinder(Record, tag="cylinder"):
    """A disc of ``radius`` around ``at`` across ``axis``, extruded over ``extent``."""

    at: Position
    radius: Radius
    axis: Id
    extent: Extent


class Prism(Record, tag="prism"):
    """A polygon across ``axis`` through ``points``, extruded over ``extent`` along it."""

    points: tuple[Position, ...]
    axis: Id
    extent: Extent


class Seeded(Record, tag="seeded"):
    """The voxels connected to the seeds whose value stays near the value at the seeds.

    The level is the median around each seed, averaged over seeds. ``tolerance`` is in
    units of the local noise or, with ``tolerance_unit="value"``, of the source.
    ``smoothing`` is a Gaussian sigma in voxels applied first, for very noisy data; it
    pulls the edges in. ``max_radius`` limits growth to that many index units from the
    nearest seed.
    """

    seeds: tuple[Position, ...]
    tolerance: Annotated[float, msgspec.Meta(gt=0)] = 3.0
    tolerance_unit: Literal["noise", "value"] = "noise"
    smoothing: Annotated[float, msgspec.Meta(ge=0)] = 0.0
    max_radius: Radius | None = None


class Band(Record, tag="band"):
    """Voxels whose signed distance from a feature lies in (inner, outer] index units.

    Voxels beyond the array are background. Inside the feature the signed distance
    is one minus the Euclidean distance to the nearest background voxel center,
    so a face neighbor of background has distance zero. Outside the feature it
    is the distance to the nearest feature voxel center. Both bands stop at the grid.
    """

    feature: Member
    outer: float
    inner: float = 0.0


type Recipe = Box | Sphere | Cylinder | Prism | Seeded | Band


class Feature(Record, tag="feature"):
    """Make a named mask on the source array's grid."""

    source: Storage | PathInput
    recipe: Recipe
    name: str | None = None


class Window(Record):
    """Half-open ROI in storage axis order. Omitted bounds select the whole array."""

    roi: tuple[tuple[int, int], ...] | None = None
    stride: tuple[int, ...] | None = None


class Read(Window, tag="read", kw_only=True):
    """Copy a registered hyperslab (up to 16 MiB), or return only its registration."""

    source: Storage | PathInput
    metadata_only: bool = False
    mode: Literal["projection", "sinogram", "planes"] | None = None
    axis: str | None = None
    index: int | None = None


class Statistics(Window, tag="statistics", kw_only=True):
    """Exact finite-value statistics over an optionally strided ROI."""

    source: Storage | PathInput
    percentiles: tuple[float, ...] = (1, 5, 50, 95, 99)
    bins: int = 32


class Plane(Record, tag="plane"):
    """Browse one plane from a registered volume."""

    source: Member
    axis: Literal["z", "y", "x"]
    index: Count


class Slice(ReconstructSettings, tag="slice", kw_only=True):
    """Select a plane from the specified full method."""

    source: Source
    axis: Literal["z", "y", "x"]
    index: Count


class CenterSweep(Slice, tag="center_sweep", kw_only=True):
    """Reconstruct at signed horizontal offsets in detector pixels."""

    centers: tuple[float, ...]


class CorrectionPreview(Correct, tag="correction_preview", kw_only=True):
    """Inspect original view IDs and corrected rows in a binned half-open interval."""

    rows: tuple[int, int]


class RegionPreview(Record, tag="region_preview"):
    """Estimate physical bounds and registered overlays."""

    source: Source


type Workflow = (
    Correct
    | Evaluate
    | Reconstruct
    | ReconstructSlabs
    | Align
    | Simulate
    | Project
    | Backproject
    | Save
    | Export
    | Select
    | Crop
    | Bin
    | Poses
    | Combine
    | Feature
)
type Interactive = Plane | Slice | CenterSweep | CorrectionPreview | RegionPreview


class Run(Record, tag="run"):
    """Execute a workflow."""

    operation: Workflow
    previews: bool = False


class Plan(Record, tag="plan"):
    """Resolve budgets, grids and placement estimates."""

    operation: Reconstruct | Align


class Validate(Record, tag="validate"):
    """Apply execution preflight to the proposed operation."""

    operation: Workflow | Interactive | Open


class Python(Record, tag="python"):
    """Run code with an admission snapshot of the live objects and arrays."""

    code: str = ""
    reset: bool = False
    traceback: bool = False
    aliases: dict[str, Member | Npy] = msgspec.field(default_factory=dict)


type ExecutionInput = (
    Python | Read | Statistics | Run | Interactive | Inspect | Open | Plan | Validate
)


def input_values(command: object) -> Iterator[Record]:
    """Visit typed operation arguments in input order, including nested config records."""
    if isinstance(command, Record):
        yield command
        for field in msgspec.structs.fields(command):
            yield from input_values(getattr(command, field.name))
    elif isinstance(command, tuple):
        for item in command:
            yield from input_values(item)


def input_files(command: ExecutionInput) -> Iterator[str]:
    """Caller-owned file roots read by this request."""
    for value in input_values(command):
        if isinstance(value, Open | Inspect | PathInput | Hdf5) or (
            isinstance(value, Npy) and value.artifact is None
        ):
            yield value.path
        if isinstance(value, Open):
            for calibration in (value.flats, value.darks):
                if isinstance(calibration, str):
                    yield calibration


def input_references(command: ExecutionInput) -> tuple[Member | Npy, ...]:
    """Return pin targets in input order, including nested configs and repeated members."""
    return tuple(
        value
        for value in input_values(command)
        if isinstance(value, Member) or (isinstance(value, Npy) and value.artifact is not None)
    )
