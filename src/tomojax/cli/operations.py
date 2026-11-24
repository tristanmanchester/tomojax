"""Typed commands admitted by a local worker, including ordered member references."""

from __future__ import annotations

from typing import Literal

import msgspec

from tomojax._parameters import AlignMethod, GeometryName, Method, Phantom, RegionName
from tomojax.cli import wire_options as w
from tomojax.wire.arrays import Count, Id, Member, Npy, Record, Storage

type Step = (
    w.BeamHardening
    | w.Paganin
    | w.RadialDistortion
    | w.RejectViews
    | w.Rings
    | w.Stripes
    | w.Zingers
)
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


class PathInput(Record, tag="path"):
    """File input opened on the execution queue."""

    path: Id


type Source = PathInput | Member
type RootKind = Literal["frames", "scan", "reconstruction", "alignment", "simulation"]


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
    apply_poses: bool = True


class Inspect(Record, tag="inspect"):
    """Read a file's inspection report."""

    path: Id


class Correct(Record, tag="correct"):
    """Correct a scan or raw frames."""

    source: Source
    corrections: tuple[Step, ...] = ()
    epsilon: float = 1e-6
    views_per_batch: int | None = None
    binning: int = 1


class Reconstruct(Record, tag="reconstruct"):
    """Reconstruct the supplied scan on its resolved grid."""

    source: Source
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


class Align(Record, tag="align"):
    """Return geometry-corrected scan and fitted pose table."""

    source: Source
    fit: str | tuple[str, ...] = "translations"
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
    artefacts: w.SimulationArtefacts | None = None
    grid: w.Grid | None = None
    detector: w.Detector | None = None
    seed: int = 0


class Project(Record, tag="project"):
    """Project registered voxels with the source scan geometry."""

    source: Source
    volume: Storage
    grid: w.Grid | None = None
    devices: tuple[Id, ...] | None = None


class Backproject(Record, tag="backproject"):
    """Apply the matched transpose with the source scan geometry."""

    source: Source
    projections: Storage
    grid: w.Grid | None = None
    devices: tuple[Id, ...] | None = None


class Save(Record, tag="save"):
    """Stage the dataset and adjacent manifest."""

    source: Member
    path: Id
    force: bool = False
    include_scan: bool = False


class Export(Record, tag="export"):
    """Stage the complete raw or TIFF export with its sidecars."""

    source: Member
    path: Id
    force: bool = False
    dtype: Literal["float32", "uint16"] = "float32"
    range: tuple[float, float] | None = None
    prefix: str = "slice"


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


class Window(Record):
    """Half-open ROI in storage axis order. Omitted bounds select the whole array."""

    roi: tuple[tuple[int, int], ...] | None = None
    stride: tuple[int, ...] | None = None


class Read(Window, tag="read", kw_only=True):
    """Copy a registered hyperslab (up to 16 MiB), or return only its registration."""

    source: Storage | PathInput
    metadata_only: bool = False


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


class Slice(Reconstruct, tag="slice", kw_only=True):
    """Select a plane from the specified full method."""

    axis: Literal["z", "y", "x"]
    index: Count


class CentreSweep(Slice, tag="centre_sweep", kw_only=True):
    """Reconstruct at signed horizontal offsets in detector pixels."""

    centres: tuple[float, ...]


class CorrectionPreview(Correct, tag="correction_preview", kw_only=True):
    """Inspect original view IDs and corrected rows in a binned half-open interval."""

    rows: tuple[int, int]


class RegionPreview(Record, tag="region_preview"):
    """Estimate physical bounds and registered overlays."""

    source: Source


type Workflow = (
    Correct
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
)
type Interactive = Plane | Slice | CentreSweep | CorrectionPreview | RegionPreview


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


type ExecutionInput = Read | Statistics | Run | Interactive | Inspect | Open | Plan | Validate


def input_references(command: ExecutionInput) -> tuple[Member | Npy, ...]:
    """Return pin targets in input order, including nested configs and repeated members."""
    references: list[Member | Npy] = []

    def collect(value: object) -> None:
        if isinstance(value, Member) or (isinstance(value, Npy) and value.artifact is not None):
            references.append(value)
        elif isinstance(value, Record):
            for field in msgspec.structs.fields(value):
                collect(getattr(value, field.name))
        elif isinstance(value, tuple):
            for item in value:
                collect(item)

    collect(command)
    return tuple(references)
