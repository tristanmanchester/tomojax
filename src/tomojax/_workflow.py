"""TomoJAX's workflow API: scans, reconstructions and alignments.

A :class:`Scan` is projections together with the geometry that produced them;
every operation takes one and returns a result that carries its geometry, so
results can be reconstructed, aligned again or saved without restating it::

    import tomojax as tj

    scan = tj.load("scan.nxs")
    recon = tj.reconstruct(scan)  # FBP, or FDK for cone beams
    result = tj.align(scan, mode="cor-then-pose")
    tj.save("recon.nxs", tj.reconstruct(result.scan, method="cgls"))
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from tomojax.geometry import (
    ConeGeometry,
    LaminographyGeometry,
    ParallelGeometry,
    RotationAxisGeometry,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from os import PathLike

    import jax

    from tomojax.alignment import AlignConfig
    from tomojax.geometry import Detector, Geometry, Grid
    from tomojax.io import ProjectionDataset

type Method = Literal["fbp", "cgls", "fista", "spdhg"]
METHODS: tuple[Method, ...] = ("fbp", "cgls", "fista", "spdhg")

# Geometry metadata keys a geometry object owns; other keys (provenance) carry over.
_GEOMETRY_KEYS = frozenset(
    {"cone_beam", "tilt_deg", "tilt_about", "axis_unit_lab", "detector_roll_deg"}
)


@dataclass(frozen=True)
class Scan:
    """Projections and the geometry that produced them.

    ``projections`` are ``(views, rows, columns)`` line integrals (absorption,
    not intensities); ``geometry`` describes every view, and carries the
    reconstruction ``grid`` and the ``detector``. A scan loaded from a file or
    returned by :func:`align` may carry per-view pose corrections, which every
    operation applies (see :attr:`poses`).
    """

    projections: np.ndarray | jax.Array
    geometry: Geometry
    name: str = "sample"
    source: ProjectionDataset | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        shape = tuple(self.projections.shape)
        detector = self.geometry.detector
        views = len(self.geometry.thetas_deg)  # pyright: ignore[reportAttributeAccessIssue]
        if shape != (views, detector.nv, detector.nu):
            raise ValueError(
                f"Scan: projections are {shape} but the geometry has {views} views of "
                f"{detector.nv} rows x {detector.nu} columns"
            )

    @property
    def grid(self) -> Grid:
        """The reconstruction grid."""
        return self.geometry.grid

    @property
    def detector(self) -> Detector:
        """The detector."""
        return self.geometry.detector

    @property
    def angles(self) -> np.ndarray:
        """Rotation angle of each view, in degrees."""
        return np.asarray(self.geometry.thetas_deg, dtype=np.float64)  # pyright: ignore[reportAttributeAccessIssue]

    @property
    def poses(self) -> np.ndarray | None:
        """Per-view pose corrections, ``(views, 6)`` as alpha, beta, phi, dx, dz, dy.

        Rotations in radians, translations in the geometry's length unit, in
        the detector frame. None when the scan carries no corrections.
        """
        params = getattr(self.geometry, "align_params", None)
        return None if params is None else np.asarray(params)


@dataclass(frozen=True)
class Reconstruction:
    """A reconstructed volume, the scan it came from and how it was made.

    ``volume`` is ``(nx, ny, nz)`` on ``grid``, in the reciprocal of the
    geometry's length unit (a NumPy or JAX array; ``np.asarray`` gives NumPy).
    ``info`` records the method's settings.
    """

    volume: np.ndarray | jax.Array
    scan: Scan
    grid: Grid
    method: str
    info: Mapping[str, object]


@dataclass(frozen=True)
class Alignment:
    """The result of :func:`align`.

    ``scan`` is the input scan with the estimated geometry and per-view poses
    applied: reconstruct it, align it again or save it. ``volume`` is the
    reconstruction the alignment converged with, ``poses`` the ``(views, 6)``
    corrections (see :attr:`Scan.poses`), and ``info`` the solver's record.
    """

    scan: Scan
    volume: np.ndarray | jax.Array
    poses: np.ndarray
    info: Mapping[str, object]


# ----------------------------------------------------------------------------- files


def load(path: str | PathLike[str], *, apply_alignment: bool = True) -> Scan:
    """Load a scan from a TomoJAX dataset (``.nxs``, ``.h5``, ``.npz``) or a Nikon ``.xtekct``.

    A saved alignment (from :func:`align` or ``tomojax align``) is applied
    unless ``apply_alignment`` is False. TIFF stacks need their geometry
    stated: import them with ``tomojax import`` or :func:`tomojax.io.load_tiff_stack`.
    """
    from tomojax.io import load_dataset, load_nikon_xtekct

    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"no such file: {file}")
    record = load_nikon_xtekct(file) if file.suffix.lower() == ".xtekct" else load_dataset(file)
    return _scan_from_record(record, apply_alignment=apply_alignment)


def load_reconstruction(path: str | PathLike[str]) -> Reconstruction:
    """Load a reconstruction saved by :func:`save` or ``tomojax recon``."""
    scan = load(path)
    record = scan.source
    if record is None or record.volume is None:
        raise ValueError(f"{path} holds no reconstructed volume")
    return Reconstruction(
        volume=np.asarray(record.volume),
        scan=scan,
        grid=record.grid or scan.grid,
        method=str(record.geometry_metadata.get("reconstruction_method", "unknown")),
        info={},
    )


def save(path: str | PathLike[str], item: Scan | Reconstruction | Alignment) -> None:
    """Save a scan, reconstruction or alignment as a TomoJAX dataset (``.nxs``, ``.npz``).

    Reconstructions and alignments are saved with their scan (projections,
    geometry and any pose corrections), so the file reloads with :func:`load`
    and :func:`load_reconstruction`.
    """
    from tomojax.io import save_dataset

    if isinstance(item, Scan):
        record = _record_of(item)
    elif isinstance(item, Reconstruction):
        record = _record_of(item.scan, grid=item.grid)
        record.volume = np.asarray(item.volume)
        record.geometry_metadata["reconstruction_method"] = item.method
    elif isinstance(item, Alignment):  # pyright: ignore[reportUnnecessaryIsInstance]
        record = _record_of(item.scan)
        record.volume = np.asarray(item.volume)
    else:
        raise TypeError(f"save takes a Scan, Reconstruction or Alignment, not {type(item)}")
    save_dataset(path, record)


def _describe(geometry: Geometry) -> tuple[str, dict[str, object], Any, Geometry]:
    """Geometry type, metadata and per-view poses of ``geometry``, and its base geometry."""
    poses = None
    meta: dict[str, object] = {}
    base = geometry
    if hasattr(base, "align_params") and hasattr(base, "base"):
        poses = (np.asarray(base.align_params), str(base.translation_frame))  # pyright: ignore[reportAttributeAccessIssue]
        base = base.base  # pyright: ignore[reportAttributeAccessIssue]
    if hasattr(base, "detector_roll_deg") and hasattr(base, "base"):
        meta["detector_roll_deg"] = float(base.detector_roll_deg)  # pyright: ignore[reportAttributeAccessIssue]
        base = base.base  # pyright: ignore[reportAttributeAccessIssue]
    if isinstance(base, ConeGeometry):
        return "cone", {**meta, **base.geometry_metadata()}, poses, base
    if isinstance(base, LaminographyGeometry):
        meta |= {"tilt_deg": float(base.tilt_deg), "tilt_about": str(base.tilt_about)}
        return "lamino", meta, poses, base
    if isinstance(base, RotationAxisGeometry):
        meta["axis_unit_lab"] = [float(x) for x in base.axis_unit_lab]
        return "lamino", meta, poses, base
    if isinstance(base, ParallelGeometry):
        return "parallel", meta, poses, base
    raise TypeError(f"cannot describe a {type(base).__name__} geometry")


def _record_of(scan: Scan, *, grid: Grid | None = None) -> ProjectionDataset:
    """A dataset record for ``scan``: its geometry, keeping the source's provenance."""
    from tomojax.io import ProjectionDataset

    geometry_type, meta, poses, _ = _describe(scan.geometry)
    source = scan.source
    kept = (
        {}
        if source is None
        else {k: v for k, v in source.geometry_metadata.items() if k not in _GEOMETRY_KEYS}
    )
    fields: dict[str, Any] = {
        "projections": np.asarray(scan.projections),
        "angles_deg": scan.angles.astype(np.float32),
        "volume": None,
        "detector": scan.detector,
        "grid": scan.grid if grid is None else grid,
        "geometry_type": geometry_type,
        "geometry_metadata": {**kept, **meta},
        # Angle offsets are already in the geometry's angles.
        "angle_offset_deg": None,
        "align_params": None if poses is None else poses[0],
        "align_gauge": None if poses is None else {"pose_translation_frame": poses[1]},
        "sample_name": scan.name,
    }
    if source is None:
        return ProjectionDataset(**fields)
    return replace(source, **fields)


def _scan_from_record(record: ProjectionDataset, *, apply_alignment: bool) -> Scan:
    from tomojax.io import build_geometry_from_dataset_metadata

    _, _, geometry = build_geometry_from_dataset_metadata(
        record.geometry_inputs(), apply_saved_alignment=apply_alignment
    )
    return Scan(
        projections=record.projections,
        geometry=geometry,
        name=record.sample_name or "sample",
        source=record,
    )


def _with_grid(scan: Scan, grid: Grid) -> Scan:
    """``scan`` reconstructed on ``grid`` instead of its own."""
    if grid == scan.grid:
        return scan
    record = _record_of(scan, grid=grid)
    return replace(_scan_from_record(record, apply_alignment=True), source=scan.source)


# ----------------------------------------------------------------------------- operations


def project(
    geometry: Geometry, volume: np.ndarray | jax.Array, *, grid: Grid | None = None
) -> jax.Array:
    """Project ``volume`` through ``geometry``; see :func:`tomojax.recon.project`."""
    from tomojax.recon import project as _project

    return _project(geometry, volume, grid=grid)


def backproject(
    geometry: Geometry, projections: np.ndarray | jax.Array, *, grid: Grid | None = None
) -> jax.Array:
    """Apply the exact transpose of :func:`project`; see :func:`tomojax.recon.backproject`."""
    from tomojax.recon import backproject as _backproject

    return _backproject(geometry, projections, grid=grid)


_METHOD_OPTIONS: dict[str, frozenset[str]] = {
    "fbp": frozenset({"filter"}),
    "cgls": frozenset({"iterations"}),
    "fista": frozenset({"iterations", "tv_weight", "nonnegative", "warm_start"}),
    "spdhg": frozenset({"iterations", "tv_weight", "nonnegative", "warm_start", "seed"}),
}


def reconstruct(
    scan: Scan,
    method: Method = "fbp",
    *,
    grid: Grid | None = None,
    filter: str | None = None,
    iterations: int | None = None,
    tv_weight: float | None = None,
    nonnegative: bool | None = None,
    warm_start: bool | None = None,
    seed: int | None = None,
) -> Reconstruction:
    """Reconstruct ``scan`` with ``method``.

    ``fbp``
        Filtered backprojection: FDK for cone beams, with Parker weights for
        short scans and Wang's for offset detectors. ``filter`` is ``ramp``
        (default), ``shepp-logan`` or ``hann``. Volumes too large for the
        device are reconstructed in slabs on the host.
    ``cgls``
        Least squares by conjugate gradients, ``iterations`` (default 50).
    ``fista``
        Least squares with total-variation weight ``tv_weight`` (default
        0.005), ``iterations`` (default 50), optionally ``nonnegative`` and
        started from FBP (``warm_start``).
    ``spdhg``
        The same objective as ``fista`` by stochastic primal-dual updates,
        with a random ``seed``.

    ``grid`` reconstructs on another grid than the scan's (a region, or a
    different voxel size). An option the method does not take raises.
    """
    from tomojax.backends import default_gather_dtype
    from tomojax.geometry.api import detector_grid_from_geometry_inputs
    from tomojax.recon.api import (
        ReconstructionAlgorithmOptions,
        ReconstructionAlgorithmRequest,
        default_views_per_batch,
        run_reconstruction_algorithm,
    )

    if method not in METHODS:
        raise ValueError(f"reconstruct: method must be one of {', '.join(METHODS)}; got {method!r}")
    given = {
        name: value
        for name, value in {
            "filter": filter,
            "iterations": iterations,
            "tv_weight": tv_weight,
            "nonnegative": nonnegative,
            "warm_start": warm_start,
            "seed": seed,
        }.items()
        if value is not None
    }
    unused = sorted(set(given) - _METHOD_OPTIONS[method])
    if unused:
        takes = ", ".join(sorted(_METHOD_OPTIONS[method]))
        raise ValueError(
            f"reconstruct: method {method!r} does not take {', '.join(unused)} (it takes {takes})"
        )
    if grid is not None:
        scan = _with_grid(scan, grid)
    options = ReconstructionAlgorithmOptions(
        algorithm=method,
        filter_name=str(given.get("filter", "ramp")),
        iters=int(given.get("iterations", 50)),
        lambda_tv=float(given.get("tv_weight", 0.005)),
        positivity=bool(given.get("nonnegative", False)),
        spdhg_seed=int(given.get("seed", 0)),
        warm_start="fbp" if given.get("warm_start") else "none",
    )
    request = ReconstructionAlgorithmRequest(
        options=options,
        geometry=scan.geometry,
        grid=scan.grid,
        detector=scan.detector,
        projections=scan.projections,
        detector_grid=detector_grid_from_geometry_inputs(scan.detector, scan.geometry),
        volume_mask=None,
        views_per_batch=default_views_per_batch(method),
        views_per_batch_mode="default",
        gather_dtype=default_gather_dtype(),
    )
    result = run_reconstruction_algorithm(request)
    return Reconstruction(
        volume=result.volume,
        scan=scan,
        grid=scan.grid,
        method=method,
        info=dict(result.algorithm_config),
    )


def align(
    scan: Scan,
    *,
    mode: str = "pose",
    quality: str = "fast",
    levels: Iterable[int] | None = None,
    freeze: Iterable[str] = (),
    config: AlignConfig | None = None,
) -> Alignment:
    """Estimate ``scan``'s geometry corrections and reconstruct with them.

    ``mode`` is ``pose`` (per-view motion, the default), ``cor`` (detector
    centre, or a cone beam's axis offset and detector roll), ``cor-then-pose``
    or ``full`` (setup geometry, then motion, coarse to fine); see
    :mod:`tomojax.alignment`. ``quality`` is ``fast`` or ``reference`` (slower,
    more conservative). ``freeze`` keeps named parameters fixed (for example
    ``("dy",)``); ``levels`` sets the coarse-to-fine binning factors.
    ``config`` (a :class:`tomojax.alignment.AlignConfig`) holds expert solver
    settings and replaces the quality profile's defaults.

    The returned :attr:`Alignment.scan` carries the corrections, so
    ``tomojax.reconstruct(result.scan)`` reconstructs with them.
    """
    import jax.numpy as jnp

    from tomojax.alignment.api import align_multires, alignment_plan, cone_setup, pad_pose_params

    if scan.poses is not None:
        raise ValueError(
            "align: the scan already carries pose corrections; align its nominal geometry "
            "(load it with apply_alignment=False)"
        )
    plan = alignment_plan(
        mode, scan.grid, quality=quality, levels=levels, freeze=freeze, config=config
    )
    geometry, grid, detector = scan.geometry, scan.grid, scan.detector
    projections = jnp.asarray(scan.projections, jnp.float32)
    cfg: AlignConfig | None = plan.config
    record: dict[str, object] = {"mode": plan.mode, "levels": list(plan.levels)}
    setup = cone_setup(geometry, grid, detector, projections, plan.config)
    if setup is not None:
        geometry, cfg = setup.geometry, setup.config
        record["cone_axis_calibration"] = {
            "axis_offset": setup.axis_offset,
            "detector_roll_deg": setup.detector_roll_deg,
            "heights": list(setup.heights),
            "slab_offsets": list(setup.slab_offsets),
        }
    calibrated = replace(scan, geometry=geometry)
    if cfg is None:
        # Only the cone axis was asked for: reconstruct with it.
        volume = reconstruct(calibrated).volume
        poses = np.zeros((len(scan.angles), 6), np.float32)
        info: dict[str, object] = dict(record)
        frame = "detector"
    else:
        volume, params, run = align_multires(
            geometry, grid, detector, projections, factors=plan.levels, config=cfg
        )
        poses = np.asarray(pad_pose_params(params), np.float32)
        info = {**dict(run), **record}
        frame = cfg.pose_translation_frame
    corrected = _record_of(calibrated)
    calibration = info.get("geometry_calibration_state")
    if isinstance(calibration, dict):
        from tomojax.geometry.api import build_calibrated_geometry_metadata_patch

        patch = build_calibrated_geometry_metadata_patch(
            calibration_state=cast("dict[str, object]", calibration),
            detector=detector.to_dict(),
            geometry_meta=corrected.geometry_metadata,
        )
        corrected.detector = _detector(patch["detector"])
        corrected.geometry_metadata = {
            **corrected.geometry_metadata,
            **{k: v for k, v in dict(patch["geometry_meta"]).items() if k in _GEOMETRY_KEYS},
        }
    corrected.align_params = poses
    corrected.align_gauge = {"pose_translation_frame": frame}
    aligned = _scan_from_record(corrected, apply_alignment=True)
    return Alignment(
        scan=replace(aligned, source=scan.source), volume=volume, poses=poses, info=info
    )


def _detector(value: object) -> Detector:
    from tomojax.geometry import Detector

    if isinstance(value, Detector):
        return value
    data = cast("dict[str, Any]", value)
    return Detector(
        nu=int(data["nu"]),
        nv=int(data["nv"]),
        du=float(data["du"]),
        dv=float(data["dv"]),
        det_center=tuple(float(x) for x in data.get("det_center", (0.0, 0.0))),  # type: ignore[arg-type]
    )


__all__ = [
    "METHODS",
    "Alignment",
    "Reconstruction",
    "Scan",
    "align",
    "backproject",
    "load",
    "load_reconstruction",
    "project",
    "reconstruct",
    "save",
]
