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

from dataclasses import dataclass, is_dataclass, replace
import json
from typing import TYPE_CHECKING, Any, Literal, cast
import warnings

import numpy as np

from tomojax._loading import load
from tomojax._scan import (
    GEOMETRY_KEYS,
    Scan,
    describe,
    record_of,
    scan_from_record,
    with_grid,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from os import PathLike

    import jax

    from tomojax._typed_arrays import Device
    from tomojax.alignment import AlignConfig
    from tomojax.alignment.api import (
        AlignmentMode,
        AlignmentPlan,
        PoseTranslationFrame,
        QualityTier,
    )
    from tomojax.geometry import Detector, Geometry, Grid, ScanGeometry
    from tomojax.recon.api import MethodConfig

type Method = Literal["fbp", "cgls", "fista", "spdhg"]


@dataclass(frozen=True)
class Reconstruction:
    """A reconstructed volume, the scan it came from and how it was made.

    ``volume`` is ``(nx, ny, nz)`` on :attr:`grid`, in the reciprocal of the
    geometry's length unit (a NumPy or JAX array; ``np.asarray`` gives NumPy).
    ``info`` records the method's settings.
    """

    volume: np.ndarray | jax.Array
    scan: Scan
    method: str
    info: Mapping[str, object]

    @property
    def grid(self) -> Grid:
        """The grid the volume is on, the scan's."""
        return self.scan.grid


@dataclass(frozen=True)
class Alignment:
    """The result of :func:`align`.

    ``volume`` is the reconstruction the alignment converged with. ``scan`` is
    the input scan with the estimated geometry and per-view poses applied:
    reconstruct it, align it again or save it. ``poses`` are its ``(views, 6)``
    poses (see :attr:`Scan.poses`), always a table, zero where nothing moved;
    ``info`` is the solver's record.
    """

    volume: np.ndarray | jax.Array
    scan: Scan
    poses: np.ndarray
    info: Mapping[str, object]


# ----------------------------------------------------------------------------- files


def load_reconstruction(path: str | PathLike[str]) -> Reconstruction:
    """Load a reconstruction saved by :func:`save` or ``tomojax recon``."""
    scan = load(path)
    record = scan.source
    if record is None or record.volume is None:
        raise ValueError(f"{path} holds no reconstructed volume")
    meta = record.geometry_metadata
    info = meta.get("reconstruction_info")
    return Reconstruction(
        volume=np.asarray(record.volume),
        scan=scan,
        method=str(meta.get("reconstruction_method", "unknown")),
        info=dict(info) if isinstance(info, dict) else {},
    )


def save(path: str | PathLike[str], item: Scan | Reconstruction | Alignment) -> None:
    """Save a scan, reconstruction or alignment as a TomoJAX dataset (``.nxs``, ``.npz``).

    Reconstructions and alignments are saved with their scan (projections,
    geometry and any pose corrections), so the file reloads with :func:`load`
    and :func:`load_reconstruction`.
    """
    from tomojax.io import save_dataset

    if isinstance(item, Scan):
        record = record_of(item)
    elif isinstance(item, Reconstruction):
        record = record_of(item.scan)
        record.volume = np.asarray(item.volume)
        record.geometry_metadata["reconstruction_method"] = item.method
        # As JSON, which the file's metadata is: a configuration as its fields, and
        # what else JSON cannot hold as text.
        record.geometry_metadata["reconstruction_info"] = json.loads(
            json.dumps(dict(item.info), default=_json_value)
        )
    elif isinstance(item, Alignment):  # pyright: ignore[reportUnnecessaryIsInstance]
        record = record_of(item.scan)
        record.volume = np.asarray(item.volume)
    else:
        raise TypeError(f"save takes a Scan, Reconstruction or Alignment, not {type(item)}")
    save_dataset(path, record)


def _json_value(value: object) -> object:
    from dataclasses import fields

    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: getattr(value, item.name) for item in fields(value)}
    return str(value)


# ----------------------------------------------------------------------------- operations


def project(
    geometry: Geometry,
    volume: np.ndarray | jax.Array,
    *,
    grid: Grid | None = None,
    devices: Device | Sequence[Device] | None = None,
) -> jax.Array:
    """Project ``volume`` through ``geometry``; see :func:`tomojax.recon.project`."""
    from tomojax.recon import project as _project

    return _project(geometry, volume, grid=grid, devices=devices)


def backproject(
    geometry: Geometry,
    projections: np.ndarray | jax.Array,
    *,
    grid: Grid | None = None,
    devices: Device | Sequence[Device] | None = None,
) -> jax.Array:
    """Apply the exact transpose of :func:`project`; see :func:`tomojax.recon.backproject`."""
    from tomojax.recon import backproject as _backproject

    return _backproject(geometry, projections, grid=grid, devices=devices)


def reconstruct(
    scan: Scan,
    method: Method = "fbp",
    *,
    grid: Grid | None = None,
    config: MethodConfig | None = None,
    filter: str | None = None,
    iterations: int | None = None,
    tv_weight: float | None = None,
    nonnegative: bool | None = None,
    warm_start: bool | None = None,
    seed: int | None = None,
    devices: Device | Sequence[Device] | None = None,
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
        0.005), ``iterations`` (default 50), optionally ``nonnegative``.
    ``spdhg``
        The same objective as ``fista`` by stochastic primal-dual updates:
        ``iterations`` (default 400) each update one block of views, in an
        order drawn with ``seed``.

    ``warm_start`` starts ``cgls``, ``fista`` and ``spdhg`` from the FBP
    reconstruction. ``grid`` reconstructs on another grid than the scan's (a
    region, or a different voxel size).

    ``config`` holds the method's expert settings: a
    :class:`~tomojax.recon.FBPConfig` (for FDK too),
    :class:`~tomojax.recon.CGLSConfig`, :class:`~tomojax.recon.FistaConfig` or
    :class:`~tomojax.recon.SPDHGConfig`; without one, the class's defaults
    apply. A config of another method's class raises :class:`ValueError`. The
    keywords are fields of that class, and each one given replaces its field,
    as :func:`dataclasses.replace` does, so a keyword wins over the config. A
    keyword the method's class has no field for raises.

    ``devices`` (``fbp``, ``cgls`` and ``fista``; one or several,
    ``jax.devices()`` for all) shares the views among them: each projects its
    own views and holds the whole volume, and their backprojections are
    summed. The volume, on the first device, is the one-device result up to
    the order of that sum.

    :attr:`Reconstruction.info` holds the resolved ``config`` with the
    solver's record (iterations run, termination, losses).
    """
    from tomojax.recon.api import method_config, reconstruct_arrays

    given = {
        name: value
        for name, value in {
            "filter": filter,
            "iterations": iterations,
            "tv_weight": tv_weight,
            "nonnegative": nonnegative,
            "seed": seed,
            "devices": devices,
        }.items()
        if value is not None
    }
    cfg = method_config(method, config=config, **given)
    if warm_start is not None and method == "fbp":
        raise ValueError("method 'fbp' does not take warm_start")
    if grid is not None:
        scan = with_grid(scan, grid)
    if method != "fbp":
        _suggest_binning(scan, "iterative reconstruction")
    volume, info = reconstruct_arrays(
        method,
        scan.geometry,
        scan.grid,
        scan.detector,
        scan.projections,
        config=cfg,
        warm_start=bool(warm_start),
    )
    return Reconstruction(volume=volume, scan=scan, method=method, info=info)


def align(
    scan: Scan,
    *,
    mode: AlignmentMode = "pose",
    quality: QualityTier = "fast",
    levels: Iterable[int] | None = None,
    freeze: Iterable[str] = (),
    grid: Grid | None = None,
    checkpoint: str | PathLike[str] | None = None,
    config: AlignConfig | None = None,
    devices: Device | Sequence[Device] | None = None,
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

    ``grid`` aligns and reconstructs on another grid than the scan's (a
    region, or a different voxel size), as :func:`reconstruct` does.

    ``checkpoint`` is a file the alignment saves its progress to after each
    outer iteration. When that file holds a checkpoint of the same alignment
    (the same projections and geometry, mode, levels, grid and settings, from
    the same TomoJAX version), the alignment resumes from it, and a finished
    one returns its result at once. Any other file there raises
    :class:`ValueError` saying how it differs, and is left as it is. A cone
    beam's ``cor`` mode runs no outer iterations and writes none.

    ``devices`` (one or several, ``jax.devices()`` for all) shares the views
    among them, as :func:`reconstruct` does, in the joint pose and volume
    update of ``pose`` alignment and in cone-beam reconstruction steps; the
    rest runs on the first. The devices change only the order of sums, so a
    checkpoint made on some resumes on others.

    A scan that already carries poses (an ASTRA import's, say, or an earlier
    alignment's) is corrected on top of them, and multi-orbit
    :class:`~tomojax.geometry.ConeSegments` scans are aligned as one, so their
    orbits come into register; only ``pose`` alignment takes either.

    The returned :attr:`Alignment.scan` carries the corrections, and the grid,
    so ``tomojax.reconstruct(result.scan)`` reconstructs with them;
    :attr:`Alignment.poses` are the scan's poses, its own included.
    :attr:`Alignment.info` holds the mode, levels and resolved ``config`` with
    the solver's record (losses, gauge, any calibrated setup geometry).
    """
    from tomojax.alignment.api import alignment_plan
    from tomojax.geometry import ConeSegments

    if grid is not None:
        scan = with_grid(scan, grid)
    plan = alignment_plan(
        mode, scan.grid, quality=quality, levels=levels, freeze=freeze, config=config
    )
    segmented = isinstance(scan.geometry, ConeSegments)
    if plan.mode != "pose" and (scan.poses is not None or segmented):
        raise ValueError(
            f"align: mode {plan.mode!r} calibrates one arrangement without pose corrections; "
            "align posed or segmented scans with mode='pose'"
        )
    _suggest_binning(scan, "alignment")
    from tomojax.core.devices import sharing

    with sharing(devices):
        geometry, volume, poses, frame, info = _run_alignment(scan, plan, checkpoint)
    corrected = record_of(replace(scan, geometry=cast("ScanGeometry", geometry)))
    calibration = info.get("geometry_calibration_state")
    if isinstance(calibration, dict):
        from tomojax.geometry.api import build_calibrated_geometry_metadata_patch

        patch = build_calibrated_geometry_metadata_patch(
            calibration_state=cast("dict[str, object]", calibration),
            detector=scan.detector.to_dict(),
            geometry_meta=corrected.geometry_metadata,
        )
        corrected.detector = _detector(patch["detector"])
        corrected.geometry_metadata = {
            **corrected.geometry_metadata,
            **{k: v for k, v in dict(patch["geometry_meta"]).items() if k in GEOMETRY_KEYS},
        }
    corrected.align_params = poses
    corrected.align_gauge = {"pose_translation_frame": frame}
    aligned = scan_from_record(corrected, poses=True)
    return Alignment(
        volume=volume, scan=replace(aligned, source=scan.source), poses=poses, info=info
    )


def _run_alignment(
    scan: Scan, plan: AlignmentPlan, checkpoint: str | PathLike[str] | None
) -> tuple[Geometry, np.ndarray | jax.Array, np.ndarray, str, dict[str, object]]:
    """Calibrated geometry, volume, ``(views, 6)`` poses, their frame and record of ``plan``."""
    import jax.numpy as jnp

    from tomojax.alignment.api import (
        AlignmentRun,
        align_multires,
        alignment_checkpointing,
        cone_setup,
        pad_pose_params,
    )

    geometry, grid, detector = scan.geometry, scan.grid, scan.detector
    projections = jnp.asarray(scan.projections, jnp.float32)
    cfg: AlignConfig | None = plan.config
    record: dict[str, object] = {"mode": plan.mode, "levels": list(plan.levels)}
    record["config"] = plan.config
    setup = cone_setup(geometry, grid, detector, projections, plan.config)
    if setup is not None:
        geometry, cfg = setup.geometry, setup.config
        record["cone_axis_calibration"] = setup.to_dict()
    if cfg is None:
        # Only the cone axis was asked for: reconstruct with it.
        volume = reconstruct(replace(scan, geometry=cast("ScanGeometry", geometry))).volume
        return geometry, volume, np.zeros((len(scan.angles), 6), np.float32), "detector", record
    run = AlignmentRun(
        projections=projections,
        geometry=scan.geometry,
        config=plan.config,
        mode=plan.mode,
        levels=plan.levels,
    )
    resume, write = alignment_checkpointing(checkpoint, run)
    volume, params, info = align_multires(
        geometry,
        grid,
        detector,
        projections,
        factors=plan.levels,
        config=cfg,
        resume_state=resume,
        checkpoint_callback=write,
    )
    poses = np.asarray(pad_pose_params(params), np.float32)
    frame = cfg.pose_translation_frame
    if plan.mode == "pose" and describe(geometry)[0] != "cone":
        record["implied_detector_u_px"] = _implied_detector_u_px(geometry, detector, poses, frame)
    if scan.poses is not None:
        from tomojax._data.geometry_meta import composed_poses

        poses, frame = composed_poses(scan.geometry, poses, frame), "detector"
    return geometry, volume, poses, frame, {**dict(info), **record}


def _implied_detector_u_px(
    geometry: Geometry, detector: Detector, poses: np.ndarray, frame: PoseTranslationFrame
) -> float | None:
    """The detector-u (centre-of-rotation) offset ``poses`` hold, in detector pixels."""
    import jax.numpy as jnp

    from tomojax.alignment.api import apply_pose_updates, implied_detector_offset
    from tomojax.geometry import stack_view_poses

    if not np.any(poses[:, 3]):
        return None
    nominal = stack_view_poses(geometry, len(poses))
    aligned = apply_pose_updates(nominal, jnp.asarray(poses), translation_frame=frame)
    offset, _ = implied_detector_offset(np.asarray(nominal), np.asarray(aligned))
    return offset / float(detector.du)


def _oversampling(scan: Scan) -> float:
    """How many times finer than the voxels the detector samples the rotation axis.

    The lesser of the row and column ratios: a cone beam's pixel pitch divided
    by its magnification (the least magnified segment's), a parallel beam's
    pitch itself, against the voxel. A tilted axis mixes the grid's axes into
    both, so laminography compares with the smallest voxel side.
    """
    from tomojax.geometry import ConeSegments, beam_of

    detector, grid, geometry = scan.detector, scan.grid, scan.geometry
    parts = geometry.segments if isinstance(geometry, ConeSegments) else (geometry,)
    beams = [beam_of(part) for part in parts]
    magnification = min(1.0 if beam is None else float(beam.magnification) for beam in beams)
    if describe(geometry)[0] == "lamino":
        across = along = min(grid.vx, grid.vy, grid.vz)
    else:
        across, along = min(grid.vx, grid.vy), grid.vz
    du, dv = float(detector.du), float(detector.dv)
    return min(across * magnification / du, along * magnification / dv)


def _suggest_binning(scan: Scan, work: str) -> None:
    """Warn when binning the detector would cut ``work``'s cost for little detail."""
    ratio = _oversampling(scan)
    if ratio < 2:
        return
    factor = int(ratio)
    warnings.warn(
        f"the detector samples the rotation axis {ratio:.1f} times more finely than the "
        f"grid's voxels: scan.binned({factor}) makes {work} up to {factor * factor} times "
        "cheaper for little loss of detail",
        stacklevel=3,
    )


def _detector(value: object) -> Detector:
    from tomojax.geometry import Detector

    if isinstance(value, Detector):
        return value
    return Detector.from_dict(cast("dict[str, Any]", value))


__all__ = [
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
