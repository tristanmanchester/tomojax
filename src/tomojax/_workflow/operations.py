"""TomoJAX's workflow API: scans, reconstructions and alignments.

A :class:`Scan` is projections together with the geometry that produced them;
every operation takes one and returns a result that carries its geometry, so
results can be reconstructed, aligned again or saved without restating it::

    import tomojax as tj

    scan = tj.load("scan.nxs")
    recon = tj.reconstruct(scan)  # Geometry-aware automatic reconstruction
    result = tj.align(scan)
    tj.save("recon.nxs", tj.reconstruct(result.scan, method="cgls"))
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, closing, contextmanager
from dataclasses import is_dataclass, replace
import inspect
import json
import logging
import os
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from tomojax import _typed_arrays, progress, recon
from tomojax._loading.loader import Frames, load
from tomojax._parameters import (
    FLAT_WEIGHTS,
    AlignMethod,
    AnnotationModule,
    Filter,
    Fit,
    FlatWeighting,
    Method,
    RegionName,
    flat_weighting,
    method_fit,
)
from tomojax._scan.model import (
    Alignment,
    Reconstruction,
    Scan,
    bind_step_geometry,
    cropped_record,
    geometry_of,
    geometry_record,
    rebuild,
    record_of,
    source_of,
    with_grid,
)
from tomojax._scan.simulation import Simulation
from tomojax._typed_arrays import Device
from tomojax.alignment import AlignConfig
from tomojax.core.registration import SlabPlanes, recorded_length_unit, registration
from tomojax.errors import explain_cuda_failure
from tomojax.geometry import Grid, ScanGeometry
from tomojax.progress import warn
from tomojax.recon.api import MethodConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    import jax

    from tomojax import corrections
    from tomojax._workflow import region as _region
    from tomojax._workflow.planning import ResolvedReconstruction
    from tomojax._workflow.smooth_alignment import CenterSelection, JointSelection
    from tomojax.core.json_values import JsonValue
    from tomojax.corrections.api import CorrectedRows
    from tomojax.geometry import Detector
    from tomojax.io import ProjectionDataset
    from tomojax.recon.types import ReconstructionSlab
else:
    corrections = AnnotationModule("tomojax.corrections")
    _region = AnnotationModule("tomojax._workflow.region")

type Region = RegionName | _region.RegionEstimate
LOG = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- files


def load_reconstruction(path: str | PathLike[str]) -> Reconstruction:
    """Load a reconstruction or alignment image saved by :func:`save` or ``tomojax`` commands.

    The reconstruction has its scan when the file holds the projections (saved with
    ``include_scan=True``, or an alignment). When loading an alignment file, the
    returned reconstruction's :attr:`~Reconstruction.grid` is the working image's
    grid while ``scan.grid`` remains the scan's own grid; use
    :func:`load_alignment` to reload an :class:`Alignment` directly. A file saved
    without projections gives a reconstruction whose ``scan`` is None, and says so;
    ``info["source"]`` names the projections it came from. Nothing looks for them.

    Parameters
    ----------
    path : str or path-like
        Saved reconstruction dataset.

    Returns
    -------
    Reconstruction
        Volume, geometry, stored settings and, when saved with them, the scan.

    Raises
    ------
    ValueError
        If the dataset has no reconstructed volume.
    FileNotFoundError
        If the input does not exist.

    Examples
    --------
    >>> import tomojax as tj
    >>> recon = tj.load_reconstruction("recon.nxs")  # doctest: +SKIP
    """
    from tomojax.io import load_dataset

    return reconstruction_from_record(path, load_dataset(os.fspath(path)), load)


def reconstruction_from_record(
    path: str | PathLike[str],
    record: ProjectionDataset,
    read_scan: Callable[[str | PathLike[str]], Scan],
) -> Reconstruction:
    from tomojax._loading.files import saved_volume_grid

    if record.volume is None:
        raise ValueError(f"{path} holds no reconstructed volume")
    meta = record.geometry_metadata
    if meta.get("volume_role") == "phantom":
        raise ValueError(
            f"{path} holds a simulation phantom, not a reconstruction; "
            "load it with tomojax.load_simulation"
        )
    scan = None if record.projections is None else read_scan(path)
    if scan is None:
        warn(
            f"{path} holds the reconstruction without its scan (saved without "
            "include_scan=True): its scan is None, and info['source'] names the projections",
            stacklevel=2,
        )
    geometry = geometry_of(record, apply_poses=True) if scan is None else scan.geometry
    info = meta.get("reconstruction_info", meta.get("alignment_info"))
    aligned = "reconstruction_method" not in meta and "alignment_info" in meta
    if aligned:
        volume_grid = saved_volume_grid(
            None if record.grid is None else record.grid.to_dict(),
            meta,
        )
        if volume_grid is not None:
            geometry = rebuild(geometry, grid=volume_grid)
    return Reconstruction(
        volume=np.asarray(record.volume),
        geometry=geometry,
        method="align" if aligned else str(meta.get("reconstruction_method", "unknown")),
        info=dict(info) if isinstance(info, dict) else {},
        scan=scan,
        length_unit=recorded_length_unit(meta),
        source_path=str(Path(path).resolve()),
    )


def load_simulation(path: str | PathLike[str]) -> Simulation:
    """Load a :class:`Simulation` saved by :func:`save` or ``tomojax simulate``.

    Parameters
    ----------
    path : str or path-like
        Saved simulation dataset.

    Returns
    -------
    Simulation
        The simulated scan, ground-truth phantom ``(nz, ny, nx)``, and any true poses.

    Raises
    ------
    ValueError
        If the dataset holds no simulation phantom.
    FileNotFoundError
        If the input does not exist.

    Examples
    --------
    >>> import tomojax as tj
    >>> scan, phantom = tj.load_simulation("synthetic.nxs")  # doctest: +SKIP
    """
    from tomojax.io import load_dataset

    return simulation_from_record(path, load_dataset(os.fspath(path)), load)


def simulation_from_record(
    path: str | PathLike[str],
    record: ProjectionDataset,
    read_scan: Callable[[str | PathLike[str]], Scan],
) -> Simulation:
    if record.volume is None or record.geometry_metadata.get("volume_role") != "phantom":
        raise ValueError(f"{path} holds no simulation phantom")
    return Simulation(read_scan(path), np.asarray(record.volume))


def load_alignment(path: str | PathLike[str]) -> Alignment:
    """Load an :class:`Alignment` saved by :func:`save` or ``tomojax align``.

    Preserves the scan's own grid on ``result.scan.grid`` separately from the
    working image's grid on ``result.grid``, and supports image-free center
    alignments such as axial ``fit="center"`` and ``method="opposite-views"``
    (leaving ``image`` and ``grid`` None).

    Parameters
    ----------
    path : str or path-like
        Saved alignment dataset.

    Returns
    -------
    Alignment
        Aligned scan, stored alignment record and optional working image and grid.

    Raises
    ------
    ValueError
        If the dataset holds no alignment record.
    FileNotFoundError
        If the input does not exist.

    Examples
    --------
    >>> import tomojax as tj
    >>> result = tj.load_alignment("aligned.nxs")  # doctest: +SKIP
    """
    from tomojax.io import load_dataset

    return alignment_from_record(path, load_dataset(os.fspath(path)), load)


def alignment_from_record(
    path: str | PathLike[str],
    record: ProjectionDataset,
    read_scan: Callable[[str | PathLike[str]], Scan],
) -> Alignment:
    meta = record.geometry_metadata
    raw_info = meta.get("alignment_info")
    if not isinstance(raw_info, dict):
        raise ValueError(f"{path} holds no alignment record")
    scan = read_scan(path)
    info = dict(raw_info)
    raw_working_grid = info.pop("working_grid", None)
    image = None if record.volume is None else np.asarray(record.volume)
    if image is None:
        working_grid: Grid | None = None
    elif isinstance(raw_working_grid, dict):
        working_grid = Grid.from_dict(raw_working_grid)
    else:
        working_grid = scan.grid
    return Alignment(scan=scan, info=info, image=image, grid=working_grid)


def validate_save(
    path: Path, item: Scan | Reconstruction | Alignment | Simulation, *, include_scan: bool
) -> None:
    """Check the destination format and availability of the requested scan before writing."""
    from tomojax.errors import InvalidParameter
    from tomojax.io.api import dataset_output_format

    dataset_output_format(path)
    if isinstance(item, Reconstruction) and item.scan is None and include_scan:
        raise InvalidParameter(
            "include_scan", "this reconstruction was loaded without its scan: there is none to save"
        )


def save(
    path: str | PathLike[str],
    item: Scan | Reconstruction | Alignment | Simulation,
    *,
    include_scan: bool = False,
) -> None:
    """Save a scan, reconstruction, alignment or simulation as a TomoJAX dataset.

    A reconstruction is saved with its volume, geometry and full record, and names
    the projections it came from (``info["source"]``: their path, shape, size and a
    hash of three views); ``include_scan=True`` saves the scan too, so the file also
    reloads with :func:`load`. An alignment is saved as its corrected scan with its
    poses and, for joint fits, its working image. A simulation is saved as its scan
    together with its ground-truth phantom as the stored volume.

    Parameters
    ----------
    path : str or path-like
        Output dataset path; an existing file is overwritten.
    item : Scan, Reconstruction, Alignment or Simulation
        Scan or result to save with its geometry and metadata.
    include_scan : bool, optional
        Also save a reconstruction's scan. Default is False.

    Returns
    -------
    None
        Write the dataset to disk.

    Raises
    ------
    TypeError
        If item is not a scan, reconstruction, alignment or simulation.
    ValueError
        If the output format is unsupported, or ``include_scan`` is asked of a
        reconstruction loaded without its scan.

    Examples
    --------
    >>> import tomojax as tj
    >>> scan = tj.load("scan.nxs")  # doctest: +SKIP
    >>> tj.save("copy.nxs", scan)  # doctest: +SKIP
    """
    from tomojax._scan.simulation import Simulation
    from tomojax.io import save_dataset
    from tomojax.io.api import replace_dataset

    with progress.task("save", Path(path).name, None, ""):
        validate_save(Path(path), item, include_scan=include_scan)
        if isinstance(item, Scan):
            record = record_of(item)
        elif isinstance(item, Simulation):
            record = record_of(item.scan)
            record.volume = item.phantom
            record.geometry_metadata["volume_role"] = "phantom"
            record.geometry_metadata.pop("reconstruction_method", None)
            record.geometry_metadata.pop("reconstruction_info", None)
        elif isinstance(item, Reconstruction):
            if item.scan is None:
                record = geometry_record(item.geometry)
                source = item.info.get("source")
            else:
                record = record_of(item.scan, grid=item.grid)
                source = source_of(item.scan)
                if not include_scan:
                    record = replace_dataset(record, projections=None)
            if item.length_unit is not None:
                record.geometry_metadata["length_unit"] = item.length_unit
            record.volume = item.volume
            record.geometry_metadata["reconstruction_method"] = item.method
            record.geometry_metadata["reconstruction_info"] = saved_info(
                {**item.info, "source": source}
            )
        elif isinstance(item, Alignment):  # pyright: ignore[reportUnnecessaryIsInstance]
            record = record_of(item.scan)
            record.volume = item.image
            # The alignment made this volume: an earlier reconstruction's method
            # and settings no longer describe it.
            record.geometry_metadata.pop("reconstruction_method", None)
            record.geometry_metadata.pop("reconstruction_info", None)
            # The image's grid, which need not be the scan's: load_alignment/export read it back.
            info = {**item.info} if item.grid is None else {**item.info, "working_grid": item.grid}
            record.geometry_metadata["alignment_info"] = saved_info(info)
        else:
            raise TypeError(
                f"save takes a Scan, Reconstruction, Alignment or Simulation, not {type(item)}"
            )
        if not isinstance(item, Simulation):  # a scan loaded from a simulation carries its tag
            _ = record.geometry_metadata.pop("volume_role", None)
        save_dataset(path, record)


def saved_info(info: dict[str, object]) -> dict[str, JsonValue]:
    """Convert result metadata to JSON, recording dataclass settings by field."""
    return cast("dict[str, JsonValue]", json.loads(json.dumps(info, default=_json_value)))


def _json_value(value: object) -> object:
    from dataclasses import fields

    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: getattr(value, item.name) for item in fields(value)}
    return str(value)


# ----------------------------------------------------------------------------- operations


def project(
    geometry: ScanGeometry,
    volume: np.ndarray | _typed_arrays.JaxArray,
    *,
    grid: Grid | None = None,
    devices: Device | Sequence[Device] | None = None,
) -> _typed_arrays.JaxArray:
    """Project a volume through the supplied geometry.

    Parameters
    ----------
    geometry : ScanGeometry
        Per-view geometry defining the matched projector.
    volume : numpy.ndarray or jax.Array
        Volume in (z, y, x) order.
    grid : Grid or None, optional
        Voxel grid; None uses the geometry's grid.
    devices : jax.Device or sequence of jax.Device or None, optional
        Devices sharing the views; None uses the current default device.

    Returns
    -------
    jax.Array
        Line-integral projections, shape (views, detector.nv, detector.nu);
        single-slice 2D input returns (views, detector.nu).

    Raises
    ------
    ValueError
        If dimensions or geometry metadata do not match the operation.

    Examples
    --------
    >>> import tomojax as tj
    >>> scan = tj.load("scan.nxs")  # doctest: +SKIP
    >>> recon = tj.reconstruct(scan)  # doctest: +SKIP
    >>> projections = tj.project(recon.geometry, recon.volume)  # doctest: +SKIP
    """
    from tomojax.recon.api import project as _project

    with progress.task("project", "project", None, ""):
        return _project(geometry, volume, grid=grid, devices=devices)


def backproject(
    geometry: ScanGeometry,
    projections: np.ndarray | _typed_arrays.JaxArray,
    *,
    grid: Grid | None = None,
    devices: Device | Sequence[Device] | None = None,
) -> _typed_arrays.JaxArray:
    """Apply the exact transpose of the forward projector.

    Parameters
    ----------
    geometry : ScanGeometry
        Per-view geometry defining the matched projector.
    projections : numpy.ndarray or jax.Array
        Line integrals in (view, v, u) order, or (view, u) for a single slice.
    grid : Grid or None, optional
        Voxel grid; None uses the geometry's grid.
    devices : jax.Device or sequence of jax.Device or None, optional
        Devices sharing the views; None uses the current default device.

    Returns
    -------
    jax.Array
        Adjoint volume, shape (grid.nz, grid.ny, grid.nx), or (grid.ny, grid.nx)
        for a single-slice 2D input.

    Raises
    ------
    ValueError
        If dimensions or geometry metadata do not match the operation.

    Examples
    --------
    >>> import tomojax as tj
    >>> scan = tj.load("scan.nxs")  # doctest: +SKIP
    >>> volume = tj.backproject(scan.geometry, scan.projections)  # doctest: +SKIP
    """
    from tomojax.recon.api import backproject as _backproject

    with progress.task("backproject", "backproject", None, ""):
        return _backproject(geometry, projections, grid=grid, devices=devices)


def reconstruct(
    scan: Scan,
    method: Method = "auto",
    *,
    grid: Grid | None = None,
    region: Region | None = None,
    out: np.ndarray | None = None,
    config: MethodConfig | None = None,
    filter: Filter | None = None,
    iterations: int | None = None,
    tv_weight: float | None = None,
    nonnegative: bool | None = None,
    warm_start: bool | None = None,
    seed: int | None = None,
    devices: Device | Sequence[Device] | None = None,
    checkpoint: str | PathLike[str] | None = None,
) -> Reconstruction:
    """Reconstruct ``scan`` with ``method``.

    ``auto`` (default) chooses Fourier inversion for unposed axial half-turn
    scans, matched CGLS for segmented cone geometry and for cone arcs FDK cannot
    weight (under 180 degrees plus the fan angle), and FBP/FDK otherwise,
    laminography and tilted-axis scans included: for those, ``fista`` gives a
    better image at the cost of its iterations (https://tomojax.dev/measurements/). It
    reconstructs with the geometry and poses the scan carries, on its grid or
    the ``grid`` given; it fits no poses (use :func:`align`).
    It estimates nothing: a center comes from the file or from :func:`align`.
    ``info["automatic"]`` records those choices and limitations. Explicit
    methods bypass this policy.

    ``fbp``
        Filtered backprojection: FDK for cone beams, with Parker weights for
        short scans and Wang's for offset detectors. ``filter`` is ``ramp``
        (default), ``shepp-logan`` or ``hann``. Volumes too large for the
        device are reconstructed in slabs on the host.
    ``fourier``
        Fourier-slice inversion of parallel-beam half-turn scans, with
        nonuniform or repeated angles and translation-only poses; no cone beams
        or tilted axes. Omitting ``filter`` (default) keeps the sharper,
        noisier native response; ``filter="ramp"``, ``"shepp-logan"`` or
        ``"hann"`` matches FBP's detector response at Fourier speed. Input and
        output stay on the host; :class:`~tomojax.recon.FourierConfig` selects
        NumPy or CuPy CUDA.
    ``cgls``
        Least squares by conjugate gradients. With no budget, tolerances, or
        penalties, stops when the projection residual RMS reaches the estimated
        detector noise and sampling error, capped at 50 updates (numerical
        convergence, roundoff, and breakdown checks also apply). ``info``
        reports ``termination``, ``early_stop``, ``effective_iterations``,
        ``measurement_noise_rms``, ``projection_error_rms``, and
        ``residual_rms``; on real scans unmodeled errors often reach the
        50-update cap (``termination="iteration_limit"``). An explicit
        ``iterations``, tolerance, or penalty selects numerical stopping alone.
    ``fista``
        Least squares with physical total variation, ``iterations`` (default
        200), optionally ``nonnegative``. Default ``tv_weight=None`` calibrates
        TV from detector noise and linear interpolation error propagated through
        the backprojector, and stops when the proximal gradient mapping falls
        below the estimated noise level or reaches ``iterations``. ``info``
        reports ``resolved_tv_weight``, ``effective_tv_weight``,
        ``noise_score_rms``, ``effective_iterations``, ``early_stop`` and
        ``termination`` (``"noise_score"`` when it stopped at the noise level). An
        explicit positive ``tv_weight`` bypasses calibration and runs to
        ``iterations``; zero disables TV.
    ``sirt``
        SIRT, OS-SART, bounds, and support via :class:`~tomojax.recon.SIRTConfig`.
        With ``iterations=None`` and ``subsets=1``, stops at the estimated
        projection discrepancy up to 50 sweeps and enables FISTA momentum when
        ``relaxation <= 1``. Pass ``SIRTConfig(acceleration=False)`` for plain
        automatic SIRT, or an explicit ``iterations`` for that many plain sweeps
        (add ``acceleration=True`` for momentum). Ordered subsets
        (``subsets > 1``) use a fixed budget (default 50 sweeps, no momentum).
        ``info`` reports ``termination``, ``early_stop``,
        ``effective_iterations``, ``acceleration``, ``measurement_noise_rms``,
        ``projection_error_rms``, ``pre_update_residual_rms``, and
        ``slab_stopping`` for independent axial slabs. Replay an automatic run
        with both ``iterations`` and ``acceleration`` from ``info``.
    ``spdhg``
        The same objective as ``fista`` by stochastic primal-dual updates:
        one block of views per update, in an order drawn with ``seed``.
        The default runs ``ceil(600 * sqrt(B))`` updates for B blocks
        (``termination="iteration_limit"``, ``early_stop=False``) and uses the
        same isolated-voxel noise calibration as ``fista`` when
        ``tv_weight=None``.

    ``mlem`` is linear Poisson MLEM/OSEM (fixed sweep budget, default 50,
    ``termination="iteration_limit"``, ``early_stop=False``), not transmission
    counts (``MLEMConfig``). ``warm_start`` uses FBP/FDK for iterative methods;
    ``grid`` overrides the scan's output grid.

    ``out`` is a writable float32 array or memmap of shape ``(grid.nz, grid.ny,
    grid.nx)`` for ``fbp``, ``fourier``, ``cgls`` with a supported host-slab
    projector, and ``sirt`` and ``mlem`` on cone beams: the volume is made in
    host slabs straight into it and returned
    as ``volume``. It must not overlap the scan, and a later failure can leave
    it partly written. Cone CGLS, SIRT and MLEM also use host slabs on their
    own when their volumes would not fit on the device; with a memmapped
    ``out``, CGLS keeps its three work volumes in temporary files beside it.
    CUDA linear Joseph CGLS also supports parallel/laminography host slabs.
    Large CGLS host state automatically uses temporary files (``TMPDIR``);
    the returned array owns its file until released.

    ``config`` is from :mod:`tomojax.recon`; keywords replace its fields.
    Mismatched config classes or unknown keywords raise ``ValueError``.

    ``devices`` (``fbp``, ``cgls`` and ``fista``; one or several,
    ``jax.devices()`` for all) splits views across the devices: each projects
    its share against a full copy of the volume and reduces backprojections
    across devices. In ``fista``, automatic TV calibration is shared across
    devices, and resident float32 CUDA volumes of at least 64 MiB split the
    gradient reduction and CUDA TV proximal into overlapping slabs when each
    device gets a proper subvolume. The returned volume, on the first device,
    matches the one-device result up to cross-device reduction order. Automatic
    FDK slabs distribute z-slabs across devices; other host-slab paths use the
    first device only.

    :attr:`Reconstruction.info` holds ``config`` and solver records (iterations,
    termination, losses). Fourier's ``backend`` records the actual selected
    implementation separately from the requested configuration.

    An object wider than the detector leaves every row short. By default
    (``exterior="cylinder"``), ``fbp`` (including cone-beam FDK) and ``fourier``
    continue each detector row past both edges as the projection of a uniform
    cylinder about the rotation axis fitted to the row's edge values, removing
    the bright rim and bowl of a zero exterior; pass
    ``config=FBPConfig(exterior="zero")`` or
    ``config=FourierConfig(exterior="zero")`` to assume zero beyond the
    detector. Edges, contrast between neighboring regions, and fine detail
    inside the field of view do not depend on the continuation, while the
    absolute level is only as accurate as the cylinder model and
    non-cylindrical exteriors (such as a square bar) retain partial cupping and
    rim error. Outside the field of view the volume holds the reconstruction of
    the continued scan (``region="cyl"`` masks it). Iterative solvers fit the
    grid to the measured rays without an exterior model: they are accurate when
    the grid holds the whole object, and keep the rim and bowl when material
    extends outside it (unpenalized ``cgls`` also runs to its 50-update cap).
    Every reconstruction records ``info["truncation"]`` regardless of
    ``exterior`` or method: ``edge_ratio`` (mean edge value over the rows' mean,
    zero for a contained object), ``outside_fraction`` (the share of the
    continued scan's attenuation beyond the detector), ``interior_shift`` (the
    estimated change on the rotation axis from using the cylinder exterior, in
    the volume's units: compare it with the contrast you measure), ``radius``
    (median fitted cylinder radius), and ``field_of_view_radius``, in the scan's
    length unit.

    ``checkpoint`` names a directory where the run saves its state, so a run
    killed by a scheduler time limit, preemption, or a node failure resumes
    when called again with the same directory, scan, and settings, returning
    the volume an uninterrupted run returns. Iterative solvers save their loop
    state at a boundary once both 10 minutes and 20 times the previous save
    duration have passed (capping periodic save overhead at 5% of the run);
    host-slab runs save each slab as it finishes (flushing and fingerprinting
    named ``out`` memmaps instead of copying their planes). On SIGTERM inside a
    checkpointed loop on the main thread, or when canceled through
    :meth:`tomojax.progress.Control.cancel`, the run finishes its current
    iteration or slab, saves, and raises :class:`tomojax.errors.Canceled` whose
    ``checkpoint`` attribute names the directory. Hard
    kills resume from the last completed save. The directory is deleted when
    the call returns. Another run's checkpoint, a directory of unrelated files,
    an incompatible format, or a directory locked by another process is refused
    with :class:`tomojax.errors.InvalidParameter`. Resuming on a different GPU
    model, device count, build, or slab partition can change the last bits.

    Parameters
    ----------
    scan : Scan
        Measured line integrals with their acquisition geometry.
    method : str, optional
        One of auto, fbp, fourier, cgls, fista, spdhg, sirt or mlem.
    grid : Grid or None, optional
        Output grid; None retains the scan's grid.
    region : {"auto", "bbox", "cube", "cyl"}, RegionEstimate or None, optional
        The part of the scan's grid to reconstruct, in place of ``grid``: the box
        every view sees ("auto" and "bbox"; "auto" keeps the scan's grid, with a
        warning, when there is none), its largest cube, or its largest cylinder
        about the axis ("cyl", which also zeros the volume outside it); a
        :class:`RegionEstimate` gives its own grid. None keeps the scan's grid.
    out : numpy.ndarray or None, optional
        Writable float32 output or memmap for fbp/fourier, CUDA Joseph fista,
        host-slab cgls, and cone sirt/mlem;
        must not overlap input.
    config : reconstruction config or None, optional
        Method-specific settings; explicit keywords replace its fields.
    filter : str or None, optional
        FBP detector response for fbp or fourier: ramp, shepp-logan or hann.
        None uses the configuration or method default (ramp for fbp; native
        response for fourier).
    iterations : int or None, optional
        Iteration count; None uses the method default or configuration.
    tv_weight : float or None, optional
        Dimensionless total-variation weight for fista or spdhg. None uses the
        configuration (automatic calibration by default); zero disables TV.
    nonnegative : bool or None, optional
        Nonnegative constraint where supported; None uses the configuration.
    warm_start : bool or None, optional
        Use FBP/FDK initialization for supported iterative methods.
    seed : int or None, optional
        Random seed for stochastic view ordering where supported.
    devices : jax.Device or sequence of jax.Device or None, optional
        Devices sharing views for supported methods; None uses the default device.
    checkpoint : str or path-like or None, optional
        Directory of the run's saved state, created if absent; None saves nothing.

    Returns
    -------
    Reconstruction
        Volume with the actual scan geometry, method and solver records.

    Raises
    ------
    ValueError
        If the method, configuration, or options are incompatible, or if
        ``checkpoint`` holds another run's state, unrelated files, an
        incompatible format, or an active lock (``InvalidParameter``).
    tomojax.errors.Canceled
        If canceled through a :class:`tomojax.progress.Control`, or if SIGTERM
        stops a checkpointed run on the main thread; when ``checkpoint`` is set,
        ``Canceled.checkpoint`` names the directory after saving.

    Examples
    --------
    >>> import tomojax as tj
    >>> scan = tj.load("scan.nxs")  # doctest: +SKIP
    >>> recon = tj.reconstruct(scan, "cgls", iterations=50)  # doctest: +SKIP
    """
    from tomojax._workflow.planning import resolve_reconstruction

    progress.check_canceled()

    plan = resolve_reconstruction(
        scan,
        method,
        grid=grid,
        region=region,
        config=config,
        filter=filter,
        iterations=iterations,
        tv_weight=tv_weight,
        nonnegative=nonnegative,
        warm_start=warm_start,
        seed=seed,
        devices=devices,
    )
    return reconstruct_resolved(plan, out=out, checkpoint=checkpoint)


@explain_cuda_failure()
def reconstruct_resolved(
    plan: ResolvedReconstruction,
    *,
    out: np.ndarray | None = None,
    checkpoint: str | PathLike[str] | None = None,
) -> Reconstruction:
    """Execute the resolved method on its original selected support and grid."""
    from tomojax.core.checkpoint import checkpointing
    from tomojax.recon.api import reconstruct_arrays

    scan, method = plan.scan, plan.method
    if method not in {"fbp", "fourier"}:
        _suggest_binning(scan, "iterative reconstruction")
    radius = None if plan.cylinder is None else plan.cylinder.radius
    with (
        checkpointing(
            checkpoint,
            described={
                "method": method,
                "config": plan.config,
                "source": scan.source.source_path if scan.source else None,
            },
            operation="reconstruct",
            settings=(plan.config, plan.warm_start, plan.cylinder, scan.geometry),
            input=scan.projections,
        ),
        registration(scan, cylinder_radius=radius),
        progress.runtime() as actual,
    ):
        volume, info = reconstruct_arrays(
            method,
            scan.geometry,
            scan.grid,
            scan.detector,
            scan.projections,
            config=plan.config,
            warm_start=plan.warm_start,
            out=out,
        )
    slabbed = bool(info.get("host_slabs") or info.get("host_volume"))
    info["placement"] = actual.record(slabbed=slabbed)
    from tomojax.core.host_volume import scratch_directory

    directory = None if out is not None else scratch_directory(volume)
    if directory is not None:  # more than half of host memory: an unlinked file in TMPDIR
        info = {**info, "temporary_file": str(directory)}
        LOG.info("The %.1f GiB volume is held in a temporary file in %s, as more than half of "
                 "host memory", volume.nbytes / 2**30, directory)  # fmt: skip
    if plan.cylinder is not None:
        from tomojax._workflow.field_of_view import zero_outside_cylinder

        volume = zero_outside_cylinder(volume, scan.grid, plan.cylinder, in_place=True)
    return Reconstruction(volume, scan.geometry, method, {**info, **plan.info}, scan)


def reconstruct_slabs(
    frames: Frames,
    method: Literal["fbp", "fourier", "sirt", "mlem"],
    *,
    grid: Grid | None = None,
    corrections: Sequence[corrections.Step] = (),
    config: recon.FBPConfig
    | recon.FourierConfig
    | recon.SIRTConfig
    | recon.MLEMConfig
    | None = None,
    warm_start: bool | None = None,
    epsilon: float = 1e-6,
    flat_weights: FlatWeighting = FLAT_WEIGHTS,
    checkpoint: str | PathLike[str] | None = None,
    info: dict[str, object] | None = None,
) -> AbstractContextManager[Iterator[tuple[slice, Reconstruction]]]:
    """Correct raw frames and reconstruct bounded z slabs.

    Use ``with reconstruct_slabs(frames, "fbp", ...) as slabs:`` and iterate
    over ``(z_slice, reconstruction)``. The slice indexes the requested output
    grid; each reconstruction retains its physical global grid, genuine corrected
    source rows (including interpolation neighbors), and correction records.
    Volumes are owned float32 host arrays. Write or consume each slab as it
    arrives: accumulating every yielded :class:`Reconstruction` retains both its
    volume and its corrected source rows in memory.

    This supports cone-beam FDK, unposed axial parallel beams and whole-row line-integral
    corrections such as Rings (geometry-dependent steps bind to
    ``frames.geometry`` automatically), not Paganin or view rejection. SIRT/OSEM
    also require matching detector and voxel planes; ``warm_start=True`` uses FBP.
    Method configurations retain the same meaning as in ``reconstruct``.
    Pass ``FourierConfig(filter="ramp")`` (or ``"hann"`` or ``"shepp-logan"``)
    to match FBP's detector response in streamed Fourier reconstruction.
    ``fbp`` (parallel and cone FDK) and ``fourier`` continue detector rows with
    ``exterior="cylinder"`` by default; pass ``FBPConfig(exterior="zero")`` or
    ``FourierConfig(exterior="zero")`` to assume zero beyond the detector. Each
    yielded slab carries ``info["truncation"]`` (``edge_ratio``,
    ``outside_fraction``, ``interior_shift``, ``radius``, and
    ``field_of_view_radius``) computed from that slab's detector rows. Passing
    a dictionary as ``info=`` collects the whole-scan record once the iterator
    finishes: ``info["truncation"]`` matches an in-memory reconstruction over
    the full detector (each sampled detector row counted once, including where
    slabs overlap), alongside ``config``, ``host_slabs=True``, the resolved
    Fourier ``backend``, and iterative stopping diagnostics (``termination``,
    ``early_stop``, ``effective_iterations``, and per-slab ``slab_stopping`` on
    independent planes). Bracketing multi-frame flat sets default to
    ``flat_weights="views"`` (see :meth:`tomojax.Frames.corrected`); pass
    ``flat_weights="position"`` to blend by position alone.
    CUDA consumes corrected rows without a corrected-projection host round trip.
    Unfiltered HDF5 stacks use parallel positional reads with cached chunk
    offsets. Raw row blocks target 1 MiB per view (at least 1 GiB per block,
    capped at 1/24 of available host memory, with at most three blocks held).
    The first block prefetches while calibration runs, and calibration is reused
    across slabs.
    The context drains pending work on normal completion, early exit or failure.
    Raw frames and calibration arrays must remain unchanged inside the context.
    Cone FDK corrects host row bands in device-sized view batches and retains
    the complete acquisition's weights while reconstructing each slab. On a rolled
    detector, Fourier reads the rotated row footprint with its interpolation
    neighbors. The oblique FBP filter and iterative projector couple the rows;
    these methods read the complete detector and return the coupled volume in one slab.

    ``checkpoint`` names a directory recording which slabs are finished, so a
    killed run resumes when called again with the same directory, frames, and
    settings. A slab counts as finished once the body of the loop receiving it
    returns; a resumed run yields only the slabs after the last finished one,
    so write and flush each slab to persistent storage (for example a memmap
    opened with ``mode="r+"`` when it exists). SIRT and MLEM also save the
    in-progress slab state once both 10 minutes and 20 times the previous save
    duration have passed (at most 5% overhead). On SIGTERM in the main thread,
    or when canceled through :meth:`tomojax.progress.Control.cancel`, the run
    stops after the current slab (or SIRT/MLEM iteration), saves, and raises
    :class:`tomojax.errors.Canceled` whose ``checkpoint`` attribute names the
    directory. Leaving the ``with`` block early
    keeps the directory for a later run; finishing the last slab deletes it.
    Another run's checkpoint, unrelated files, an incompatible format, or a
    locked directory is refused with :class:`tomojax.errors.InvalidParameter`.

    Parameters
    ----------
    frames : Frames
        Raw detector frames for cone FDK or unposed axial parallel-beam geometry.
    method : {'fbp', 'fourier', 'sirt', 'mlem'}
        Slab reconstruction method.
    grid : Grid or None, optional
        Output grid; None uses the frames' grid.
    corrections : sequence of Step, optional
        Whole-row line-integral corrections applied before reconstruction.
    config : FBPConfig or FourierConfig or SIRTConfig or MLEMConfig or None, optional
        Method-specific settings, as for ``reconstruct``.
    warm_start : bool or None, optional
        Start SIRT/OSEM from FBP; None uses the method's default.
    epsilon : float, optional
        Positive finite lower bound for flat field and transmission.
    flat_weights : {'views', 'position'}, optional
        How a view between two flat sets blends them, as for
        :meth:`tomojax.Frames.corrected`.
    checkpoint : str or path-like or None, optional
        Directory recording the finished slabs, created if absent; None records nothing.
    info : dict or None, optional
        Whole-volume diagnostics, populated when the iterator finishes.

    Returns
    -------
    contextlib.AbstractContextManager
        Context manager yielding owned slabs paired with their z slices in the requested grid.

    Raises
    ------
    ValueError
        If geometry, method, corrections, or configuration do not support slabs,
        or if ``checkpoint`` holds another run's state, unrelated files, an
        incompatible format, or an active lock (``InvalidParameter``).
    tomojax.errors.Canceled
        If canceled through a :class:`tomojax.progress.Control`, or if SIGTERM
        stops a checkpointed run on the main thread; when ``checkpoint`` is set,
        ``Canceled.checkpoint`` names the directory after saving.

    Examples
    --------
    >>> import tomojax as tj
    >>> frames = tj.load_frames("raw_scan.nxs")  # doctest: +SKIP
    >>> with tj.reconstruct_slabs(frames, "fbp") as slabs:  # doctest: +SKIP
    ...     for z_slice, recon in slabs:
    ...         print(z_slice, recon.volume.shape)
    """
    return _reconstruct_slabs_context(
        frames,
        method,
        grid=grid,
        corrections=corrections,
        config=config,
        warm_start=warm_start,
        epsilon=epsilon,
        flat_weights=flat_weights,
        checkpoint=checkpoint,
        info=info,
    )


def slab_settings(
    frames: Frames,
    method: str,
    *,
    grid: Grid | None,
    config: MethodConfig | None,
    warm_start: bool | None,
    corrections: Sequence[corrections.Step],
    epsilon: float,
    flat_weights: FlatWeighting,
) -> tuple[Grid, MethodConfig, tuple[corrections.Step, ...]]:
    """Resolve method, geometry and supported corrections before opening a raw row source."""
    from tomojax.corrections.api import positive_finite, slab_steps
    from tomojax.errors import InvalidParameter
    from tomojax.geometry import ParallelGeometry
    from tomojax.geometry.api import is_cone_beam
    from tomojax.recon.api import method_config, validate_reconstruction

    cone = is_cone_beam(frames.geometry)
    cfg = method_config(method, config=config, warm_start=warm_start)
    if cfg.method not in {"fbp", "fourier", "sirt", "mlem"} or not (
        type(frames.geometry) is ParallelGeometry or (cone and cfg.method == "fbp")
    ):
        raise InvalidParameter(
            "method", "reconstruct_slabs requires cone FBP or unposed axial FBP/Fourier/SIRT/MLEM"
        )
    target = frames.geometry.grid if grid is None else grid
    validate_reconstruction(frames.geometry, target, cfg, warm_start=bool(warm_start))
    bound = bind_step_geometry(corrections, frames.geometry)
    slab_steps(bound)
    positive_finite(epsilon, "epsilon")
    _ = flat_weighting(flat_weights)
    return target, cfg, bound


@contextmanager
def _reconstruct_slabs_context(
    frames: Frames,
    method: Literal["fbp", "fourier", "sirt", "mlem"],
    *,
    grid: Grid | None,
    corrections: Sequence[corrections.Step],
    config: MethodConfig | None,
    warm_start: bool | None,
    epsilon: float,
    flat_weights: FlatWeighting,
    checkpoint: str | PathLike[str] | None,
    info: dict[str, object] | None,
) -> Iterator[Iterator[tuple[slice, Reconstruction]]]:
    from tomojax.core.checkpoint import checkpointing, keep_checkpoint
    from tomojax.corrections.api import corrected_row_source
    from tomojax.geometry.api import is_cone_beam
    from tomojax.recon.api import reconstruct_slabs as array_slabs, solver_name

    cone = is_cone_beam(frames.geometry)
    with (
        explain_cuda_failure(),
        progress.task("reconstruct", solver_name(method, frames.geometry), None, "slabs"),
    ):
        target, cfg, bound = slab_settings(
            frames,
            method,
            grid=grid,
            config=config,
            warm_start=warm_start,
            corrections=corrections,
            epsilon=epsilon,
            flat_weights=flat_weights,
        )
        raw = (frames.flats, frames.darks, frames.flat_positions, frames.white_level)
        timing = (frames.view_positions, frames.exposure_times, frames.flat_exposure_times)
        with (
            checkpointing(
                checkpoint,
                described={
                    "method": method,
                    "config": cfg,
                    "source": frames.source.source_path if frames.source else None,
                },
                operation="reconstruct_slabs",
                settings=(cfg, target, bound, epsilon, flat_weights, warm_start, frames.geometry),
                input=(frames.counts, *raw, *timing, frames.dark_source, frames.linear_limit),
            ),
            corrected_row_source(
                frames.counts,
                flats=frames.flats,
                flat_positions=frames.flat_positions,
                darks=frames.darks,
                white_level=frames.white_level,
                steps=bound,
                epsilon=epsilon,
                view_positions=frames.view_positions,
                exposure_times=frames.exposure_times,
                flat_exposure_times=frames.flat_exposure_times,
                host=cone or bool(getattr(frames.geometry, "detector_roll", 0)),
                dark_source=frames.dark_source,
                linear_limit=frames.linear_limit,
                flat_weights=flat_weights,
            ) as source,
        ):

            def read(
                first: int, stop: int
            ) -> tuple[jax.Array | np.ndarray, CorrectedRows | corrections.Corrected]:
                corrected = source(first, stop)
                return corrected.projections, corrected

            completed = array_slabs(
                frames.geometry,
                target,
                frames.geometry.detector,
                read,
                config=cfg,
                warm_start=bool(warm_start),
                info=info,
            )
            with (
                closing(completed),
                closing(_slab_results(completed, frames, target, method, cfg)) as output,
            ):
                yield output
                if inspect.getgeneratorstate(output) != inspect.GEN_CLOSED:
                    keep_checkpoint()  # the caller stopped early: later slabs remain


def _slab_results(
    prepared: Iterator[ReconstructionSlab[Any]],
    frames: Frames,
    target: Grid,
    method: str,
    cfg: MethodConfig,
) -> Generator[tuple[slice, Reconstruction]]:
    """Assemble the scan and reconstruction for each completed, owned output slab."""
    from tomojax._scan.model import detector_window, rebuild
    from tomojax.geometry import grid_volume_origin

    origin = grid_volume_origin(target)
    previews = (
        SlabPlanes(target) if progress.previewing() and getattr(cfg, "iterations", 0) != 0 else None
    )
    with registration(frames), progress.outcome():
        progress.check_canceled()
        for slab in prepared:
            block = replace(
                target,
                nz=slab.stop - slab.start,
                vol_center=None,
                vol_origin=(origin[0], origin[1], origin[2] + slab.start * target.vz),
            )
            geometry = rebuild(
                frames.geometry,
                grid=block,
                detector=detector_window(slab.rows, slice(None), frames.geometry.detector),
            )
            cropped = cropped_record(slab.rows, slice(None), frames.geometry.detector)
            scan = Scan(
                slab.projections,
                geometry,
                frames.name,
                (*frames.corrections, cropped, *slab.context.records),
                frames.source,
                view_positions=frames.view_positions,
            )
            info = {
                "config": cfg,
                "host_slabs": True,
                "source_rows": (slab.rows.start, slab.rows.stop),
                **slab.info,
            }
            result = (
                slice(slab.start, slab.stop),
                Reconstruction(slab.volume, scan.geometry, method, info, scan),
            )
            last = slab.stop == target.nz
            if previews is not None:
                previews.add(slab.start, slab.stop, slab.volume)
                if last:
                    progress.preview(previews.images, final=True)
            progress.step("slabs", remaining=0 if last else None)
            yield result
            del result
            if not last:
                progress.check_canceled()
            del slab, scan  # Do not retain device rows while preparing the next slab.


def _estimate_center(scan: Scan, selection: CenterSelection) -> tuple[Scan, dict[str, object]]:
    """Update the detector center and record the selected estimator's diagnostics."""
    if selection.focus is None:
        report = selection.measure(scan.projections)
        geometry, diagnostics = report.geometry, report.to_dict()
        method = "opposite_view_correlation"
    else:
        progress.placed(("cpu:0",))
        value, diagnostics = selection.focused(scan.projections, scan.view_positions)
        d = selection.geometry.detector
        geometry = replace(
            selection.geometry, detector=replace(d, center=(value * d.du, d.center[1]))
        )
        method = selection.method
    calibration: dict[str, object] = {
        "status": "estimated",
        "method": method,
        "diagnostics": diagnostics,
        "input": list(scan.detector.center),
        "output": list(geometry.detector.center),
        "unit": "geometry_length",
    }
    source = scan.source
    if source is not None:
        source = replace(
            source,
            geometry_metadata={
                **source.geometry_metadata,
                "detector_center_status": "inferred",
                "detector_center_calibration": calibration,
            },
        )
    return replace(scan, geometry=geometry, source=source), calibration


def align(
    scan: Scan,
    *,
    fit: Fit | tuple[Fit, ...] | None = None,
    grid: Grid | None = None,
    region: Region | None = None,
    method: AlignMethod | None = None,
    motion: Literal["independent", "smooth"] | None = None,
    iterations: int | None = None,
    image_iterations: int | None = None,
    tolerance: float | None = None,
    interpolation: Literal["linear", "cubic"] | None = None,
    resolution: int | None = None,
    config: AlignConfig | None = None,
    checkpoint: str | PathLike[str] | None = None,
) -> Alignment:
    """Estimate geometry errors and per-view motion for ``scan``.

    ``fit`` is one name or a tuple of names: ``center``, ``tilt``, ``roll``,
    ``translations`` and ``rotations``. Center, tilt and roll are shared by all
    views. Translations and rotations are per view; every unselected parameter
    is held at its declared value. Translations have two components for parallel
    rays and three for cone beams. Rotations are the per-view wobble of the
    rotation axis, two components; each view's angle about the axis is taken as
    recorded, since stage encoders give it and errors in it are rare. By
    default each view's corrections are free of its neighbors';
    ``motion="smooth"`` fits drift and slips instead. Per-view rotations have a
    prior on their physical size.

    On an unposed, unrolled axial parallel scan with a well-sampled half-turn,
    ``fit="center"`` alone (with ``method=None`` and no ``grid``, ``region``,
    ``config`` or joint solver keywords) measures the rotation axis directly
    from the projections across the full detector width without reconstructing
    a working volume. Over up to 401 views it scores Fourier-slice edge
    sharpness, half-turn Fourier parity and the sinusoidal first projection
    moment, then refines the center to fractions of a native pixel with up to
    16 opposing view pairs on their common illuminated support (including
    endpoint-exclusive half-turns and narrow detectors), accounting for
    stationary detector structure and linear flat-field drift. ``result.image``
    and ``result.grid`` are None.

    Passing ``method="joint"``, ``grid``, ``region``, ``config``, any joint
    solver keyword, a multi-parameter ``fit`` or another geometry runs the
    joint solver, fitting an image and the selected parameters together across
    parallel, laminography and cone-beam scans. Axial parallel joint fits that
    include ``"center"`` initialize from the measured center. Existing poses
    are preserved and the fitted correction is composed with them. The returned
    scan carries both calibrated global geometry and local poses, so
    reconstructing or saving it preserves the fitted ray geometry.

    ``method="opposite-views"`` runs the pair-only check without an image,
    matching up to 16 near-opposite view pairs (within two degrees) of an
    unposed axial parallel scan by row-wise gradient correlation. It can be
    ambiguous under strong defects or repetitive structure and records its pair
    diagnostics in ``info["diagnostics"]``.

    ``checkpoint`` names a directory where a joint fit saves its state, so a
    killed run resumes when called again with the same directory, scan, and
    settings, returning the alignment an uninterrupted run returns. Each
    finished resolution level is saved, and within a level and refinement the
    volume, poses, damping, and step history are saved at step boundaries once
    both 10 minutes and 20 times the previous save duration have passed (at
    most 5% overhead). On SIGTERM inside a checkpointed loop on the main
    thread, or when canceled through :meth:`tomojax.progress.Control.cancel`,
    the fit saves after the current step and raises
    :class:`tomojax.errors.Canceled` whose ``checkpoint`` attribute names the
    directory. Hard kills resume from the last completed
    save. The directory is deleted when the call returns. Another run's
    checkpoint, unrelated files, an incompatible format, or a locked directory
    is refused with :class:`tomojax.errors.InvalidParameter`. Cone-beam pose
    derivatives accumulate with floating-point atomics, so two cone fits can
    differ in the last bits; resuming on a different GPU model or build can
    also change the last bits. Direct axial center fits run on CPU without an
    iterative loop.

    Parameters
    ----------
    scan : Scan
        Measured line integrals with their declared acquisition geometry.
    fit : str or tuple of str, optional
        Parameters to fit. None fits the method's own: ``"translations"`` for
        "joint" and ``"center"`` for "opposite-views".
    grid : Grid or None, optional
        Reconstruction region and sampling. None uses a bounded working grid
        with the scan's physical field of view; an explicit grid is kept.
    region : {"auto", "bbox", "cube", "cyl"}, RegionEstimate or None, optional
        The part of the scan's grid to fit on, in place of ``grid``, as for
        :func:`reconstruct`.
    method : {"joint", "opposite-views"} or None, optional
        None selects the axial center search when applicable and the joint fit otherwise.
    motion : {"independent", "smooth"} or None, optional
        Whether each view moves independently of its neighbors (the default) or the
        motion is smooth drift with occasional slips, for a steady stage (see
        :class:`tomojax.alignment.AlignConfig`); None keeps ``config``'s value.
    iterations, image_iterations, tolerance, interpolation, resolution : optional
        The everyday settings, replacing the fields of the same name in ``config``
        (see :class:`tomojax.alignment.AlignConfig`); None keeps its value.
    config : tomojax.alignment.AlignConfig or None, optional
        Every setting of the joint fit (imported from :mod:`tomojax.alignment`);
        None uses the defaults.
    checkpoint : str or path-like or None, optional
        Directory of the fit's saved state, created if absent; None saves nothing.

    Returns
    -------
    Alignment
        Corrected scan, its per-view poses, the record and (for joint fits) the
        working image on ``result.grid``. Direct axial center fits
        (``info["method"] == "sinogram-focus"``) and ``method="opposite-views"``
        leave ``image`` and ``grid`` as None and record their diagnostics in
        ``info["diagnostics"]`` (``focus_px``, ``binning``, ``views``, ``pairs``
        for direct fits; per-pair constraints and consistency statistics for
        ``opposite-views``). Joint fits record resolved settings in
        ``info["config"]``, convergence diagnostics in ``info["report"]``, and
        cross-detector column disturbance screening in
        ``info["disturbed_views"]`` (``score``, ``ranges``, ``angles``, and
        ``message``; pass confirmed ranges to
        ``AlignConfig(exclude_views=...)``). When ``exclude_views`` omits views,
        ``info["fit_views"]`` lists the retained input indices and excluded views
        remain in ``result.scan`` with pose increments interpolated within each
        trajectory.
        Each fitted global appears under its name in ``info["calibration"]`` as
        ``{"value": ..., "change": ...}``: the final value, the center in geometry
        length units (u, or u/v for cone beams) and angles in degrees, and the
        fitted change, the center in detector pixels and angles in degrees. Tilt
        is the signed inclination ``atan2(axis_y, axis_z)``; positive roll turns
        detector u towards v (right-handed about the detector normal, which points
        back towards the source), so a point above the detector center projects
        towards +u. Cone segments with different calibrated values report them in
        segment order.

    Raises
    ------
    ValueError
        If the fit selection is empty, repeated, or unknown, the method does
        not fit it, the geometry has not been declared, or ``checkpoint`` holds
        another run's state, unrelated files, an incompatible format, or an
        active lock (``InvalidParameter``).
    tomojax.errors.Canceled
        If canceled through a :class:`tomojax.progress.Control`, or if SIGTERM
        stops a checkpointed run on the main thread; when ``checkpoint`` is set,
        ``Canceled.checkpoint`` names the directory after saving.

    Examples
    --------
    >>> import tomojax as tj
    >>> from tomojax.alignment import AlignConfig
    >>> scan = tj.load("scan.nxs")  # doctest: +SKIP
    >>> centered = tj.align(scan, fit="center")  # doctest: +SKIP
    >>> result = tj.align(scan, fit=("translations", "roll"))  # doctest: +SKIP
    >>> result = tj.align(scan, config=AlignConfig(levels=(1,)))  # doctest: +SKIP
    >>> recon = tj.reconstruct(result.scan)  # doctest: +SKIP
    """
    from tomojax._workflow.smooth_alignment import resolve_alignment

    with alignment_task(method, fit):
        plan = resolve_alignment(
            scan,
            fit=fit,
            method=method,
            grid=grid,
            region=region,
            motion=motion,
            iterations=iterations,
            image_iterations=image_iterations,
            tolerance=tolerance,
            interpolation=interpolation,
            resolution=resolution,
            config=config,
        )
        return align_resolved(scan, plan, checkpoint=checkpoint)


def alignment_task(
    method: str | None, fit: str | tuple[str, ...] | None
) -> AbstractContextManager[None]:
    """The outer progress task of one alignment, around its resolution and its fit."""
    label = ", ".join(map(str, method_fit(method, fit)))
    return progress.task("align", label, None, "fit")


def align_resolved(
    scan: Scan,
    plan: CenterSelection | JointSelection,
    *,
    checkpoint: str | PathLike[str] | None = None,
) -> Alignment:
    """Execute the selected center or joint-image fit, in the checkpoint at ``checkpoint``."""
    from tomojax._workflow.smooth_alignment import CenterSelection, align_poses
    from tomojax.core.checkpoint import checkpointing

    settings = plan if isinstance(plan, CenterSelection) else replace(plan, info={})
    described: dict[str, object] = (
        {"method": plan.method}
        if isinstance(plan, CenterSelection)
        else {"method": "joint", "fit": plan.fit, "config": plan.config}
    )
    source = scan.source.source_path if scan.source else None
    with checkpointing(
        checkpoint,
        described={**described, "source": source},
        operation="align",
        settings=(settings, scan.geometry),
        input=(scan.projections, scan.view_positions),
    ):
        if isinstance(plan, CenterSelection):
            with registration(scan), progress.runtime() as actual:
                alignment = _align_center(scan, plan)
            return replace(
                alignment, info={**alignment.info, "placement": actual.record(slabbed=False)}
            )
        radius = None if plan.cylinder is None else plan.cylinder.radius
        with registration(scan, cylinder_radius=radius), progress.runtime() as actual:
            result = align_poses(scan, plan)
        slabbed = any(
            "host" in level["stacks"]["placement"].values()
            for level in cast("Sequence[dict[str, Any]]", result.info.get("levels", ()))
        )
        grid, cylinder = plan.grid, plan.cylinder
        assert result.scan is not None  # the alignment's corrected scan
        image = result.volume
        if cylinder is not None and image is not None and result.grid is not None:
            from tomojax._workflow.field_of_view import zero_outside_cylinder

            image = zero_outside_cylinder(image, result.grid, cylinder, in_place=True)
        # The fit's working grid belongs to its image; the scan keeps the caller's grid.
        corrected = with_grid(result.scan, scan.grid if grid is None else grid)
        info = {**result.info, **plan.info, "placement": actual.record(slabbed=slabbed)}
        return Alignment(corrected, info, image, result.grid)


def _align_center(scan: Scan, selection: CenterSelection) -> Alignment:
    """Return the calibrated scan and its center correction in native detector pixels."""
    centered, record = _estimate_center(scan, selection)
    before, after = scan.detector.center[0], centered.detector.center[0]
    center = {"value": float(after), "change": float((after - before) / scan.detector.du)}
    info = {
        "method": selection.method,
        "fit": ["center"],
        "calibration": {"center": center},
        "diagnostics": record["diagnostics"],
    }
    return Alignment(centered, info)


def _oversampling(scan: Scan) -> float:
    """Voxel-to-detector pitch ratio, using the least magnified cone segment.

    Compare the lesser row/column ratio; tilted axes use the smallest voxel side.
    """
    from tomojax.geometry import ConeSegments, LaminographyGeometry, RotationAxisGeometry, beam_of

    detector, grid, geometry = scan.detector, scan.grid, scan.geometry
    # Only the physical base matters here, not serialized poses or file metadata.
    base: Any = geometry
    while hasattr(base, "base"):
        base = base.base
    parts = base.segments if isinstance(base, ConeSegments) else (base,)
    beams = [beam_of(part) for part in parts]
    magnification = min(1.0 if beam is None else float(beam.magnification) for beam in beams)
    if isinstance(base, LaminographyGeometry | RotationAxisGeometry):
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
    warn(
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
    "load_alignment",
    "load_reconstruction",
    "load_simulation",
    "project",
    "reconstruct",
    "save",
]
