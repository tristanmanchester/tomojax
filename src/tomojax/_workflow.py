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
import json
import logging
import os
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from tomojax import _typed_arrays, progress, recon
from tomojax._loading import Frames, load
from tomojax._parameters import AlignMethod, AnnotationModule, Filter, Fit, Method, RegionName
from tomojax._scan import (
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
from tomojax._simulate import Simulation
from tomojax._typed_arrays import Device
from tomojax.alignment import AlignConfig
from tomojax.geometry import Grid, ScanGeometry
from tomojax.progress import warn
from tomojax.recon.api import MethodConfig

if TYPE_CHECKING:
    from collections.abc import Generator

    from tomojax import _region, corrections
    from tomojax._planning import ResolvedReconstruction
    from tomojax._smooth_alignment import CentreSelection, JointSelection
    from tomojax.geometry import Detector
    from tomojax.recon.types import ReconstructionSlab
else:
    corrections = AnnotationModule("tomojax.corrections")
    _region = AnnotationModule("tomojax._region")

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
    from tomojax._files import saved_volume_grid
    from tomojax.io import load_dataset

    record = load_dataset(os.fspath(path))
    if record.volume is None:
        raise ValueError(f"{path} holds no reconstructed volume")
    meta = record.geometry_metadata
    if meta.get("volume_role") == "phantom":
        raise ValueError(
            f"{path} holds a simulation phantom, not a reconstruction; "
            "load it with tomojax.load_simulation"
        )
    scan = None if record.projections is None else load(path)
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
        The simulated scan and its ground-truth phantom, ``(nz, ny, nx)``.

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
    from tomojax._simulate import Simulation
    from tomojax.io import load_dataset

    record = load_dataset(os.fspath(path))
    if record.volume is None or record.geometry_metadata.get("volume_role") != "phantom":
        raise ValueError(f"{path} holds no simulation phantom")
    return Simulation(load(path), np.asarray(record.volume))


def load_alignment(path: str | PathLike[str]) -> Alignment:
    """Load an :class:`Alignment` saved by :func:`save` or ``tomojax align``.

    Preserves the scan's own grid on ``result.scan.grid`` separately from the
    working image's grid on ``result.grid``, and supports image-free alignments
    such as ``method="opposite-views"`` (leaving ``image`` and ``grid`` None).

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
    from tomojax.geometry import Grid
    from tomojax.io import load_dataset

    record = load_dataset(os.fspath(path))
    meta = record.geometry_metadata
    raw_info = meta.get("alignment_info")
    if not isinstance(raw_info, dict):
        raise ValueError(f"{path} holds no alignment record")
    scan = load(path)
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
    poses and working image. A simulation is saved as its scan together with its
    ground-truth phantom as the stored volume.

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
    from tomojax._simulate import Simulation
    from tomojax.io import save_dataset
    from tomojax.io.api import replace_dataset

    with progress.task("save", Path(path).name, None, ""):
        if isinstance(item, Scan):
            record = record_of(item)
        elif isinstance(item, Simulation):
            record = record_of(item.scan)
            record.volume = item.phantom
            record.geometry_metadata["volume_role"] = "phantom"
            record.geometry_metadata.pop("reconstruction_method", None)
            record.geometry_metadata.pop("reconstruction_info", None)
        elif isinstance(item, Reconstruction):
            if item.scan is None and include_scan:
                raise ValueError(
                    "this reconstruction was loaded without its scan: there is none to save"
                )
            if item.scan is None:
                record = geometry_record(item.geometry)
                source = item.info.get("source")
            else:
                record = record_of(item.scan, grid=item.grid)
                source = source_of(item.scan)
                if not include_scan:
                    record = replace_dataset(record, projections=None)
            record.volume = item.volume
            record.geometry_metadata["reconstruction_method"] = item.method
            # As JSON, which the file's metadata is: a configuration as its fields, and
            # what else JSON cannot hold as text.
            record.geometry_metadata["reconstruction_info"] = json.loads(
                json.dumps({**item.info, "source": source}, default=_json_value)
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
            record.geometry_metadata["alignment_info"] = json.loads(
                json.dumps(info, default=_json_value)
            )
        else:
            raise TypeError(
                f"save takes a Scan, Reconstruction, Alignment or Simulation, not {type(item)}"
            )
        if not isinstance(item, Simulation):  # a scan loaded from a simulation carries its tag
            _ = record.geometry_metadata.pop("volume_role", None)
        save_dataset(path, record)


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
) -> Reconstruction:
    """Reconstruct ``scan`` with ``method``.

    ``auto`` (default) chooses Fourier inversion for unposed axial half-turn
    scans, matched CGLS for segmented cone geometry, and FBP/FDK otherwise,
    laminography and tilted-axis scans included: for those, ``fista`` gives a
    better image at the cost of its iterations (docs/measurements.md). It
    reconstructs with the geometry and poses the scan carries, on its grid or
    the ``grid`` given; it fits no poses (use :func:`align`).
    It estimates nothing: a centre comes from the file or from :func:`align`.
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
        or tilted axes. Input and output stay on the host;
        :class:`~tomojax.recon.FourierConfig` selects NumPy or CuPy CUDA.
    ``cgls``
        Least squares by conjugate gradients, ``iterations`` (default 50).
    ``fista``
        Least squares with physical total variation, ``iterations`` (default
        50), optionally ``nonnegative``. Default ``tv_weight=None`` raises an
        empirical minimum of 5e-5 using a projection-only noise estimate. An
        explicit positive weight bypasses estimation; zero disables TV.
    ``sirt``
        SIRT, OS-SART, bounds and support via :class:`~tomojax.recon.SIRTConfig`.
    ``spdhg``
        The same objective as ``fista`` by stochastic primal-dual updates:
        ``iterations`` (default 600) each update one block of views, in an
        order drawn with ``seed``.

    ``mlem`` is linear Poisson MLEM/OSEM, not transmission counts (``MLEMConfig``).
    ``warm_start`` uses FBP/FDK for least squares; ``grid`` overrides the scan's output grid.

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
    ``jax.devices()`` for all) shares the views among them: each projects its
    own views and holds the whole volume, and their backprojections are
    summed. The volume, on the first device, is the one-device result up to
    the order of that sum. Host slabs use the first device only.

    :attr:`Reconstruction.info` holds ``config`` and solver records (iterations,
    termination, losses). Fourier's ``backend`` records the actual selected
    implementation separately from the requested configuration.

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
        about the axis ("cyl", which also zeroes the volume outside it); a
        :class:`RegionEstimate` gives its own grid. None keeps the scan's grid.
    out : numpy.ndarray or None, optional
        Writable float32 output or memmap for fbp/fourier, CUDA Joseph fista,
        host-slab cgls, and cone sirt/mlem;
        must not overlap input.
    config : reconstruction config or None, optional
        Method-specific settings; explicit keywords replace its fields.
    filter : str or None, optional
        FBP filter: ramp, shepp-logan or hann; None uses the configuration.
    iterations : int or None, optional
        Iteration count; None uses the method default or configuration.
    tv_weight : float or None, optional
        Total-variation weight for fista/spdhg; None uses the configuration.
    nonnegative : bool or None, optional
        Nonnegative constraint where supported; None uses the configuration.
    warm_start : bool or None, optional
        Use FBP/FDK initialization for supported iterative methods.
    seed : int or None, optional
        Random seed for stochastic view ordering where supported.
    devices : jax.Device or sequence of jax.Device or None, optional
        Devices sharing views for supported methods; None uses the default device.

    Returns
    -------
    Reconstruction
        Volume with the actual scan geometry, method and solver records.

    Raises
    ------
    ValueError
        If the method, configuration or options are incompatible.

    Examples
    --------
    >>> import tomojax as tj
    >>> scan = tj.load("scan.nxs")  # doctest: +SKIP
    >>> recon = tj.reconstruct(scan, "cgls", iterations=50)  # doctest: +SKIP
    """
    from tomojax._planning import resolve_reconstruction

    progress.check_cancelled()

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
    return reconstruct_resolved(plan, warm_start=bool(warm_start), out=out)


def reconstruct_resolved(
    plan: ResolvedReconstruction, *, warm_start: bool, out: np.ndarray | None = None
) -> Reconstruction:
    """Execute the resolved method on its original selected support and grid."""
    from tomojax.recon.api import reconstruct_arrays

    scan, method = plan.scan, plan.method
    if method not in {"fbp", "fourier"}:
        _suggest_binning(scan, "iterative reconstruction")
    with progress.runtime() as actual:
        volume, info = reconstruct_arrays(
            method,
            scan.geometry,
            scan.grid,
            scan.detector,
            scan.projections,
            config=plan.config,
            warm_start=bool(warm_start),
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
        from tomojax._field_of_view import zero_outside_cylinder

        volume = zero_outside_cylinder(volume, scan.grid, plan.cylinder, in_place=out is not None)
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
    corrections such as Stripes or Rings (geometry-dependent steps bind to
    ``frames.geometry`` automatically), not Paganin or view rejection. SIRT/OSEM
    also require matching detector and voxel planes; ``warm_start=True`` uses FBP.
    Method configurations retain the same meaning as in ``reconstruct``.
    CUDA consumes corrected rows without a corrected-projection host round trip.
    Raw reads are prefetched within bounded storage and calibration is reused.
    The context drains pending work on normal completion, early exit or failure.
    Raw frames and calibration arrays must remain unchanged inside the context.
    Cone FDK corrects host row bands in device-sized view batches and retains
    the complete acquisition's weights while reconstructing each slab.

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

    Returns
    -------
    contextlib.AbstractContextManager
        Context manager yielding owned slabs paired with their z slices in the requested grid.

    Raises
    ------
    ValueError
        If geometry, method, corrections or configuration do not support slabs.

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
):
    """Resolve method, geometry and supported corrections before opening a raw row source."""
    from tomojax.corrections.api import positive_finite, slab_steps
    from tomojax.errors import InvalidParameter
    from tomojax.geometry import ParallelGeometry
    from tomojax.geometry.api import is_cone_beam
    from tomojax.recon.api import method_config, validate_reconstruction

    cone = is_cone_beam(frames.geometry)
    cfg = method_config(method, config=config, warm_start=warm_start)
    if method not in {"fbp", "fourier", "sirt", "mlem"} or not (
        type(frames.geometry) is ParallelGeometry or (cone and method == "fbp")
    ):
        raise InvalidParameter(
            "method", "reconstruct_slabs requires cone FBP or unposed axial FBP/Fourier/SIRT/MLEM"
        )
    target = frames.geometry.grid if grid is None else grid
    validate_reconstruction(method, frames.geometry, target, cfg, slabs=True)
    bound = bind_step_geometry(corrections, frames.geometry)
    slab_steps(bound)
    positive_finite(epsilon, "epsilon")
    return target, cfg, bound, cone


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
) -> Iterator[Iterator[tuple[slice, Reconstruction]]]:
    from tomojax.corrections.api import corrected_row_source
    from tomojax.recon.api import reconstruct_slabs as array_slabs

    target, cfg, bound, cone = slab_settings(
        frames,
        method,
        grid=grid,
        config=config,
        warm_start=warm_start,
        corrections=corrections,
        epsilon=epsilon,
    )
    with (
        progress.task("reconstruct", "fdk" if cone else method, None, "slabs"),
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
            host=cone,
            dark_source=frames.dark_source,
            linear_limit=frames.linear_limit,
        ) as source,
    ):

        def read(first: int, stop: int):
            corrected = source(first, stop)
            return corrected.projections, corrected

        completed = array_slabs(
            method,
            frames.geometry,
            target,
            frames.geometry.detector,
            read,
            config=cfg,
            warm_start=bool(warm_start),
        )
        with (
            closing(completed),
            closing(_slab_results(completed, frames, target, method, cfg)) as output,
        ):
            yield output


def _slab_results(
    prepared: Iterator[ReconstructionSlab[Any]],
    frames: Frames,
    target: Grid,
    method: str,
    cfg: MethodConfig,
) -> Generator[tuple[slice, Reconstruction]]:
    """Assemble the scan and reconstruction for each completed, owned output slab."""
    from tomojax._scan import detector_window, rebuild
    from tomojax.geometry import grid_volume_origin

    origin = grid_volume_origin(target)
    with progress.outcome():
        progress.check_cancelled()
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
            progress.step("slabs", remaining=0 if last else None)
            yield result
            del result
            if not last:
                progress.check_cancelled()
            del slab, scan  # Do not retain device rows while preparing the next slab.


def _estimate_center(scan: Scan, selection: CentreSelection) -> tuple[Scan, dict[str, object]]:
    """The scan with its centre from near-opposite views, and the estimate's record.

    Pixels are unchanged; the record gives the pair diagnostics, input and output.
    The centre fixes the median sampled-pair mean shift to zero, so it aliases
    common sample motion.
    """
    report = selection.measure(scan.projections)
    geometry = report.geometry
    calibration: dict[str, object] = {
        "status": "estimated",
        "method": "opposite_view_correlation",
        "diagnostics": report.to_dict(),
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
    fit: Fit | tuple[Fit, ...] = "translations",
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
) -> Alignment:
    """Fit selected geometry parameters jointly with a reconstruction.

    ``fit`` is one name or a tuple of names: ``center``, ``tilt``, ``roll``,
    ``translations`` and ``rotations``. Centre, tilt and roll are shared by all
    views. Translations and rotations are per view; every unselected parameter
    is held at its declared value. Translations have two components for parallel
    rays and three for cone beams. Rotations are the per-view wobble of the
    rotation axis, two components; each view's angle about the axis is taken as
    recorded, since stage encoders give it and errors in it are rare. By
    default each view's corrections are free of its neighbours';
    ``motion="smooth"`` fits drift and slips instead. Per-view rotations have a
    prior on their physical size.

    ``method`` "joint" (the default) fits an image and the parameters together;
    the same solver handles parallel, laminography and cone-beam scans. Existing
    poses are preserved and the fitted correction is composed with them. The
    returned scan carries both calibrated global geometry and local poses, so
    reconstructing or saving it preserves the fitted ray geometry. "opposite-views"
    fits ``fit="center"`` alone, without an image, from near-opposite views of an
    unposed axial parallel scan (within two degrees): a quick estimate from
    row-wise gradient correlations, which can be ambiguous for repetitive
    structure and records its pair diagnostics in ``info``.

    Parameters
    ----------
    scan : Scan
        Measured line integrals with their declared acquisition geometry.
    fit : str or tuple of str, optional
        Parameters to fit. Defaults to ``"translations"``.
    grid : Grid or None, optional
        Reconstruction region and sampling. None uses a bounded working grid
        with the scan's physical field of view; an explicit grid is kept.
    region : {"auto", "bbox", "cube", "cyl"}, RegionEstimate or None, optional
        The part of the scan's grid to fit on, in place of ``grid``, as for
        :func:`reconstruct`.
    method : {"joint", "opposite-views"} or None, optional
        How the parameters are estimated; None is "joint".
    motion : {"independent", "smooth"} or None, optional
        Whether each view moves independently of its neighbours (the default) or the
        motion is smooth drift with occasional slips, for a steady stage (see
        :class:`tomojax.alignment.AlignConfig`); None keeps ``config``'s value.
    iterations, image_iterations, tolerance, interpolation, resolution : optional
        The everyday settings, replacing the fields of the same name in ``config``
        (see :class:`tomojax.alignment.AlignConfig`); None keeps its value.
    config : tomojax.alignment.AlignConfig or None, optional
        Every setting of the joint fit (imported from :mod:`tomojax.alignment`);
        None uses the defaults.

    Returns
    -------
    Alignment
        Corrected scan, its per-view poses, the record and the joint fit's working image.
        ``info["config"]`` holds the settings used.
        Each fitted global appears under its name in ``info["calibration"]`` as
        ``{"value": ..., "change": ...}``: the final value, the centre in geometry
        length units (u, or u/v for cone beams) and angles in degrees, and the
        fitted change, the centre in detector pixels and angles in degrees. Tilt
        is the signed inclination ``atan2(axis_y, axis_z)``; positive roll turns
        detector u towards v (right-handed about the detector normal, which points
        back towards the source), so a point above the detector centre projects
        towards +u. Cone segments with different calibrated values report them in
        segment order.

    Raises
    ------
    ValueError
        If the fit selection is empty, repeated or unknown, the method does not
        fit it, or the geometry has not been declared.

    Examples
    --------
    >>> import tomojax as tj
    >>> from tomojax.alignment import AlignConfig
    >>> scan = tj.load("scan.nxs")  # doctest: +SKIP
    >>> result = tj.align(scan, fit=("translations", "roll"))  # doctest: +SKIP
    >>> result = tj.align(scan, config=AlignConfig(levels=(1,)))  # doctest: +SKIP
    >>> recon = tj.reconstruct(result.scan)  # doctest: +SKIP
    """
    from tomojax._smooth_alignment import resolve_alignment

    label = ", ".join(map(str, fit)) if isinstance(fit, tuple) else str(fit)
    with progress.task("align", label, None, "fit"):
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
        return align_resolved(scan, fit, plan)


def align_resolved(
    scan: Scan, fit: str | tuple[str, ...], plan: CentreSelection | JointSelection
) -> Alignment:
    """Execute the selected centre or joint-image fit."""
    from tomojax._smooth_alignment import CentreSelection, align_poses

    if isinstance(plan, CentreSelection):
        with progress.runtime() as actual:
            alignment = _opposite_views(scan, plan)
        return replace(
            alignment, info={**alignment.info, "placement": actual.record(slabbed=False)}
        )
    with progress.runtime() as actual:
        result = align_poses(scan, fit=fit, plan=plan)
    slabbed = any(
        "host" in level["stacks"]["placement"].values()
        for level in cast("Sequence[dict[str, Any]]", result.info.get("levels", ()))
    )
    grid, cylinder = plan.grid, plan.cylinder
    assert result.scan is not None  # the alignment's corrected scan
    image = result.volume
    if cylinder is not None and image is not None and result.grid is not None:
        from tomojax._field_of_view import zero_outside_cylinder

        image = zero_outside_cylinder(image, result.grid, cylinder)
    # The fit's working grid belongs to its image; the scan keeps the caller's grid.
    corrected = with_grid(result.scan, scan.grid if grid is None else grid)
    info = {**result.info, **plan.info, "placement": actual.record(slabbed=slabbed)}
    return Alignment(corrected, info, image, result.grid)


def _opposite_views(scan: Scan, selection: CentreSelection) -> Alignment:
    """Fit the centre from near-opposite views, without an image."""
    centered, record = _estimate_center(scan, selection)
    before, after = scan.detector.center[0], centered.detector.center[0]
    center = {"value": float(after), "change": float((after - before) / scan.detector.du)}
    info = {
        "method": "opposite-views",
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
