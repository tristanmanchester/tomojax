"""TomoJAX's workflow API: scans, reconstructions and alignments.

A :class:`Scan` is projections together with the geometry that produced them;
every operation takes one and returns a result that carries its geometry, so
results can be reconstructed, aligned again or saved without restating it::

    import tomojax as tj

    scan = tj.load("scan.nxs")
    recon = tj.reconstruct(scan)  # Geometry-aware automatic reconstruction
    result = tj.align(scan, mode="cor-then-pose")
    tj.save("recon.nxs", tj.reconstruct(result.scan, method="cgls"))
"""

from __future__ import annotations

from contextlib import closing, contextmanager
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
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from os import PathLike

    import jax

    from tomojax._loading import Frames
    from tomojax._typed_arrays import Device
    from tomojax.alignment import AlignConfig, ProjectionMatchingConfig
    from tomojax.alignment.api import (
        AlignmentMode,
        AlignmentPlan,
        PoseTranslationFrame,
        QualityTier,
    )
    from tomojax.corrections import Step
    from tomojax.geometry import Detector, Geometry, Grid, ScanGeometry
    from tomojax.recon import FBPConfig, FourierConfig, MLEMConfig, SIRTConfig
    from tomojax.recon.api import MethodConfig

type Method = Literal["auto", "fbp", "fourier", "cgls", "fista", "spdhg", "sirt", "mlem"]


@dataclass(frozen=True)
class Reconstruction:
    """A reconstructed volume, the scan it came from and how it was made.

    ``volume`` is ``(nx, ny, nz)`` on :attr:`grid`, in the reciprocal of the
    geometry's length unit (a NumPy or JAX array; ``np.asarray`` gives NumPy).
    ``info`` records the method's settings.

    Parameters
    ----------
    volume : numpy.ndarray or jax.Array
        Volume in (x, y, z) order and reciprocal geometry-length units.
    scan : Scan
        Input scan, including the output grid and any inferred geometry.
    method : str
        Reconstruction method used.
    info : mapping
        Resolved method settings and solver records.
    """

    volume: np.ndarray | jax.Array
    scan: Scan
    method: str
    info: Mapping[str, object]

    @property
    def grid(self) -> Grid:
        """The grid the volume is on, the scan's.

        Returns
        -------
        Grid
            Reconstruction grid carried by the scan.
        """
        return self.scan.grid


@dataclass(frozen=True)
class Alignment:
    """The result of :func:`align`.

    ``volume`` is the reconstruction the alignment converged with. ``scan`` is
    the input scan with the estimated geometry and per-view poses applied:
    reconstruct it, align it again or save it. ``poses`` are its ``(views, 6)``
    poses (see :attr:`Scan.poses`), always a table, zero where nothing moved;
    ``info`` is the solver's record.

    Parameters
    ----------
    volume : numpy.ndarray or jax.Array
        Reconstruction used by the alignment, in (x, y, z) order.
    scan : Scan
        Scan with estimated geometry and per-view corrections applied.
    poses : numpy.ndarray
        Complete (views, 6) correction table, zero where nothing moved.
    info : mapping
        Solver record and resolved alignment settings.
    """

    volume: np.ndarray | jax.Array
    scan: Scan
    poses: np.ndarray
    info: Mapping[str, object]


# ----------------------------------------------------------------------------- files


def load_reconstruction(path: str | PathLike[str]) -> Reconstruction:
    """Load a reconstruction saved by :func:`save` or ``tomojax recon``.

    Parameters
    ----------
    path : str or path-like
        Saved reconstruction dataset.

    Returns
    -------
    Reconstruction
        Volume, source scan and stored reconstruction settings.

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
    scan = load(path)
    record = scan.source
    if record is None or record.volume is None:
        raise ValueError(f"{path} holds no reconstructed volume")
    meta = record.geometry_metadata
    info = meta.get("reconstruction_info", meta.get("alignment_info"))
    return Reconstruction(
        volume=np.asarray(record.volume),
        scan=scan,
        method=str(meta.get("reconstruction_method", "unknown")),
        info=dict(info) if isinstance(info, dict) else {},
    )


def save(path: str | PathLike[str], item: Scan | Reconstruction | Alignment) -> None:
    """Save a scan, reconstruction or alignment as a TomoJAX dataset.

    Reconstructions and alignments are saved with their scan (projections,
    geometry and any pose corrections), so the file reloads with :func:`load`
    and :func:`load_reconstruction`.

    Parameters
    ----------
    path : str or path-like
        Output dataset path; an existing file is overwritten.
    item : Scan, Reconstruction or Alignment
        Scan or result to save with its geometry and metadata.

    Returns
    -------
    None
        Write the dataset to disk.

    Raises
    ------
    TypeError
        If item is not a scan, reconstruction or alignment.
    ValueError
        If the output format is unsupported.

    Examples
    --------
    >>> import tomojax as tj
    >>> scan = tj.load("scan.nxs")  # doctest: +SKIP
    >>> tj.save("copy.nxs", scan)  # doctest: +SKIP
    """
    from tomojax.io import save_dataset

    if isinstance(item, Scan):
        record = record_of(item)
    elif isinstance(item, Reconstruction):
        record = record_of(item.scan)
        record.volume = item.volume
        record.geometry_metadata["reconstruction_method"] = item.method
        # As JSON, which the file's metadata is: a configuration as its fields, and
        # what else JSON cannot hold as text.
        record.geometry_metadata["reconstruction_info"] = json.loads(
            json.dumps(dict(item.info), default=_json_value)
        )
    elif isinstance(item, Alignment):  # pyright: ignore[reportUnnecessaryIsInstance]
        record = record_of(item.scan)
        record.volume = item.volume
        record.geometry_metadata["alignment_info"] = json.loads(
            json.dumps(dict(item.info), default=_json_value)
        )
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
    """Project a volume through the supplied geometry.

    Parameters
    ----------
    geometry : Geometry
        Per-view geometry defining the matched projector.
    volume : numpy.ndarray or jax.Array
        Volume in (x, y, z) order.
    grid : Grid or None, optional
        Voxel grid; None uses the geometry's grid.
    devices : jax.Device or sequence of jax.Device or None, optional
        Devices sharing the views; None uses the current default device.

    Returns
    -------
    jax.Array
        Line-integral projections, shape (views, detector.nv, detector.nu).

    Raises
    ------
    ValueError
        If dimensions or geometry metadata do not match the operation.

    Examples
    --------
    >>> import tomojax as tj
    >>> scan = tj.load("scan.nxs")  # doctest: +SKIP
    >>> projections = tj.project(scan.geometry, tj.reconstruct(scan).volume)  # doctest: +SKIP
    """
    from tomojax.recon import project as _project

    return _project(geometry, volume, grid=grid, devices=devices)


def backproject(
    geometry: Geometry,
    projections: np.ndarray | jax.Array,
    *,
    grid: Grid | None = None,
    devices: Device | Sequence[Device] | None = None,
) -> jax.Array:
    """Apply the exact transpose of the forward projector.

    Parameters
    ----------
    geometry : Geometry
        Per-view geometry defining the matched projector.
    projections : numpy.ndarray or jax.Array
        Line integrals in (view, v, u) order.
    grid : Grid or None, optional
        Voxel grid; None uses the geometry's grid.
    devices : jax.Device or sequence of jax.Device or None, optional
        Devices sharing the views; None uses the current default device.

    Returns
    -------
    jax.Array
        Adjoint volume, shape (grid.nx, grid.ny, grid.nz).

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
    from tomojax.recon import backproject as _backproject

    return _backproject(geometry, projections, grid=grid, devices=devices)


def reconstruct(
    scan: Scan,
    method: Method = "auto",
    *,
    grid: Grid | None = None,
    out: np.ndarray | None = None,
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

    ``auto`` (default) chooses Fourier inversion for unposed axial half-turn
    scans, the guarded laminography workflow for oblique scans, matched CGLS
    for segmented cone geometry, and FBP/FDK otherwise.
    An explicitly unrecorded axial centre may be inferred; oblique auto
    may infer setup, support and a binned grid. The scan and ``info["automatic"]``
    record those choices and limitations. Explicit methods bypass this policy.

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
        Least squares with total-variation weight ``tv_weight`` (default
        0.005), ``iterations`` (default 50), optionally ``nonnegative``.
    ``sirt``
        SIRT, OS-SART, bounds and support via :class:`~tomojax.recon.SIRTConfig`.
    ``spdhg``
        The same objective as ``fista`` by stochastic primal-dual updates:
        ``iterations`` (default 400) each update one block of views, in an
        order drawn with ``seed``.

    ``mlem`` is linear Poisson MLEM/OSEM, not transmission counts (``MLEMConfig``).
    ``warm_start`` uses FBP/FDK for least squares; ``grid`` overrides the scan's output grid.

    ``out`` is a writable float32 array or memmap of shape ``(grid.nx, grid.ny,
    grid.nz)`` for ``fbp`` and ``fourier``: the volume is made in host slabs
    straight into it and returned as ``volume``. It must not overlap the scan,
    and a later failure can leave it partly written.

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
    out : numpy.ndarray or None, optional
        Writable float32 output or memmap for fbp/fourier; must not overlap input.
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
    if method == "auto":
        options = {"grid": grid, "out": out, "config": config, "warm_start": warm_start, **given}
        return _automatic_reconstruct(scan, {k: v for k, v in options.items() if v is not None})
    cfg = method_config(method, config=config, **given)
    if warm_start is not None and method in {"fbp", "fourier"}:
        raise ValueError(f"method {method!r} does not take warm_start")
    if grid is not None:
        scan = with_grid(scan, grid)
    if method not in {"fbp", "fourier"}:
        _suggest_binning(scan, "iterative reconstruction")
    volume, info = reconstruct_arrays(
        method,
        scan.geometry,
        scan.grid,
        scan.detector,
        scan.projections,
        config=cfg,
        warm_start=bool(warm_start),
        out=out,
    )
    return Reconstruction(volume=volume, scan=scan, method=method, info=info)


def _automatic_reconstruct(scan: Scan, options: dict[str, Any]) -> Reconstruction:
    """Select an inverse without changing an explicit scan's acquisition geometry."""
    from tomojax.geometry import (
        ConeSegments,
        LaminographyGeometry,
        ParallelGeometry,
        RotationAxisGeometry,
    )

    base: Any = scan.geometry
    while hasattr(base, "base"):
        base = base.base
    acquisition = (
        {} if scan.source is None else scan.source.geometry_metadata.get("acquisition", {})
    )
    unknown_axis = (
        isinstance(acquisition, dict) and acquisition.get("geometry_status") == "not_recorded"
    )
    if isinstance(base, LaminographyGeometry | RotationAxisGeometry) or unknown_axis:
        if options:
            raise ValueError(
                "automatic geometry reconstruction does not take expert overrides; "
                "choose method='fista' (or another explicit method) to supply them"
            )
        from tomojax._laminography import reconstruct_laminography

        return reconstruct_laminography(scan, align=align, reconstruct=reconstruct)
    reasons = []
    method: Method = "fbp"
    if isinstance(base, ConeSegments):
        method = "cgls"
        reasons.append(
            "Segmented cone geometry uses matched CGLS because FDK requires one arrangement; "
            "iteration count and image quality are not automatically validated."
        )
    elif type(scan.geometry) is ParallelGeometry:
        angles = np.unique(np.mod(np.asarray(scan.angles, np.float64), 180.0))
        gaps = np.diff(np.r_[angles, angles[0] + 180])
        if len(angles) >= 16 and np.max(gaps) <= max(22.5, 2 * float(np.median(gaps))):
            method = "fourier"
        else:
            reasons.append(
                "Sparse or incomplete half-turn coverage: Fourier inversion was not selected."
            )
    elif getattr(base, "beam", None) is None:
        reasons.append(
            "Posed or custom parallel geometry uses FBP; "
            "automatic iterative validation is unavailable."
        )
    scan, center_estimation, center_calibration_reason = _automatic_axial_setup(
        scan, method=method, reasons=reasons
    )
    result = reconstruct(scan, method, **options)
    if result.info.get("backend") == "numpy":
        reasons.append("CUDA Fourier is not active; NumPy execution cannot promise GPU throughput.")
    automatic = {
        "method": method,
        "status": "limited" if reasons else "ok",
        "reasons": reasons,
        "center_calibration_reason": center_calibration_reason,
        "geometry": (
            "center_estimated"
            if center_estimation and center_estimation["status"] == "estimated"
            else "preserved"
        ),
        "center_estimation": center_estimation,
        "warning": "Method selection does not validate geometry, support or detector artefacts.",
    }
    return replace(result, info={**result.info, "automatic": automatic})


def _automatic_axial_setup(
    scan: Scan, *, method: Method, reasons: list[str]
) -> tuple[Scan, dict[str, Any] | None, str | None]:
    """Infer an absent axial centre and identify unresolved raw-workflow limitations."""
    center_estimation = None
    source_meta = {} if scan.source is None else scan.source.geometry_metadata
    if method == "fourier" and source_meta.get("detector_center_status") == "not_recorded":
        try:
            scan = estimate_center(scan)
            assert scan.source is not None
            source_meta = scan.source.geometry_metadata
            center_estimation = source_meta["detector_center_calibration"]
        except ValueError as exc:
            center_estimation = {"status": "failed", "reason": str(exc)}
            reasons.append("Automatic centre estimation failed: " + str(exc))
    center_calibration_reason = None
    if method == "fourier" and any(step.name == "log" for step in scan.corrections):
        if source_meta.get("detector_center_status") not in {"inferred", "recorded", "provided"}:
            center_calibration_reason = (
                "Reconstruction did not perform centre calibration; the supplied geometry was "
                "preserved. Use align if the centre needs estimating."
            )
            reasons.append(center_calibration_reason)
        reasons.append(
            "Corrected projections were already materialized; use reconstruct_slabs on raw "
            "Frames to avoid that host round trip. Raw-workflow speed is not guaranteed here."
        )
        if not any(step.name == "stripes" for step in scan.corrections):
            reasons.append(
                "No stripe correction is recorded; detector-ring suppression is unassessed."
            )
    return scan, center_estimation, center_calibration_reason


@contextmanager
def reconstruct_slabs(
    frames: Frames,
    method: Literal["fbp", "fourier", "sirt", "mlem"],
    *,
    grid: Grid | None = None,
    corrections: Sequence[Step] = (),
    config: FBPConfig | FourierConfig | SIRTConfig | MLEMConfig | None = None,
    warm_start: bool | None = None,
    epsilon: float = 1e-6,
) -> Iterator[Iterator[tuple[slice, Reconstruction]]]:
    """Correct raw frames and reconstruct bounded axial slabs.

    Use ``with reconstruct_slabs(frames, "fbp", ...) as slabs:`` and iterate
    over ``(z_slice, reconstruction)``. The slice indexes the requested output
    grid; each reconstruction retains its physical global grid, genuine corrected
    source rows (including interpolation neighbors), and correction records.
    Volumes are owned float32 host arrays. Results remain valid after advancing
    or closing the iterator; collecting them retains their volume and projections.

    This supports unposed axial parallel beams and whole-row line-integral
    corrections such as Stripes, not Paganin or view rejection. SIRT/OSEM also
    require matching detector and voxel planes; ``warm_start=True`` uses FBP.
    Method configurations retain the same meaning as in ``reconstruct``.
    CUDA consumes corrected rows without a corrected-projection host round trip.
    Raw reads are prefetched within bounded storage and calibration is reused.
    The context drains pending work on normal completion, early exit or failure.
    Raw frames and calibration arrays must remain unchanged inside the context.

    Parameters
    ----------
    frames : Frames
        Raw detector frames for unposed axial parallel-beam geometry.
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

    Yields
    ------
    iterator of tuple of (slice, Reconstruction)
        Context-managed slab iterator; each z slice indexes the requested grid.

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
    from tomojax._scan import detector_window, rebuild
    from tomojax.corrections import corrected_row_source
    from tomojax.geometry import ParallelGeometry, grid_volume_origin
    from tomojax.recon.api import method_config, reconstruct_slabs as array_slabs

    cfg = method_config(method, config=config)
    if (
        method not in {"fbp", "fourier", "sirt", "mlem"}
        or type(frames.geometry) is not ParallelGeometry
    ):
        raise ValueError(
            "reconstruct_slabs requires FBP/Fourier/SIRT/MLEM and unposed axial ParallelGeometry"
        )
    if warm_start is not None and method in {"fbp", "fourier"}:
        raise ValueError(f"method {method!r} does not take warm_start")
    target = frames.geometry.grid if grid is None else grid
    origin = grid_volume_origin(target)
    with corrected_row_source(
        frames.counts,
        flats=frames.flats,
        flat_positions=frames.flat_positions,
        darks=frames.darks,
        white_level=frames.white_level,
        steps=corrections,
        epsilon=epsilon,
        view_positions=frames.view_positions,
    ) as source:

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
        with closing(completed) as prepared:

            def results():
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
                    scan = Scan(
                        slab.projections, geometry, frames.name, slab.context.records, frames.source
                    )
                    info = {
                        "config": cfg,
                        "host_slabs": True,
                        "source_rows": (slab.rows.start, slab.rows.stop),
                        **slab.info,
                    }
                    yield (
                        slice(slab.start, slab.stop),
                        Reconstruction(slab.volume, scan, method, info),
                    )
                    del slab, scan  # Do not retain device rows while preparing the next slab.

            with closing(results()) as output:
                yield output


def estimate_center(scan: Scan) -> Scan:
    """Estimate a parallel scan's rotation centre without reconstructing a volume.

    Returns the scan with its absolute physical detector-u centre set, available as
    ``result.detector.center[0]``; no pixels are shifted. For a cropped or binned
    preview, add ``result.detector.center[0] - preview.detector.center[0]`` to the
    full scan's centre rather than copying the preview's. Needs corrected line
    integrals of an unposed axial :class:`ParallelGeometry` over a half turn (at least
    16 views spanning 175 degrees); anything else raises ``ValueError``. The estimate
    can be wrong for very compact objects, so check the reconstruction. Source
    provenance, when present, records the inferred centre without changing the input.

    Parameters
    ----------
    scan : Scan
        Unposed axial parallel line integrals spanning at least 175 degrees.

    Returns
    -------
    Scan
        Scan with the inferred absolute physical detector-u centre; pixels are unchanged.

    Raises
    ------
    ValueError
        If geometry or angular coverage is unsupported, or fewer than 16 views exist.

    Examples
    --------
    >>> import tomojax as tj
    >>> scan = tj.estimate_center(tj.load("scan.nxs"))  # doctest: +SKIP
    >>> recon = tj.reconstruct(scan)  # doctest: +SKIP
    """
    from tomojax.alignment.api import estimate_center as estimate_geometry_center

    geometry = estimate_geometry_center(cast("Geometry", scan.geometry), scan.projections)
    source = scan.source
    if source is not None:
        source = replace(
            source,
            geometry_metadata={
                **source.geometry_metadata,
                "detector_center_status": "inferred",
                "detector_center_calibration": {
                    "status": "estimated",
                    "method": "half_turn_fourier",
                    "input": list(scan.detector.center),
                    "output": list(geometry.detector.center),
                    "unit": "geometry_length",
                },
            },
        )
    return replace(scan, geometry=geometry, source=source)


def align(
    scan: Scan,
    *,
    mode: AlignmentMode | Literal["auto"] = "auto",
    quality: QualityTier = "fast",
    levels: Iterable[int] | None = None,
    freeze: Iterable[str] = (),
    grid: Grid | None = None,
    checkpoint: str | PathLike[str] | None = None,
    config: AlignConfig | ProjectionMatchingConfig | None = None,
    devices: Device | Sequence[Device] | None = None,
) -> Alignment:
    """Estimate ``scan``'s geometry corrections and reconstruct with them.

    ``mode="auto"`` selects centre estimation for axial parallel scans and
    guarded setup/support/motion selection for unposed laminography. Its
    conditional outcome and limitations are in ``info["automatic"]``.
    Centre estimation records its input/output physical detector coordinates;
    its success resolves only the reconstruction-only calibration limitation.
    Explicit ``mode`` is ``pose`` (per-view motion), ``cor`` (detector
    centre, or a cone beam's axis offset and detector roll), ``cor-then-pose``
    or ``full`` (setup geometry, then motion, coarse to fine); see
    :mod:`tomojax.alignment`. ``quality`` is ``fast`` or ``reference`` (slower,
    more conservative). ``freeze`` keeps named parameters fixed (for example
    ``("dy",)``); ``levels`` sets the coarse-to-fine binning factors.
    ``config`` (a :class:`tomojax.alignment.AlignConfig`) holds expert solver
    settings and replaces the quality profile's defaults.
    ``ProjectionMatchingConfig`` instead selects native-texture, translation-only
    joint refinement for unposed axial scans, without resampling the input.
    It may fit noise or model mismatch as motion; inspect poses and image quality.
    Its image uses Fourier. Checkpoints and level/quality overrides are unsupported.
    A ``spectral_setup`` initializer does not reconstruct; reconstruct its returned scan.

    ``grid`` overrides the scan's reconstruction grid, as :func:`reconstruct` does.

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

    Parameters
    ----------
    scan : Scan
        Measured line integrals with their acquisition geometry.
    mode : str, optional
        Alignment mode: auto, pose, cor, cor-then-pose or full.
    quality : {'fast', 'reference'}
        Quality profile; reference is slower and more conservative.
    levels : iterable of int or None, optional
        Coarse-to-fine binning factors; None uses the mode's defaults.
    freeze : iterable of str, optional
        Names of parameters to keep fixed, such as ("dy",).
    grid : Grid or None, optional
        Alignment and reconstruction grid; None keeps the scan's grid.
    checkpoint : str, path-like or None, optional
        Progress file; compatible existing checkpoints resume automatically.
    config : AlignConfig or None, optional
        Expert solver settings, replacing the quality profile's defaults.
    devices : jax.Device or sequence of jax.Device or None, optional
        Devices sharing supported pose and reconstruction updates.

    Returns
    -------
    Alignment
        Reconstruction, corrected scan, full pose table and solver records.

    Raises
    ------
    ValueError
        If options, geometry or an existing checkpoint are incompatible.

    Examples
    --------
    >>> import tomojax as tj
    >>> scan = tj.load("scan.nxs")  # doctest: +SKIP
    >>> result = tj.align(scan, mode="cor-then-pose")  # doctest: +SKIP
    >>> recon = tj.reconstruct(result.scan)  # doctest: +SKIP
    """
    if config is not None:
        from tomojax.alignment import ProjectionMatchingConfig

        if isinstance(config, ProjectionMatchingConfig):
            if mode not in {"auto", "pose"} or levels is not None or tuple(freeze):
                raise ValueError("projection matching takes mode='pose', without levels or freeze")
            if checkpoint is not None or quality != "fast":
                raise ValueError(
                    "projection matching does not take checkpoint or quality overrides"
                )
            return _align_projection_matching(scan, config=config, grid=grid, devices=devices)
    if mode == "auto":
        freeze = tuple(freeze)
        explicit = config is not None or levels is not None or bool(freeze)
        explicit = explicit or grid is not None or checkpoint is not None or quality != "fast"
        if not explicit and describe(scan.geometry)[0] != "cone":
            from tomojax.core.devices import sharing

            with sharing(devices):
                return _align_automatically(scan)
        if explicit and describe(scan.geometry)[0] != "cone":
            from tomojax._laminography import uses_laminography_workflow

            if uses_laminography_workflow(scan):
                raise ValueError(
                    "automatic laminography needs an explicit mode for expert overrides"
                )
        mode = "pose"
    return _align_explicit(
        scan,
        mode=mode,
        quality=quality,
        levels=levels,
        freeze=freeze,
        grid=grid,
        checkpoint=checkpoint,
        config=config,
        devices=devices,
    )


def _align_projection_matching(
    scan: Scan,
    *,
    config: ProjectionMatchingConfig,
    grid: Grid | None,
    devices: Device | Sequence[Device] | None,
) -> Alignment:
    from tomojax.alignment.api import match_projections
    from tomojax.core.devices import sharing

    if grid is not None:
        scan = with_grid(scan, grid)
    with sharing(devices):
        poses, info = match_projections(scan.geometry, scan.projections, config=config)
        corrected = scan.with_poses(poses)
        volume = np.asarray(reconstruct(corrected, "fourier").volume)
    return Alignment(volume, corrected, poses, {**info, "reconstruction": "fourier"})


def _align_explicit(
    scan: Scan,
    *,
    mode: AlignmentMode,
    quality: QualityTier,
    levels: Iterable[int] | None,
    freeze: Iterable[str],
    grid: Grid | None,
    checkpoint: str | PathLike[str] | None,
    config: AlignConfig | None,
    devices: Device | Sequence[Device] | None,
) -> Alignment:
    from tomojax.alignment.api import alignment_plan
    from tomojax.geometry import ConeSegments

    if grid is not None:
        scan = with_grid(scan, grid)
    plan = alignment_plan(
        mode,
        scan.grid,
        quality=quality,
        geometry=scan.geometry,
        levels=levels,
        freeze=freeze,
        config=config,
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


def _align_automatically(scan: Scan) -> Alignment:
    """Geometry-aware policy; every numerical operation below has an explicit mode."""
    from tomojax._laminography import reconstruct_laminography, uses_laminography_workflow

    if scan.poses is not None:
        return align(scan, mode="pose")
    if uses_laminography_workflow(scan):
        result = reconstruct_laminography(scan, align=align, reconstruct=reconstruct)
        poses = result.scan.poses
        if poses is None:
            poses = np.zeros((len(result.scan.angles), 6), np.float32)
        return Alignment(result.volume, result.scan, np.asarray(poses), result.info)
    automatic: dict[str, object] = {"status": "ok", "method": "center", "reasons": []}
    center: dict[str, object] = {
        "input": list(scan.detector.center),
        "unit": "geometry_length",
    }
    try:
        scan = estimate_center(scan)
        center.update(status="estimated", output=list(scan.detector.center))
    except ValueError as exc:
        center.update(status="failed", reason=str(exc))
        automatic.update(status="limited", reasons=[str(exc)])
    automatic["center_estimation"] = center
    result = reconstruct(scan)
    reconstruction = dict(cast("Mapping[str, Any]", result.info.get("automatic", {})))
    automatic["reconstruction"] = reconstruction
    reasons = list(reconstruction.get("reasons", []))
    if center["status"] == "estimated":
        # Resolve only the named calibration issue in the composed workflow;
        # nested reconstruction context still describes that operation alone.
        resolved = reconstruction.get("center_calibration_reason")
        reasons = [reason for reason in reasons if reason != resolved]
    if reconstruction.get("status") == "limited" and (reasons or not reconstruction.get("reasons")):
        automatic["status"] = "limited"
    automatic["reasons"] = [*cast("list[str]", automatic["reasons"]), *reasons]
    return Alignment(
        result.volume,
        result.scan,
        np.zeros((len(scan.angles), 6), np.float32),
        {**result.info, "automatic": automatic},
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
    "estimate_center",
    "load",
    "load_reconstruction",
    "project",
    "reconstruct",
    "save",
]
