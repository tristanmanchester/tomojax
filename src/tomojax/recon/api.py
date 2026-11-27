"""Expert reconstruction helpers: method configuration, array and slab solvers.

The solvers and their configuration classes are exported by
:mod:`tomojax.recon`; :func:`tomojax.reconstruct` is the workflow entry point.
"""

from __future__ import annotations

from dataclasses import fields, replace
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from tomojax import recon
from tomojax.errors import InvalidParameter
from tomojax.recon._fbp_storage import fbp_volume, fdk_config

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    import jax.numpy as jnp
    import numpy as np

    from tomojax.geometry import Detector, Geometry, Grid, ParallelGeometry, ScanGeometry
    from tomojax.recon._operators import Projector, backproject, project
    from tomojax.recon.cgls import CGLSConfig
    from tomojax.recon.fbp import FBPConfig, default_fbp_scale
    from tomojax.recon.filters import clear_filter_caches
    from tomojax.recon.fista_tv import FISTAConfig
    from tomojax.recon.mlem import MLEMConfig
    from tomojax.recon.sirt import SIRTConfig
    from tomojax.recon.spdhg_tv import SPDHGConfig
    from tomojax.recon.types import ReconstructionSlab

type MethodConfig = (
    recon.FBPConfig
    | recon.FourierConfig
    | recon.CGLSConfig
    | recon.FISTAConfig
    | recon.SPDHGConfig
    | recon.SIRTConfig
    | recon.MLEMConfig
)

_CONFIGS = {
    "fbp": "FBPConfig",
    "fourier": "FourierConfig",
    "cgls": "CGLSConfig",
    "fista": "FISTAConfig",
    "spdhg": "SPDHGConfig",
    "sirt": "SIRTConfig",
    "mlem": "MLEMConfig",
}

_SOURCES = {
    "Projector": "_operators",
    "backproject": "_operators",
    "clear_filter_caches": "filters",
    "default_fbp_scale": "fbp",
    "project": "_operators",
}


def __getattr__(name: str) -> Any:
    source = _SOURCES.get(name)
    if source is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"tomojax.recon.{source}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


def method_config(
    method: str,
    *,
    config: MethodConfig | None = None,
    warm_start: bool | None = None,
    **settings: object,
) -> MethodConfig:
    """Build a method configuration from defaults or named replacements.

    ``fbp`` (FDK for cone beams) takes an :class:`FBPConfig`, ``cgls`` a
    :class:`CGLSConfig`, ``fista`` a :class:`FISTAConfig` and ``spdhg`` an
    :class:`SPDHGConfig`; ``sirt`` takes :class:`SIRTConfig`, ``mlem``
    :class:`MLEMConfig` and ``fourier`` :class:`FourierConfig`. Each setting
    replaces the field of its name, as :func:`dataclasses.replace` does. A config
    of another class, or a setting its class has no field for, raises
    :class:`ValueError` naming the valid ones; so does ``warm_start`` for the
    direct methods (fbp, fourier), which start from nothing.

    Parameters
    ----------
    method : str
        Reconstruction method: "fbp", "fourier", "cgls", "fista", "spdhg",
        "sirt" or "mlem".
    config : MethodConfig or None, optional
        Existing configuration of the method's type. None uses its defaults.
    warm_start : bool or None, optional
        Whether the caller asked for a warm start; only the iterative methods take one.
    **settings : object
        Named fields replacing the corresponding configuration values.

    Returns
    -------
    MethodConfig
        Configuration for the requested method with replacements applied.

    Raises
    ------
    ValueError
        If method is unknown, config has the wrong type or a setting is unknown.

    Examples
    --------
    >>> from tomojax.recon.api import method_config
    >>> method_config("cgls", iterations=20).iterations
    20
    """
    name = _CONFIGS.get(method)
    if name is None:
        raise InvalidParameter(
            "method", f"method must be one of {', '.join(_CONFIGS)}; got {method!r}"
        )
    kind: type[MethodConfig] = getattr(recon, name)
    if warm_start and method in {"fbp", "fourier"}:
        raise InvalidParameter("warm_start", f"method {method!r} does not take warm_start")
    if config is not None and not isinstance(config, kind):
        raise InvalidParameter(
            "config", f"method {method!r} takes a {kind.__name__}, not a {type(config).__name__}"
        )
    resolved_settings = dict(settings)
    if "nonnegative" in resolved_settings:
        nonnegative = resolved_settings.pop("nonnegative")
        if method not in {"fista", "spdhg", "sirt"}:
            detail = (
                "mlem is intrinsically nonnegative"
                if method == "mlem"
                else (
                    "cgls is unconstrained" if method == "cgls" else "direct methods take no bounds"
                )
            )
            raise InvalidParameter(
                "nonnegative", f"method {method!r} does not take nonnegative ({detail})"
            )
        if nonnegative is not None:
            current = (
                None
                if config is None
                else cast("float | None", getattr(config, "lower_bound", None))
            )
            if nonnegative:
                resolved_settings["lower_bound"] = 0.0 if current is None else max(0.0, current)
            elif "lower_bound" not in resolved_settings and current == 0.0:
                resolved_settings["lower_bound"] = None
    names = sorted(item.name for item in fields(kind) if item.init)
    unknown = sorted(set(resolved_settings) - set(names))
    if unknown:
        raise InvalidParameter(
            unknown[0],
            f"method {method!r} does not take {', '.join(unknown)} "
            f"({kind.__name__} has {', '.join(names)})",
        )
    base = kind() if config is None else config
    return _replace_settings(base, resolved_settings) if resolved_settings else base


def _replace_settings(base: MethodConfig, settings: dict[str, object]) -> MethodConfig:
    """Name an invalid field where its value came from in the operation."""
    try:
        return replace(base, **settings)
    except InvalidParameter as err:
        if err.path in settings:
            raise
        raise InvalidParameter(f"config.{err.path}", str(err)) from err


def solver_name(method: str, geometry: Geometry) -> str:
    """Return the method as users name it: FDK for a cone-beam FBP.

    Parameters
    ----------
    method : str
        Reconstruction method, as :func:`method_config` takes it.
    geometry : Geometry
        Acquisition the method reconstructs.

    Returns
    -------
    str
        ``"fdk"`` for FBP of a cone-beam geometry, otherwise ``method``.
    """
    from tomojax.core.geometry.cone import is_cone_beam

    return "fdk" if method == "fbp" and is_cone_beam(geometry) else method


class AngularCoverageError(InvalidParameter):
    """The resolved cone acquisition does not cover the angles needed for FDK."""


def validate_reconstruction(
    geometry: ScanGeometry,
    grid: Grid,
    config: MethodConfig,
    *,
    warm_start: bool = False,
    axial_index: int | None = None,
    projections: jnp.ndarray | np.ndarray | None = None,
    check_samples: bool = True,
) -> None:
    """Check resolved solver settings against the actual acquisition before dispatch.

    Parameters
    ----------
    geometry : ScanGeometry
        Calibrated views and detector.
    grid : Grid
        Working volume grid.
    config : MethodConfig
        Settings for the resolved method; its class names the method.
    warm_start : bool, optional
        Initialize an iterative solve with filtered backprojection.
    axial_index : int or None, optional
        Requested axial plane when validating a slice or sweep.
    projections : array-like or None, optional
        Projection array or layout for storage and device placement checks.
    check_samples : bool, optional
        Inspect measurement values when projections contain acquired samples.

    Raises
    ------
    InvalidParameter
        If the method, support, view subsets or slab grid are incompatible.
    """
    from tomojax.core.geometry.cone import is_cone_beam
    from tomojax.core.validation import (
        validate_angles,
        validate_optional_broadcastable_shape,
        validate_optional_same_shape,
    )

    method = config.method
    validate_angles(geometry, method)
    _check_solver_inputs(geometry, config, projections, check_samples=check_samples)
    views = len(geometry.angles)
    if subsets := getattr(config, "subsets", None):
        from tomojax.errors import within
        from tomojax.recon._solver_validation import check_subset_count

        with within("config"):
            check_subset_count(method, subsets, views)
    try:
        check_support = (
            validate_optional_same_shape
            if config.method == "sirt"
            else validate_optional_broadcastable_shape
        )
        _ = check_support(
            getattr(config, "support", None),
            grid.shape,
            context=method,
            name="support",
            fix="use a support mask broadcastable to shape (grid.nz, grid.ny, grid.nx).",
        )
    except ValueError as err:
        raise InvalidParameter("config.support", str(err)) from err
    if (warm_start or config.method == "fbp") and is_cone_beam(geometry):
        from tomojax.recon.fdk import check_angular_coverage

        if config.method == "fbp":
            _ = fdk_config(config)
        try:
            check_angular_coverage(geometry, geometry.detector, views)
        except ValueError as err:
            raise AngularCoverageError("warm_start" if warm_start else "method", str(err)) from err
    if config.method == "fista" and projections is not None:
        from tomojax.recon._fista_host import check_fista_storage

        check_fista_storage(geometry, grid, projections, config, warm_start=warm_start)
    if config.method == "fbp" and projections is not None and check_samples:
        from tomojax.recon._fbp_storage import check_host_fbp_samples

        check_host_fbp_samples(
            geometry,
            grid,
            geometry.detector,
            projections,
            config,
            axial_index=axial_index,
        )
    if config.method == "fourier":
        from tomojax.recon.fourier import check_fourier_inputs

        check_fourier_inputs(geometry, grid, projections, check_samples=check_samples)


def _check_solver_inputs(
    geometry: ScanGeometry,
    config: MethodConfig,
    projections: jnp.ndarray | np.ndarray | None,
    *,
    check_samples: bool,
) -> None:
    """Check interpolation, accelerator placement and measured values for this inverse."""
    from tomojax.core.geometry.cone import is_cone_beam

    method = config.method
    if method != "fourier":
        from tomojax.core.geometry.views import host_view_poses, stack_view_poses
        from tomojax.core.validation import validate_physical_poses, validate_pose_stack

        views = len(geometry.angles)
        try:
            host = host_view_poses(geometry, views)
            poses = stack_view_poses(geometry, views) if host is None else host
            _ = validate_pose_stack(poses, views, context=method)
        except ValueError as err:
            raise InvalidParameter("source", str(err)) from err
        validate_physical_poses(poses, method, rigid=method != "fbp" and not is_cone_beam(geometry))
    if config.method == "cgls":
        from tomojax.recon.cgls import check_cgls_geometry

        check_cgls_geometry(geometry, config.interpolation)
    if config.backend == "cuda":
        import jax

        from tomojax.recon._backend import reconstruction_backend

        cone = is_cone_beam(geometry)
        _ = reconstruction_backend(
            config.backend,
            cone=cone,
            context=solver_name(method, geometry),
            devices=projections.devices()
            if method == "fbp" and isinstance(projections, jax.Array)
            else None,
        )
    if method not in {"fourier", "fbp"} and projections is not None and check_samples:
        from tomojax.recon._host_arrays import check_measurements

        check_measurements(projections, method)
    if config.backend == "cupy":
        from tomojax.recon._fourier_backend import resolve_fourier_backend

        _ = resolve_fourier_backend(config.backend)


def validate_slab_grid(
    config: MethodConfig, geometry: Geometry, grid: Grid, detector: Detector
) -> None:
    """Check that iterative raw reconstruction uses independent axial planes.

    Parameters
    ----------
    config : MethodConfig
        Settings of the slab method.
    geometry : Geometry
        Acquisition geometry.
    grid : Grid
        Requested output grid.
    detector : Detector
        Detector measuring the raw projections.

    Raises
    ------
    InvalidParameter
        If a voxel plane depends on multiple detector rows in an iterative solve.
    """
    if config.method in {"sirt", "mlem"} and not getattr(geometry, "detector_roll", 0):
        from tomojax.recon._axial_slabs import independent_planes

        if not independent_planes(geometry, grid, detector):
            raise InvalidParameter(
                "grid",
                "iterative slabs require independent axial Joseph planes "
                "(matching grid and detector rows)",
            )


def reconstruct_slabs[T](
    geometry: ScanGeometry,
    grid: Grid,
    detector: Detector,
    read: Callable[[int, int], tuple[jnp.ndarray | np.ndarray, T]],
    *,
    config: MethodConfig,
    warm_start: bool = False,
    info: dict[str, object] | None = None,
) -> Generator[ReconstructionSlab[T], None, None]:
    """Reconstruct owned axial slabs from bounded corrected detector-row reads.

    Parameters
    ----------
    geometry : Geometry
        Cone-beam FDK or unposed axial parallel-beam geometry.
    grid : Grid
        Output grid.
    detector : Detector
        Detector of the corrected rows.
    read : callable
        ``read(first_row, stop_row)`` returning corrected rows and the producer's
        context for that interval.
    config : MethodConfig
        Configuration of an FBP, Fourier, SIRT or MLEM reconstruction, as for
        ``tomojax.reconstruct``; its class names the method.
    warm_start : bool, optional
        Start SIRT/OSEM from FBP. Default is False.
    info : dict or None, optional
        Destination for the completed solver diagnostics.

    Yields
    ------
    ReconstructionSlab
        Owned slabs in z order, each with its source interval and context.

    Raises
    ------
    ValueError
        If the method or a setting does not support slabs.
    """
    from ._truncation import ChordSample, ScanSample

    if info is None:
        yield from _solve_slabs(
            geometry, grid, detector, read, config, warm_start=warm_start, info=None
        )
        return

    sample = ChordSample.for_band(
        geometry, detector, len(geometry.angles), 0, detector.nv
    ) or ScanSample(geometry, detector)

    def sampled(first: int, stop: int) -> tuple[np.ndarray | jnp.ndarray, T]:
        data, context = read(first, stop)
        sample.add(first, data)
        return data, context

    diagnostics: dict[str, object] = {}
    yield from _solve_slabs(
        geometry, grid, detector, sampled, config, warm_start=warm_start, info=diagnostics
    )
    truncation = sample.record(lambda first, stop: read(first, stop)[0])
    info.update(truncation=truncation, config=config, host_slabs=True)
    info.update(diagnostics)


def _solve_slabs[T](
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    read: Callable[[int, int], tuple[jnp.ndarray | np.ndarray, T]],
    config: MethodConfig,
    *,
    warm_start: bool,
    info: dict[str, object] | None,
) -> Generator[ReconstructionSlab[T], None, None]:
    cfg = method_config(config.method, config=config, warm_start=warm_start or None)
    from tomojax.geometry import ParallelGeometry

    if (
        cfg.method in {"fbp", "sirt", "mlem"}
        and isinstance(geometry, ParallelGeometry)
        and geometry.detector_roll
    ):
        return _coupled_detector_volume(
            geometry, grid, detector, read, cfg, warm_start=warm_start, diagnostics=info
        )

    def measured(
        first: int, stop: int
    ) -> tuple[np.ndarray | jnp.ndarray, tuple[T, np.ndarray | jnp.ndarray]]:
        from ._truncation import truncation_values

        data, context = read(first, stop)
        return data, (context, truncation_values(geometry, detector, data, first=first))

    match cfg.method:
        case "fbp":
            from tomojax.core.geometry.cone import is_cone_beam

            if is_cone_beam(geometry):
                from .fdk import fdk_slabs

                slabs = fdk_slabs(geometry, grid, detector, measured, config=fdk_config(cfg))
                return _with_truncation(slabs)
            from .fbp_host import fbp_slabs

            slabs = fbp_slabs(_parallel(geometry), grid, detector, measured, config=cfg)
            return _with_truncation(slabs)
        case "fourier":
            from ._fourier_stream import fourier_slabs

            slabs = fourier_slabs(
                _parallel(geometry), grid, detector, measured, config=cfg, info=info
            )
            return _with_truncation(slabs)
        case "sirt" | "mlem":
            from ._iterative_slabs import iterative_slabs

            return iterative_slabs(
                _parallel(geometry),
                grid,
                detector,
                read,
                config=cfg,
                warm_start=warm_start,
                diagnostics=info,
            )
        case _:
            raise ValueError("reconstruct_slabs supports FBP/Fourier/SIRT/MLEM")


def _with_truncation[T](
    slabs: Generator[ReconstructionSlab[tuple[T, np.ndarray | jnp.ndarray]], None, None],
) -> Generator[ReconstructionSlab[T], None, None]:
    """Give each analytic slab the truncation record of its rows, as a whole volume has."""
    from contextlib import closing

    from ._truncation import record_values
    from .types import ReconstructionSlab

    with closing(slabs):
        for slab in slabs:
            context, values = slab.context
            slab.info["truncation"] = record_values(values)
            yield ReconstructionSlab(
                slab.start, slab.stop, slab.rows, slab.projections, slab.volume, context, slab.info
            )


def _parallel(geometry: Geometry) -> ParallelGeometry:
    from tomojax.geometry import ParallelGeometry

    if not isinstance(geometry, ParallelGeometry):
        raise ValueError("reconstruct_slabs requires unposed axial ParallelGeometry")
    return geometry


def _coupled_detector_volume[T](
    geometry: ParallelGeometry,
    grid: Grid,
    detector: Detector,
    read: Callable[[int, int], tuple[jnp.ndarray | np.ndarray, T]],
    config: MethodConfig,
    *,
    warm_start: bool,
    diagnostics: dict[str, object] | None,
) -> Generator[ReconstructionSlab[T], None, None]:
    """Oblique filter lines cross rows. Joseph updates couple neighboring voxel planes."""
    import numpy as np

    from tomojax.core import checkpoint

    from .types import ReconstructionSlab

    with checkpoint.slabs() as finished:
        if finished.prefix == grid.nz:
            if diagnostics is not None:
                diagnostics.update(finished.reports[0, grid.nz])
            return
        data, context = read(0, detector.nv)
        with checkpoint.scope("z0"):
            volume, info = reconstruct_arrays(
                config.method, geometry, grid, detector, data, config=config, warm_start=warm_start
            )
        yield ReconstructionSlab(
            0, grid.nz, slice(0, detector.nv), data, np.asarray(volume), context, info
        )
        finished.add(0, grid.nz, info=info)
        if diagnostics is not None:
            diagnostics.update(finished.reports[0, grid.nz])


def reconstruct_arrays(
    method: str,
    geometry: ScanGeometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    *,
    config: MethodConfig | None = None,
    warm_start: bool = False,
    out: np.ndarray | None = None,
) -> tuple[jnp.ndarray | np.ndarray, dict[str, object]]:
    """Reconstruct projection arrays and return their solver record.

    ``config`` is the method's configuration (see :func:`method_config`).
    ``warm_start`` starts the iterative methods from the FBP
    reconstruction, with calibrated per-view cone arrangements retained. The
    whole cone scan must cover an FDK-compatible arc. Returns the volume and its
    record: the ``config`` used,
    ``warm_start`` for the iterative methods and the solver's own record. FDK
    and FBP volumes too large for the device come back as host NumPy arrays,
    made in slabs (``host_slabs``). Independent axial SIRT/MLEM planes and their
    warm starts also use host slabs when needed. Coupled cone MLEM uses host-owned
    volume slabs when its device workspace does not fit. Fourier always returns
    host arrays in slabs.
    ``out`` supplies writable float32 host storage for FBP/FDK, Fourier, and
    cone-beam CGLS, SIRT and MLEM, and forces host slabs; it is returned
    without allocating another full volume.

    Parameters
    ----------
    method : str
        Reconstruction method accepted by method_config.
    geometry : ScanGeometry
        Scan geometry, including angles and per-view poses.
    grid : Grid
        Reconstruction grid, with physical spacing in ``(x, y, z)`` order.
    detector : Detector
        Detector shape and placement in the same length unit as the grid.
    projections : array-like
        Line integrals of shape ``(views, nv, nu)``. For MLEM these represent
        nonnegative linear Poisson measurements, not raw transmission counts.
    config : MethodConfig or None, optional
        Method-specific configuration. None uses that method's defaults.
    warm_start : bool, optional
        Start iterative reconstruction from FBP or FDK. Default is False.
    out : numpy.ndarray or None, optional
        Writable float32 ``(nz, ny, nx)`` host storage for FBP/FDK, Fourier,
        CUDA Joseph FISTA, or cone-beam CGLS, SIRT and MLEM. None selects storage;
        other methods do not accept out.

    Returns
    -------
    volume : jax.Array or numpy.ndarray
        Reconstruction of shape ``(nz, ny, nx)``; uses out itself if supplied.
    record : dict
        Configuration used, solver diagnostics and any host-slab or warm-start flags.

    Raises
    ------
    ValueError
        If method, configuration, shapes or geometry are incompatible, warm_start
        is requested for a noniterative method, or out is unsupported.
    """
    from tomojax.recon._truncation import truncation_record

    cfg = method_config(method, config=config, warm_start=warm_start or None)
    truncation = truncation_record(geometry, detector, projections)
    if cfg.method == "fourier":
        import numpy as np

        from tomojax.recon._fourier_backend import resolve_fourier_backend

        backend = resolve_fourier_backend(cfg.backend)
        volume = recon.fourier(
            geometry,
            grid,
            detector,
            np.asarray(projections),
            config=replace(cfg, backend=backend),
            out=out,
        )
        record = {"config": cfg, "host_slabs": True, "backend": backend}
        return volume, {**record, "truncation": truncation}
    if out is not None and method not in {"fbp", "fista", "cgls", "sirt", "mlem"}:
        raise ValueError(f"method {method!r} does not take out")
    if out is not None and method not in {"fbp", "fista", "cgls"}:
        from tomojax.core.geometry.cone import is_cone_beam

        if not is_cone_beam(geometry):
            raise ValueError(f"method {method!r} takes out for cone-beam geometry only")
    if cfg.method == "fbp":
        volume, record = fbp_volume(geometry, grid, detector, projections, cfg, None, out)
        return volume, {**record, "truncation": truncation}
    from tomojax import progress
    from tomojax.recon._exterior import Treatment, outside_data, subtract_background
    from tomojax.recon._host_arrays import validate_host_arrays

    det_grid = _detector_grid(geometry, detector)
    if out is not None:
        _ = validate_host_arrays(projections, out, grid.shape, method, device_input=True)
    with progress.task(
        "reconstruct", method, 50 if cfg.iterations is None else cfg.iterations, "iterations"
    ):
        with progress.preparing("estimate exterior projection"):
            background, share = outside_data(geometry, grid, detector, projections, cfg)
            treatment = Treatment.record(cfg.method, share, applied=background is not None)
            warm_data = projections if warm_start and background is not None else None
            if background is not None and cfg.method != "mlem":
                projections = subtract_background(projections, background)
                background = None
        volume, record = _iterative_result(
            geometry,
            grid,
            detector,
            projections,
            cfg,
            det_grid,
            warm_start=warm_start,
            out=out,
            background=background,
            warm_data=warm_data,
        )
        return volume, {**record, "truncation": {**truncation, **treatment._asdict()}}


def _iterative_result(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    cfg: CGLSConfig | FISTAConfig | SPDHGConfig | SIRTConfig | MLEMConfig,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    *,
    warm_start: bool,
    background: np.ndarray | None = None,
    warm_data: np.ndarray | jnp.ndarray | None = None,
    out: np.ndarray | None = None,
) -> tuple[jnp.ndarray | np.ndarray, dict[str, object]]:
    """Plan independent axial slabs before allocating a full warm initializer."""
    match cfg.method:
        case "sirt" | "mlem":
            algebraic = cfg
        case _:
            algebraic = None
    if algebraic is not None:
        from tomojax.recon._axial_slabs import axial_slabs, independent_planes

        if det_grid is None and independent_planes(geometry, grid, detector):
            support = algebraic.support
            if support is not None:
                import numpy as np

                if algebraic.method == "sirt":
                    from tomojax.core.validation import validate_volume

                    _ = validate_volume(support, grid, context="sirt", name="support")
                # A host broadcast view: each slab uploads only its own rows.
                support = np.broadcast_to(np.asarray(support, bool), (grid.nz, grid.ny, grid.nx))

            solver = summary = None
            if algebraic.method == "sirt":
                from tomojax.recon._discrepancy import sirt_slab_report
                from tomojax.recon.sirt import SIRTCall

                solver = SIRTCall().solve
                if algebraic.iterations is None and algebraic.subsets == 1:
                    summary = sirt_slab_report

            def solve(
                local: ParallelGeometry,
                data: jnp.ndarray | np.ndarray,
                _initial: jnp.ndarray | np.ndarray | None,
                local_support: jnp.ndarray | np.ndarray | None,
                section: slice,
            ) -> tuple[jnp.ndarray | np.ndarray, dict[str, object]]:
                return _iterative_incore(
                    local, local.grid, local.detector, data,
                    replace(algebraic, support=local_support), None,
                    warm_start=warm_start, solver=solver,
                    background=None if background is None else background[:, section],
                    warm_data=None if warm_data is None else warm_data[:, section],
                )  # fmt: skip

            result = axial_slabs(
                geometry, grid, detector, projections, initial=None, support=support,
                config=algebraic,
                solve=solve, summarize=summary, additive=background is not None,
            )  # fmt: skip
            if result is not None:
                volume, record = result
                return volume, {**record, "config": cfg, "warm_start": warm_start}
    return _iterative_incore(
        geometry,
        grid,
        detector,
        projections,
        cfg,
        det_grid,
        warm_start=warm_start,
        out=out,
        background=background,
        warm_data=warm_data,
    )


def _iterative_incore(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    cfg: CGLSConfig | FISTAConfig | SPDHGConfig | SIRTConfig | MLEMConfig,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    *,
    warm_start: bool,
    background: np.ndarray | None = None,
    warm_data: np.ndarray | jnp.ndarray | None = None,
    solver: Callable[..., tuple[jnp.ndarray | np.ndarray, dict[str, object]]] | None = None,
    out: np.ndarray | None = None,
) -> tuple[jnp.ndarray | np.ndarray, dict[str, object]]:
    """Initialize and dispatch an iterative solve with its method's data model."""
    if cfg.method == "mlem":
        from .mlem import run_mlem

        volume, record = run_mlem(
            geometry, projections, grid=grid, detector=detector, config=cfg,
            det_grid=det_grid, warm_start=warm_start, out=out, background=background,
        )  # fmt: skip
        return volume, {"config": cfg, "warm_start": warm_start, **record}
    start = None
    if warm_start:
        from tomojax.recon._warm_start import fbp_start

        devices = getattr(cfg, "devices", None) if det_grid is None else None
        start = fbp_start(
            geometry, grid, detector, projections if warm_data is None else warm_data,
            config=recon.FBPConfig(devices=devices),
            det_grid=det_grid, out=out,
        )  # fmt: skip
        lower = getattr(cfg, "lower_bound", None)
        if lower is not None:
            import jax.numpy as jnp

            start = jnp.maximum(start, float(lower))
    args = (geometry, grid, detector, projections)
    solve = getattr(recon, cfg.method) if solver is None else solver
    storage = {} if out is None else {"out": out}
    volume, record = solve(*args, init_x=start, config=cfg, det_grid=det_grid, **storage)
    return volume, {"config": cfg, "warm_start": warm_start, **record}


def axial_fbp_plane(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: np.ndarray | jnp.ndarray,
    cfg: FBPConfig,
    index: int,
) -> jnp.ndarray | np.ndarray | None:
    """Use the full solver's view batches when a voxel plane takes one detector row.

    Parameters
    ----------
    geometry : Geometry
        Calibrated acquisition.
    grid : Grid
        Full output voxel grid.
    detector : Detector
        Original measured detector.
    projections : numpy.ndarray or jax.Array
        Input stack in view, row and column order.
    cfg : FBPConfig
        Full reconstruction's algorithm and device settings.
    index : int
        Requested axial plane on the output grid.

    Returns
    -------
    array or None
        Requested attenuation plane, or None when other rows can contribute.
    """
    import numpy as np

    from tomojax.geometry import grid_volume_origin

    from ._axial_slabs import independent_planes
    from ._fbp_storage import too_large_for_the_device
    from .fbp import parallel_fbp

    if not independent_planes(geometry, grid, detector):
        return None
    if too_large_for_the_device(geometry, grid, cfg):
        from ._fbp_storage import first_device, selected_slab
        from .fbp_host import FBPHostConfig, fbp_host

        slab, first = selected_slab(grid, index)
        with first_device(cfg):
            volume = fbp_host(
                geometry,
                slab,
                detector,
                projections,
                config=FBPHostConfig(fbp=cfg, slices_per_batch=slab.nz),
            )
        return volume[index - first].copy()
    # Compilers can fuse the z multiply-add and replace division by a reciprocal.
    # Prove each operation exact in FP32 before removing interpolation neighbors.
    pitch = float(np.float32(detector.dv))
    z = index * pitch + float(np.float32(grid_volume_origin(grid)[2]))
    centered = z - float(np.float32(detector.center[1]))
    if (
        np.frexp(pitch)[0] != 0.5
        or float(np.float32(z)) != z
        or float(np.float32(centered)) != centered
        or centered / pitch != index - (detector.nv - 1) / 2
    ):
        return None
    return parallel_fbp(geometry, grid, detector, projections, cfg, None, row=index)[0]


def predict_projections(
    geometry: ScanGeometry, volume: np.ndarray | jnp.ndarray, *, grid: Grid, config: MethodConfig
) -> jnp.ndarray:
    """Project a reconstruction with the fitted method's detector lattice and interpolation.

    Parameters
    ----------
    geometry : ScanGeometry
        Acquisition geometry of the views to predict.
    volume : array
        Reconstructed attenuation in (z, y, x) order.
    grid : Grid
        Physical sampling of the reconstruction.
    config : MethodConfig
        Resolved reconstruction settings, including CGLS interpolation.

    Returns
    -------
    jax.Array
        Predicted line integrals in (view, v, u) order.
    """
    import jax.numpy as jnp
    import numpy as np

    from tomojax.recon._fbp_storage import first_device
    from tomojax.recon._operators import matched_operators
    from tomojax.recon._prediction import host_prediction

    analytic = isinstance(config, recon.FBPConfig | recon.FourierConfig)
    if isinstance(volume, np.ndarray):
        with first_device(config):
            return host_prediction(
                geometry,
                volume,
                grid,
                backend="auto" if analytic else config.backend,
                interpolation=getattr(config, "interpolation", "linear"),
            )
    _, split, build, _ = matched_operators(
        geometry,
        grid,
        "evaluate",
        None if analytic else getattr(config, "devices", None),
        interpolation=getattr(config, "interpolation", "linear"),
        backend="auto" if analytic else config.backend,
    )
    forward, _ = build()
    if split is None:
        return forward(jnp.asarray(volume))
    return split.take(forward(split.everywhere(jnp.asarray(volume))))


def _detector_grid(
    geometry: Geometry, detector: Detector
) -> tuple[jnp.ndarray, jnp.ndarray] | None:
    from tomojax.geometry.detector_grid import detector_grid_from_geometry_inputs

    return detector_grid_from_geometry_inputs(detector, geometry)


__all__ = [
    "MethodConfig",
    "Projector",
    "axial_fbp_plane",
    "backproject",
    "clear_filter_caches",
    "default_fbp_scale",
    "method_config",
    "predict_projections",
    "project",
    "reconstruct_arrays",
    "reconstruct_slabs",
    "solver_name",
    "validate_reconstruction",
    "validate_slab_grid",
]
