"""Expert reconstruction helpers: method configuration, array and slab solvers.

The solvers and their configuration classes are exported by
:mod:`tomojax.recon`; :func:`tomojax.reconstruct` is the workflow entry point.
"""

from __future__ import annotations

from dataclasses import fields, replace
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from tomojax import recon
from tomojax.recon._fbp_storage import fbp_volume, fdk_config

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    import jax.numpy as jnp
    import numpy as np

    from tomojax.geometry import Detector, Geometry, Grid, ParallelGeometry
    from tomojax.recon.cgls import CGLSConfig
    from tomojax.recon.fbp import FBPConfig, default_fbp_scale, run_parallel_fbp_direct_cuda
    from tomojax.recon.filters import clear_filter_caches
    from tomojax.recon.fista_tv import FistaConfig
    from tomojax.recon.fourier import FourierConfig
    from tomojax.recon.mlem import MLEMConfig
    from tomojax.recon.polychromatic import PolychromaticConfig
    from tomojax.recon.sirt import SIRTConfig
    from tomojax.recon.spdhg_tv import SPDHGConfig
    from tomojax.recon.types import ReconstructionSlab

type MethodConfig = (
    recon.FBPConfig
    | recon.FourierConfig
    | recon.CGLSConfig
    | recon.FistaConfig
    | recon.SPDHGConfig
    | recon.SIRTConfig
    | recon.MLEMConfig
    | recon.PolychromaticConfig
)

_CONFIGS = {
    "fbp": "FBPConfig",
    "fourier": "FourierConfig",
    "cgls": "CGLSConfig",
    "fista": "FistaConfig",
    "spdhg": "SPDHGConfig",
    "sirt": "SIRTConfig",
    "mlem": "MLEMConfig",
    "polychromatic": "PolychromaticConfig",
}

_SOURCES = {
    "clear_filter_caches": "filters",
    "default_fbp_scale": "fbp",
    "run_parallel_fbp_direct_cuda": "fbp",
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
    :class:`CGLSConfig`, ``fista`` a :class:`FistaConfig` and ``spdhg`` an
    :class:`SPDHGConfig`; ``sirt`` takes :class:`SIRTConfig`, ``mlem``
    :class:`MLEMConfig`, ``polychromatic`` :class:`PolychromaticConfig` and
    ``fourier`` :class:`FourierConfig`. Each setting
    replaces the field of its name, as :func:`dataclasses.replace` does. A config
    of another class, or a setting its class has no field for, raises
    :class:`ValueError` naming the valid ones; so does ``warm_start`` for the
    direct methods (fbp, fourier), which start from nothing.

    Parameters
    ----------
    method : str
        Reconstruction method: "fbp", "fourier", "cgls", "fista", "spdhg",
        "sirt", "mlem" or "polychromatic".
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
        raise ValueError(f"method must be one of {', '.join(_CONFIGS)}; got {method!r}")
    kind = cast("type[MethodConfig]", getattr(recon, name))
    if warm_start is not None and method in {"fbp", "fourier"}:
        raise ValueError(f"method {method!r} does not take warm_start")
    if config is not None and not isinstance(config, kind):
        raise ValueError(
            f"method {method!r} takes a {kind.__name__}, not a {type(config).__name__}"
        )
    names = sorted(item.name for item in fields(kind) if item.init)
    unknown = sorted(set(settings) - set(names))
    if unknown:
        raise ValueError(
            f"method {method!r} does not take {', '.join(unknown)} "
            f"({kind.__name__} has {', '.join(names)})"
        )
    base = kind() if config is None else config
    return replace(base, **settings) if settings else base


def reconstruct_slabs[T](
    method: str,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    read: Callable[[int, int], tuple[jnp.ndarray | np.ndarray, T]],
    *,
    config: MethodConfig,
    warm_start: bool = False,
) -> Generator[ReconstructionSlab[T], None, None]:
    """Reconstruct owned axial slabs from bounded corrected detector-row reads.

    Parameters
    ----------
    method : {'fbp', 'fourier', 'sirt', 'mlem'}
        Reconstruction method.
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
        Method configuration, as for ``tomojax.reconstruct``.
    warm_start : bool, optional
        Start SIRT/OSEM from FBP. Default is False.

    Returns
    -------
    generator of ReconstructionSlab
        Owned slabs in z order, each with its source interval and context.

    Raises
    ------
    ValueError
        If the method or a setting does not support slabs.
    """
    cfg = method_config(method, config=config, warm_start=warm_start or None)
    if method in {"fbp", "fourier"}:
        if method == "fbp":
            from tomojax.core.geometry.cone import is_cone_beam

            if is_cone_beam(geometry):
                from .fdk import fdk_slabs

                return fdk_slabs(
                    geometry, grid, detector, read, config=fdk_config(cast("FBPConfig", cfg))
                )
            from .fbp_host import fbp_slabs

            return fbp_slabs(
                cast("ParallelGeometry", geometry),
                grid,
                detector,
                read,
                config=cast("FBPConfig", cfg),
            )
        from ._fourier_stream import fourier_slabs

        return fourier_slabs(
            cast("ParallelGeometry", geometry),
            grid,
            detector,
            read,
            config=cast("FourierConfig", cfg),
        )
    if method not in {"sirt", "mlem"}:
        raise ValueError("reconstruct_slabs supports FBP/Fourier/SIRT/MLEM")
    from ._iterative_slabs import iterative_slabs

    return iterative_slabs(
        cast("ParallelGeometry", geometry),
        grid,
        detector,
        read,
        method=method,
        config=cast("SIRTConfig | MLEMConfig", cfg),
        warm_start=warm_start,
    )


def reconstruct_arrays(
    method: str,
    geometry: Geometry,
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
    geometry : Geometry
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
        Writable float32 ``(nx, ny, nz)`` host storage for FBP/FDK, Fourier, or
        cone-beam CGLS, SIRT and MLEM. None lets the method select storage;
        other methods do not accept out.

    Returns
    -------
    volume : jax.Array or numpy.ndarray
        Reconstruction of shape ``(nx, ny, nz)``; uses out itself if supplied.
    record : dict
        Configuration used, solver diagnostics and any host-slab or warm-start flags.

    Raises
    ------
    ValueError
        If method, configuration, shapes or geometry are incompatible, warm_start
        is requested for a noniterative method, or out is unsupported.
    """
    cfg = method_config(method, config=config, warm_start=warm_start or None)
    if method == "fourier":
        cfg = cast("FourierConfig", cfg)
        import numpy as np

        from tomojax.recon._fourier_backend import resolve_fourier_backend

        backend = resolve_fourier_backend(cfg.backend)
        volume = recon.fourier_reconstruct(
            cast("ParallelGeometry", geometry),
            grid,
            detector,
            np.asarray(projections),
            config=replace(cfg, backend=backend),
            out=out,
        )
        return volume, {"config": cfg, "host_slabs": True, "backend": backend}
    if out is not None and method not in {"fbp", "cgls", "sirt", "mlem"}:
        raise ValueError(f"method {method!r} does not take out")
    if out is not None and method != "fbp":
        from tomojax.core.geometry.cone import is_cone_beam

        if not is_cone_beam(geometry):
            raise ValueError(f"method {method!r} takes out for cone-beam geometry only")
    det_grid = _detector_grid(geometry, detector)
    if method == "fbp":
        cfg = cast("FBPConfig", cfg)
        return fbp_volume(geometry, grid, detector, projections, cfg, det_grid, out)
    if method == "polychromatic":
        if warm_start:
            raise ValueError("method 'polychromatic' does not take warm_start")
        volume, record = recon.polychromatic(
            geometry, grid, detector, projections, config=cast("PolychromaticConfig", cfg)
        )
        return volume, {"config": cfg, **record}
    # The method name has ruled out the FBP and Fourier configurations above.
    iterative = cast("CGLSConfig | FistaConfig | SPDHGConfig | SIRTConfig | MLEMConfig", cfg)
    return _iterative_result(
        method, geometry, grid, detector, projections, iterative, det_grid,
        warm_start=warm_start, out=out,
    )  # fmt: skip


def _iterative_result(
    method: str,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    cfg: CGLSConfig | FistaConfig | SPDHGConfig | SIRTConfig | MLEMConfig,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    *,
    warm_start: bool,
    out: np.ndarray | None = None,
) -> tuple[jnp.ndarray | np.ndarray, dict[str, object]]:
    """Plan independent axial slabs before allocating a full warm initializer."""
    if method in {"sirt", "mlem"}:
        from tomojax.recon._axial_slabs import axial_slabs, independent_planes

        algebraic = cast("SIRTConfig | MLEMConfig", cfg)
        if (
            algebraic.projector_model in {"auto", "joseph"}
            and det_grid is None
            and independent_planes(geometry, grid, detector)
        ):
            support = algebraic.support
            if support is not None:
                import jax.numpy as jnp

                if method == "sirt":
                    from tomojax.core.validation import validate_volume

                    validate_volume(support, grid, context="sirt", name="support")
                support = jnp.broadcast_to(jnp.asarray(support, bool), (grid.nx, grid.ny, grid.nz))

            solver = None
            if method == "sirt":
                from tomojax.recon.sirt import SIRTCall

                solver = SIRTCall().solve

            def solve(
                local: ParallelGeometry,
                data: jnp.ndarray | np.ndarray,
                _initial: jnp.ndarray | np.ndarray | None,
                local_support: jnp.ndarray | np.ndarray | None,
            ) -> tuple[jnp.ndarray | np.ndarray, dict[str, object]]:
                return _iterative_incore(
                    method, local, local.grid, local.detector, data,
                    replace(algebraic, support=local_support), None,
                    warm_start=warm_start, solver=solver,
                )  # fmt: skip

            # EM stores axial sensitivities as single planes; the cache policy
            # separately reserves capacity for them after inputs are resident.
            buffers = (
                4
                if method == "mlem"
                else 3 + min(algebraic.subsets, 16) + 2 * getattr(algebraic, "acceleration", False)
            )
            result = axial_slabs(
                geometry, grid, detector, projections, initial=None, support=support,
                volume_buffers=buffers,
                projection_buffers=2 if method == "mlem" else 2 + 2 / algebraic.subsets,
                solve=solve,
            )  # fmt: skip
            if result is not None:
                volume, record = result
                return volume, {**record, "config": cfg, "warm_start": warm_start}
    return _iterative_incore(
        method, geometry, grid, detector, projections, cfg, det_grid,
        warm_start=warm_start, out=out,
    )  # fmt: skip


def _iterative_incore(
    method: str,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    cfg: CGLSConfig | FistaConfig | SPDHGConfig | SIRTConfig | MLEMConfig,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    *,
    warm_start: bool,
    solver: Callable[..., tuple[jnp.ndarray | np.ndarray, dict[str, object]]] | None = None,
    out: np.ndarray | None = None,
) -> tuple[jnp.ndarray | np.ndarray, dict[str, object]]:
    """Initialize and dispatch an iterative solve with its method's data model."""
    if method == "mlem":
        from .mlem import run_mlem

        volume, record = run_mlem(
            geometry, projections, grid=grid, detector=detector, config=cast("MLEMConfig", cfg),
            det_grid=det_grid, warm_start=warm_start, out=out,
        )  # fmt: skip
        return volume, {"config": cfg, "warm_start": warm_start, **record}
    start = None
    if warm_start:
        from tomojax.recon._warm_start import fbp_start

        devices = getattr(cfg, "devices", None) if det_grid is None else None
        start = fbp_start(
            geometry, grid, detector, projections, config=recon.FBPConfig(devices=devices),
            det_grid=det_grid, out=out,
        )  # fmt: skip
        if getattr(cfg, "nonnegative", False):
            import jax.numpy as jnp

            start = jnp.maximum(start, 0.0)
    args = (geometry, grid, detector, projections)
    name = {
        "cgls": "cgls",
        "fista": "fista_tv",
        "spdhg": "spdhg_tv",
        "sirt": "sirt",
        "mlem": "mlem",
    }[method]
    solve = getattr(recon, name) if solver is None else solver
    storage = {} if out is None else {"out": out}
    volume, record = solve(*args, init_x=start, config=cfg, det_grid=det_grid, **storage)
    return volume, {"config": cfg, "warm_start": warm_start, **record}


def _detector_grid(
    geometry: Geometry, detector: Detector
) -> tuple[jnp.ndarray, jnp.ndarray] | None:
    from tomojax.geometry.detector_grid import detector_grid_from_geometry_inputs

    return detector_grid_from_geometry_inputs(detector, geometry)


__all__ = [
    "MethodConfig",
    "clear_filter_caches",
    "default_fbp_scale",
    "method_config",
    "reconstruct_arrays",
    "reconstruct_slabs",
    "run_parallel_fbp_direct_cuda",
]
