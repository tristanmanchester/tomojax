"""Public API for reconstruction routines."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import fields, replace
from importlib import import_module
import sys
from typing import TYPE_CHECKING, Any, cast

from tomojax import recon

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from contextlib import AbstractContextManager

    import jax.numpy as jnp
    import numpy as np

    from tomojax.geometry import Detector, Geometry, Grid, ParallelGeometry
    from tomojax.recon.cgls import CGLSConfig, cgls
    from tomojax.recon.cgls_multires import cgls_multires
    from tomojax.recon.fbp import FBPConfig, default_fbp_scale, fbp, run_parallel_fbp_direct_pallas
    from tomojax.recon.fbp_host import FBPHostConfig, fbp_host
    from tomojax.recon.filters import clear_filter_caches
    from tomojax.recon.fista_tv import FistaConfig, fista_tv
    from tomojax.recon.fourier import FourierConfig, fourier_reconstruct
    from tomojax.recon.mlem import MLEMConfig, mlem
    from tomojax.recon.sirt import SIRTConfig, sirt
    from tomojax.recon.spdhg_tv import SPDHGConfig, spdhg_tv
    from tomojax.recon.types import ReconstructionSlab, Regulariser

type MethodConfig = (
    recon.FBPConfig
    | recon.FourierConfig
    | recon.CGLSConfig
    | recon.FistaConfig
    | recon.SPDHGConfig
    | recon.SIRTConfig
    | recon.MLEMConfig
)

_CONFIGS = {
    "fbp": "FBPConfig",
    "fourier": "FourierConfig",
    "cgls": "CGLSConfig",
    "fista": "FistaConfig",
    "spdhg": "SPDHGConfig",
    "sirt": "SIRTConfig",
    "mlem": "MLEMConfig",
}

_SOURCES = {
    "CGLSConfig": "cgls",
    "FBPConfig": "fbp",
    "FBPHostConfig": "fbp_host",
    "FistaConfig": "fista_tv",
    "FourierConfig": "fourier",
    "MLEMConfig": "mlem",
    "Regulariser": "types",
    "SIRTConfig": "sirt",
    "SPDHGConfig": "spdhg_tv",
    "cgls": "cgls",
    "cgls_multires": "cgls_multires",
    "clear_filter_caches": "filters",
    "default_fbp_scale": "fbp",
    "fbp": "fbp",
    "fbp_host": "fbp_host",
    "fista_tv": "fista_tv",
    "fourier_reconstruct": "fourier",
    "mlem": "mlem",
    "run_parallel_fbp_direct_pallas": "fbp",
    "sirt": "sirt",
    "spdhg_tv": "spdhg_tv",
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
    method: str, *, config: MethodConfig | None = None, **settings: object
) -> MethodConfig:
    """Build a method configuration from defaults or named replacements.

    ``fbp`` (FDK for cone beams) takes an :class:`FBPConfig`, ``cgls`` a
    :class:`CGLSConfig`, ``fista`` a :class:`FistaConfig` and ``spdhg`` an
    :class:`SPDHGConfig`; ``sirt`` takes :class:`SIRTConfig`, and ``fourier``
    :class:`FourierConfig`. Each setting
    replaces the field of its name, as :func:`dataclasses.replace` does. A config
    of another class, or a setting its class has no field for, raises
    :class:`ValueError` naming the valid ones.

    Parameters
    ----------
    method : str
        Reconstruction method: "fbp", "fourier", "cgls", "fista", "spdhg",
        "sirt" or "mlem".
    config : MethodConfig or None, optional
        Existing configuration of the method's type. None uses its defaults.
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
    geometry: ParallelGeometry,
    grid: Grid,
    detector: Detector,
    read: Callable[[int, int], tuple[jnp.ndarray | np.ndarray, T]],
    *,
    config: MethodConfig,
    warm_start: bool = False,
) -> Generator[ReconstructionSlab[T], None, None]:
    """Reconstruct owned axial slabs from bounded corrected detector-row reads."""
    cfg = method_config(method, config=config)
    if method in {"fbp", "fourier"}:
        if warm_start:
            raise ValueError(f"method {method!r} does not take warm_start")
        if method == "fbp":
            from .fbp_host import fbp_slabs

            return fbp_slabs(geometry, grid, detector, read, config=cast("FBPConfig", cfg))
        from ._fourier_stream import fourier_slabs

        return fourier_slabs(geometry, grid, detector, read, config=cast("FourierConfig", cfg))
    if method not in {"sirt", "mlem"}:
        raise ValueError("reconstruct_slabs supports FBP/Fourier/SIRT/MLEM")
    from ._iterative_slabs import iterative_slabs

    return iterative_slabs(
        geometry,
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
    reconstruction. Cone segments start from the view-count-weighted mean of
    their individual FDK images; each segment must cover an FDK-compatible
    arc. Returns the volume and its record: the ``config`` used,
    ``warm_start`` for the iterative methods and the solver's own record. FDK
    and FBP volumes too large for the device come back as host NumPy arrays,
    made in slabs (``host_slabs``). Independent axial SIRT/MLEM planes and their
    warm starts also use host slabs when needed. Fourier always returns host arrays in slabs.
    ``out`` supplies writable float32 host storage for FBP/FDK or Fourier and
    forces host slabs; it is returned without allocating another full volume.

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
        Writable float32 ``(nx, ny, nz)`` host storage for FBP/FDK or Fourier.
        None lets the method select storage; other methods do not accept out.

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
    cfg = method_config(method, config=config)
    if method == "fourier":
        cfg = cast("FourierConfig", cfg)
        import numpy as np

        from tomojax.recon._fourier_backend import resolve_fourier_backend

        if warm_start:
            raise ValueError("method 'fourier' does not take warm_start")
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
    if out is not None and method != "fbp":
        raise ValueError(f"method {method!r} does not take out")
    det_grid = _detector_grid(geometry, detector)
    if method == "fbp":
        cfg = cast("FBPConfig", cfg)
        if warm_start:
            raise ValueError("method 'fbp' does not take warm_start")
        return _fbp_result(geometry, grid, detector, projections, cfg, det_grid, out)
    # The method name has ruled out the FBP and Fourier configurations above.
    iterative = cast("CGLSConfig | FistaConfig | SPDHGConfig | SIRTConfig | MLEMConfig", cfg)
    return _iterative_result(
        method, geometry, grid, detector, projections, iterative, det_grid, warm_start=warm_start
    )


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

            def solve(
                local: ParallelGeometry,
                data: jnp.ndarray | np.ndarray,
                _initial: jnp.ndarray | np.ndarray | None,
                local_support: jnp.ndarray | np.ndarray | None,
            ) -> tuple[jnp.ndarray | np.ndarray, dict[str, object]]:
                return _iterative_incore(
                    method, local, local.grid, local.detector, data,
                    replace(algebraic, support=local_support), None, warm_start=warm_start,
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
        method, geometry, grid, detector, projections, cfg, det_grid, warm_start=warm_start
    )


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
) -> tuple[jnp.ndarray | np.ndarray, dict[str, object]]:
    """Initialize and dispatch an iterative solve with its method's data model."""
    start = None
    if warm_start:
        from tomojax.recon._warm_start import fbp_start

        # On the solver's devices; an explicit detector grid takes one.
        devices = getattr(cfg, "devices", None) if det_grid is None else None
        start = fbp_start(
            geometry, grid, detector, projections, config=recon.FBPConfig(devices=devices),
            det_grid=det_grid,
        )  # fmt: skip
        if method == "mlem" or getattr(cfg, "nonnegative", False):
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
    solve = getattr(sys.modules[__name__], name)
    volume, record = solve(*args, init_x=start, config=cfg, det_grid=det_grid)
    return volume, {"config": cfg, "warm_start": warm_start, **record}


def _fbp_result(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    config: FBPConfig,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    out: np.ndarray | None,
) -> tuple[jnp.ndarray | np.ndarray, dict[str, object]]:
    """Choose FBP storage, respecting caller-owned output and device capacity."""
    if out is not None and det_grid is not None:
        raise ValueError("host output does not support explicit detector grids")
    if out is not None or (det_grid is None and _too_large_for_the_device(geometry, grid, config)):
        volume = _host_fbp(geometry, grid, detector, projections, config, out=out)
        return volume, {"config": config, "host_slabs": True}
    volume = recon.fbp(geometry, grid, detector, projections, config=config, det_grid=det_grid)
    return volume, {"config": config}


def _detector_grid(
    geometry: Geometry, detector: Detector
) -> tuple[jnp.ndarray, jnp.ndarray] | None:
    from tomojax.geometry.detector_grid import detector_grid_from_geometry_inputs

    return detector_grid_from_geometry_inputs(detector, geometry)


def _first_device(cfg: FBPConfig) -> AbstractContextManager[object]:
    """Make the configuration's first device the default, or leave JAX's own."""
    import jax

    from tomojax.core.devices import as_devices

    devices = as_devices(cfg.devices)
    return jax.default_device(devices[0]) if devices else nullcontext()


def _too_large_for_the_device(geometry: Geometry, grid: Grid, cfg: FBPConfig) -> bool:
    """Whether an FBP volume would take more than 40% of free device memory."""
    from tomojax.backends import device_free_memory_bytes
    from tomojax.core.geometry.cone import ConeSegments

    if isinstance(geometry, ConeSegments):
        return False  # FDK refuses segmented scans with its own message.
    # View-shared reconstruction holds the whole volume on every device.
    with _first_device(cfg):
        free = device_free_memory_bytes()
    return free is not None and 4 * grid.nx * grid.ny * grid.nz > 0.4 * free


def _host_fbp(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    cfg: FBPConfig,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Bounded slabs for either beam type, keeping the result on the host."""
    import logging

    from tomojax.core.geometry.cone import beam_of

    if beam_of(geometry) is not None:
        return _host_fdk(geometry, grid, detector, projections, cfg, out=out)
    logging.info("Reconstructing %dx%dx%d in z-slabs on the host", grid.nx, grid.ny, grid.nz)
    host = recon.FBPHostConfig(filter=cfg.filter, scale=cfg.scale, backprojector=cfg.backprojector)
    # Slabs are not shared among devices; the first selected one does the work.
    with _first_device(cfg):
        return recon.fbp_host(
            cast("ParallelGeometry", geometry), grid, detector, projections, config=host, out=out
        )


def _host_fdk(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    cfg: FBPConfig,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """FDK in z-slabs with the volume on the host (see :func:`fdk_host`)."""
    import logging

    import numpy as np

    from tomojax.recon.fdk import FDKConfig, FDKHostConfig, fdk_host

    logging.info(
        "Reconstructing %dx%dx%d in z-slabs on the host",
        grid.nx,
        grid.ny,
        grid.nz,
    )
    # As fbp's own FDK: the same filter, and the CUDA kernel for "pallas".
    backend = {"auto": "auto", "jax": "jax", "pallas": "cuda"}[cfg.backprojector]
    fdk = FDKConfig(filter=cfg.filter, backend=backend, devices=cfg.devices)
    data = np.asarray(projections, np.float32)
    return fdk_host(geometry, grid, detector, data, config=FDKHostConfig(fdk=fdk), out=out)


__all__ = [
    "CGLSConfig",
    "FBPConfig",
    "FBPHostConfig",
    "FistaConfig",
    "FourierConfig",
    "MLEMConfig",
    "Regulariser",
    "SIRTConfig",
    "SPDHGConfig",
    "cgls",
    "cgls_multires",
    "clear_filter_caches",
    "default_fbp_scale",
    "fbp",
    "fbp_host",
    "fista_tv",
    "fourier_reconstruct",
    "method_config",
    "mlem",
    "reconstruct_arrays",
    "reconstruct_slabs",
    "run_parallel_fbp_direct_pallas",
    "sirt",
    "spdhg_tv",
]
