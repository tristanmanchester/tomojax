"""Public API for reconstruction routines."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import fields, replace
from typing import TYPE_CHECKING, cast

import jax.numpy as jnp

from tomojax.recon._fourier_stream import fourier_slabs
from tomojax.recon._warm_start import fbp_start
from tomojax.recon.cgls import CGLSConfig, cgls
from tomojax.recon.cgls_multires import cgls_multires
from tomojax.recon.fbp import FBPConfig, default_fbp_scale, fbp, run_parallel_fbp_direct_pallas
from tomojax.recon.fbp_host import FBPHostConfig, fbp_host, fbp_slabs
from tomojax.recon.filters import clear_filter_caches
from tomojax.recon.fista_tv import FistaConfig, fista_tv
from tomojax.recon.fourier import FourierConfig, fourier_reconstruct
from tomojax.recon.sirt import SIRTConfig, sirt
from tomojax.recon.spdhg_tv import SPDHGConfig, spdhg_tv
from tomojax.recon.types import Regulariser

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    import numpy as np

    from tomojax.geometry import Detector, Geometry, Grid, ParallelGeometry

type MethodConfig = FBPConfig | FourierConfig | CGLSConfig | FistaConfig | SPDHGConfig | SIRTConfig

_CONFIGS: dict[str, type[MethodConfig]] = {
    "fbp": FBPConfig,
    "fourier": FourierConfig,
    "cgls": CGLSConfig,
    "fista": FistaConfig,
    "spdhg": SPDHGConfig,
    "sirt": SIRTConfig,
}


def method_config(
    method: str, *, config: MethodConfig | None = None, **settings: object
) -> MethodConfig:
    """``method``'s configuration: ``config``, or its class's defaults, with ``settings``.

    ``fbp`` (FDK for cone beams) takes an :class:`FBPConfig`, ``cgls`` a
    :class:`CGLSConfig`, ``fista`` a :class:`FistaConfig` and ``spdhg`` an
    :class:`SPDHGConfig`; ``sirt`` takes :class:`SIRTConfig`, and ``fourier``
    :class:`FourierConfig`. Each setting
    replaces the field of its name, as :func:`dataclasses.replace` does. A config
    of another class, or a setting its class has no field for, raises
    :class:`ValueError` naming the valid ones.
    """
    kind = _CONFIGS.get(method)
    if kind is None:
        raise ValueError(f"method must be one of {', '.join(_CONFIGS)}; got {method!r}")
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
    """Reconstruct ``projections`` with ``method``; :func:`tomojax.reconstruct` on arrays.

    ``config`` is the method's configuration (see :func:`method_config`).
    ``warm_start`` starts ``cgls``, ``fista`` and ``spdhg`` from the FBP
    reconstruction. Cone segments start from the view-count-weighted mean of
    their individual FDK images; each segment must cover an FDK-compatible
    arc. Returns the volume and its record: the ``config`` used,
    ``warm_start`` for the iterative methods and the solver's own record. FDK
    and FBP volumes too large for the device come back as host NumPy arrays,
    made in slabs (``host_slabs``). Fourier always returns host arrays in slabs.
    ``out`` supplies writable float32 host storage for FBP/FDK or Fourier and
    forces host slabs; it is returned without allocating another full volume.
    """
    cfg = method_config(method, config=config)
    if isinstance(cfg, FourierConfig):
        import numpy as np

        from tomojax.recon._fourier_backend import resolve_fourier_backend

        if warm_start:
            raise ValueError("method 'fourier' does not take warm_start")
        backend = resolve_fourier_backend(cfg.backend)
        volume = fourier_reconstruct(
            cast("ParallelGeometry", geometry),
            grid,
            detector,
            np.asarray(projections),
            config=replace(cfg, backend=backend),
            out=out,
        )
        return volume, {"config": cfg, "host_slabs": True, "backend": backend}
    if out is not None and not isinstance(cfg, FBPConfig):
        raise ValueError(f"method {method!r} does not take out")
    det_grid = _detector_grid(geometry, detector)
    if isinstance(cfg, FBPConfig):
        if warm_start:
            raise ValueError("method 'fbp' does not take warm_start")
        return _fbp_result(geometry, grid, detector, projections, cfg, det_grid, out)
    start = None
    if warm_start:
        # On the solver's devices; an explicit detector grid takes one.
        devices = getattr(cfg, "devices", None) if det_grid is None else None
        start = fbp_start(
            geometry, grid, detector, projections, config=FBPConfig(devices=devices),
            det_grid=det_grid,
        )  # fmt: skip
        if getattr(cfg, "nonnegative", False):
            start = jnp.maximum(start, 0.0)
    volume, record = _iterative_result(geometry, grid, detector, projections, cfg, start, det_grid)
    return volume, {"config": cfg, "warm_start": warm_start, **record}


def _iterative_result(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    cfg: CGLSConfig | FistaConfig | SPDHGConfig | SIRTConfig,
    start: jnp.ndarray | None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> tuple[jnp.ndarray, dict[str, object]]:
    """Dispatch normalized method configurations to the matched solvers."""
    args = (geometry, grid, detector, projections)
    if isinstance(cfg, CGLSConfig):
        return cgls(*args, init_x=start, config=cfg, det_grid=det_grid)
    if isinstance(cfg, FistaConfig):
        return fista_tv(*args, init_x=start, config=cfg, det_grid=det_grid)
    if isinstance(cfg, SIRTConfig):
        return sirt(*args, init_x=start, config=cfg, det_grid=det_grid)
    return spdhg_tv(*args, init_x=start, config=cfg, det_grid=det_grid)


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
    volume = fbp(geometry, grid, detector, projections, config=config, det_grid=det_grid)
    return volume, {"config": config}


def _detector_grid(
    geometry: Geometry, detector: Detector
) -> tuple[jnp.ndarray, jnp.ndarray] | None:
    from tomojax.geometry.api import detector_grid_from_geometry_inputs

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
    host = FBPHostConfig(filter=cfg.filter, scale=cfg.scale, backprojector=cfg.backprojector)
    # Slabs are not shared among devices; the first selected one does the work.
    with _first_device(cfg):
        return fbp_host(
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
    "Regulariser",
    "SIRTConfig",
    "SPDHGConfig",
    "cgls",
    "cgls_multires",
    "clear_filter_caches",
    "default_fbp_scale",
    "fbp",
    "fbp_host",
    "fbp_slabs",
    "fista_tv",
    "fourier_reconstruct",
    "fourier_slabs",
    "method_config",
    "reconstruct_arrays",
    "run_parallel_fbp_direct_pallas",
    "sirt",
    "spdhg_tv",
]
