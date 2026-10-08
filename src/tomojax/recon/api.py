"""Public API for reconstruction routines."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import TYPE_CHECKING

import jax.numpy as jnp

from tomojax.recon.cgls import CGLSConfig, cgls
from tomojax.recon.cgls_multires import cgls_multires
from tomojax.recon.fbp import FBPConfig, default_fbp_scale, fbp, run_parallel_fbp_direct_pallas
from tomojax.recon.fbp_host import FBPHostConfig, fbp_host
from tomojax.recon.filters import clear_filter_caches
from tomojax.recon.fista_tv import FistaConfig, fista_tv
from tomojax.recon.fourier import FourierConfig, fourier_reconstruct
from tomojax.recon.spdhg_tv import SPDHGConfig, spdhg_tv
from tomojax.recon.types import Regulariser

if TYPE_CHECKING:
    import numpy as np

    from tomojax.geometry import Detector, Geometry, Grid

type MethodConfig = FBPConfig | CGLSConfig | FistaConfig | SPDHGConfig

_CONFIGS: dict[str, type[MethodConfig]] = {
    "fbp": FBPConfig,
    "cgls": CGLSConfig,
    "fista": FistaConfig,
    "spdhg": SPDHGConfig,
}


def method_config(
    method: str, *, config: MethodConfig | None = None, **settings: object
) -> MethodConfig:
    """``method``'s configuration: ``config``, or its class's defaults, with ``settings``.

    ``fbp`` (FDK for cone beams) takes an :class:`FBPConfig`, ``cgls`` a
    :class:`CGLSConfig`, ``fista`` a :class:`FistaConfig` and ``spdhg`` an
    :class:`SPDHGConfig`. Each setting replaces the field of its name, as
    :func:`dataclasses.replace` does. A config of another class, or a setting
    its class has no field for, raises :class:`ValueError` naming the valid ones.
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
) -> tuple[jnp.ndarray | np.ndarray, dict[str, object]]:
    """Reconstruct ``projections`` with ``method``; :func:`tomojax.reconstruct` on arrays.

    ``config`` is the method's configuration (see :func:`method_config`).
    ``warm_start`` starts ``cgls``, ``fista`` and ``spdhg`` from the FBP
    reconstruction. Returns the volume and its record: the ``config`` used,
    ``warm_start`` for the iterative methods and the solver's own record. FDK
    volumes too large for the device come back as host NumPy arrays, made in
    z-slabs (``host_slabs``).
    """
    cfg = method_config(method, config=config)
    det_grid = _detector_grid(geometry, detector)
    if isinstance(cfg, FBPConfig):
        if warm_start:
            raise ValueError("method 'fbp' does not take warm_start")
        if det_grid is None and _too_large_for_the_device(geometry, grid):
            return _host_fdk(geometry, grid, detector, projections, cfg), {
                "config": cfg,
                "host_slabs": True,
            }
        volume = fbp(geometry, grid, detector, projections, config=cfg, det_grid=det_grid)
        return volume, {"config": cfg}
    start = None
    if warm_start:
        start = fbp(geometry, grid, detector, projections, det_grid=det_grid)
        if getattr(cfg, "nonnegative", False):
            start = jnp.maximum(start, 0.0)
    args = (geometry, grid, detector, projections)
    if isinstance(cfg, CGLSConfig):
        volume, record = cgls(*args, init_x=start, config=cfg, det_grid=det_grid)
    elif isinstance(cfg, FistaConfig):
        volume, record = fista_tv(*args, init_x=start, config=cfg, det_grid=det_grid)
    else:
        volume, record = spdhg_tv(*args, init_x=start, config=cfg, det_grid=det_grid)
    return volume, {"config": cfg, "warm_start": warm_start, **record}


def _detector_grid(
    geometry: Geometry, detector: Detector
) -> tuple[jnp.ndarray, jnp.ndarray] | None:
    from tomojax.geometry.api import detector_grid_from_geometry_inputs

    return detector_grid_from_geometry_inputs(detector, geometry)


def _too_large_for_the_device(geometry: Geometry, grid: Grid) -> bool:
    """Whether a cone-beam FDK volume would take more than 40% of free device memory."""
    from tomojax.backends import device_free_memory_bytes
    from tomojax.core.geometry.cone import ConeSegments, beam_of

    if isinstance(geometry, ConeSegments):
        return False  # FDK refuses segmented scans with its own message.
    if beam_of(geometry) is None:
        return False
    free = device_free_memory_bytes()
    return free is not None and 4 * grid.nx * grid.ny * grid.nz > 0.4 * free


def _host_fdk(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    cfg: FBPConfig,
) -> np.ndarray:
    """FDK in z-slabs with the volume on the host (see :func:`fdk_host`)."""
    import logging

    import numpy as np

    from tomojax.recon.fdk import FDKConfig, FDKHostConfig, fdk_host

    logging.info(
        "The %dx%dx%d volume exceeds device memory; reconstructing in z-slabs on the host",
        grid.nx,
        grid.ny,
        grid.nz,
    )
    # As fbp's own FDK: the same filter, and the CUDA kernel for "pallas".
    backend = {"auto": "auto", "jax": "jax", "pallas": "cuda"}[cfg.backprojector]
    fdk = FDKConfig(filter=cfg.filter, backend=backend)
    data = np.asarray(projections, np.float32)
    return fdk_host(geometry, grid, detector, data, config=FDKHostConfig(fdk=fdk))


__all__ = [
    "CGLSConfig",
    "FBPConfig",
    "FBPHostConfig",
    "FistaConfig",
    "FourierConfig",
    "Regulariser",
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
    "reconstruct_arrays",
    "run_parallel_fbp_direct_pallas",
    "spdhg_tv",
]
