"""Public API for reconstruction routines."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import TYPE_CHECKING, cast

import jax.numpy as jnp

from tomojax.recon._host_arrays import host_projection_input, validate_host_arrays
from tomojax.recon._warm_start import fbp_start
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

    from tomojax.geometry import Detector, Geometry, Grid, ParallelGeometry

type MethodConfig = FBPConfig | FourierConfig | CGLSConfig | FistaConfig | SPDHGConfig

_CONFIGS: dict[str, type[MethodConfig]] = {
    "fbp": FBPConfig,
    "fourier": FourierConfig,
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
    :class:`SPDHGConfig`; ``fourier`` takes :class:`FourierConfig`. Each setting
    replaces its field as :func:`dataclasses.replace` does. A config of another class, or a setting
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

        if warm_start:
            raise ValueError("method 'fourier' does not take warm_start")
        volume = fourier_reconstruct(
            cast("ParallelGeometry", geometry),
            grid,
            detector,
            np.asarray(projections),
            config=cfg,
            out=out,
        )
        return volume, {"config": cfg, "host_slabs": True}
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
    args = (geometry, grid, detector, projections)
    if isinstance(cfg, CGLSConfig):
        volume, record = cgls(*args, init_x=start, config=cfg, det_grid=det_grid)
    elif isinstance(cfg, FistaConfig):
        volume, record = fista_tv(*args, init_x=start, config=cfg, det_grid=det_grid)
    else:
        volume, record = spdhg_tv(*args, init_x=start, config=cfg, det_grid=det_grid)
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
    volume = fbp(geometry, grid, detector, projections, config=config, det_grid=det_grid)
    return volume, {"config": config}


def _detector_grid(
    geometry: Geometry, detector: Detector
) -> tuple[jnp.ndarray, jnp.ndarray] | None:
    from tomojax.geometry.api import detector_grid_from_geometry_inputs

    return detector_grid_from_geometry_inputs(detector, geometry)


def _too_large_for_the_device(geometry: Geometry, grid: Grid, cfg: FBPConfig) -> bool:
    """Whether an FBP volume would take more than 40% of free device memory."""
    import jax

    from tomojax.backends import device_free_memory_bytes
    from tomojax.core.devices import as_devices
    from tomojax.core.geometry.cone import ConeSegments

    if isinstance(geometry, ConeSegments):
        return False  # FDK refuses segmented scans with its own message.

    def oversized() -> bool:
        free = device_free_memory_bytes()
        return free is not None and 4 * grid.nx * grid.ny * grid.nz > 0.4 * free

    devices = as_devices(cfg.devices)
    if devices is None:
        return oversized()
    # View-shared reconstruction holds the entire volume on every device.
    # A selected device need not be the process default (nor GPU 0).
    for device in devices:
        with jax.default_device(device):
            if oversized():
                return True
    return False


def _host_fbp(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    cfg: FBPConfig,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Use bounded slabs for either beam type, keeping the result on the host."""
    from concurrent.futures import ThreadPoolExecutor
    import logging

    import jax
    import numpy as np

    from tomojax.core.devices import as_devices
    from tomojax.core.geometry.cone import beam_of
    from tomojax.geometry import grid_volume_origin

    if beam_of(geometry) is not None:
        return _host_fdk(geometry, grid, detector, projections, cfg, out=out)
    logging.info(
        "Reconstructing %dx%dx%d in slabs on the host",
        grid.nx,
        grid.ny,
        grid.nz,
    )
    data = host_projection_input(projections)
    host_cfg = FBPHostConfig(
        filter=cfg.filter, padding=cfg.padding, scale=cfg.scale, backprojector=cfg.backprojector
    )
    devices = as_devices(cfg.devices)
    if devices is None:
        return fbp_host(geometry, grid, detector, data, config=host_cfg, out=out)
    # Share spatial regions instead of views: summing whole device volumes
    # would defeat host storage. Each worker still uses bounded local slabs.
    devices = devices[: grid.nz]
    boundaries = np.linspace(0, grid.nz, len(devices) + 1, dtype=int)
    output = validate_host_arrays(data, out, (grid.nx, grid.ny, grid.nz), device_input=True)
    ox, oy, oz = grid_volume_origin(grid)

    def run(index: int) -> None:
        start, stop = map(int, boundaries[index : index + 2])
        slab = replace(
            grid, nz=stop - start, vol_origin=(ox, oy, oz + start * grid.vz), vol_center=None
        )
        with jax.default_device(devices[index]):
            _ = fbp_host(
                geometry, slab, detector, data, config=host_cfg, out=output[:, :, start:stop]
            )

    with ThreadPoolExecutor(len(devices)) as pool:
        _ = list(pool.map(run, range(len(devices))))
    return output


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

    if cfg.padding == "edge":
        raise ValueError("FBP edge padding requires canonical parallel-beam detector pixels")
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
