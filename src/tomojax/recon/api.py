"""Public API for reconstruction routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

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
    from collections.abc import Sequence

    import numpy as np

    from tomojax._typed_arrays import Device
    from tomojax.geometry import Detector, Geometry, Grid

type ReconstructionAlgorithm = Literal["fbp", "cgls", "fista", "spdhg"]


@dataclass(frozen=True)
class ReconstructionAlgorithmOptions:
    """User-facing solver choices after CLI/config parsing."""

    method: ReconstructionAlgorithm
    filter: str = "ramp"
    iterations: int = 50
    tv_weight: float = 0.005
    regulariser: Regulariser = "tv"
    huber_delta: float = 1e-2
    lipschitz: float | None = None
    nonnegative: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None
    theta: float = 1.0
    seed: int = 0
    spdhg_tau: float | None = None
    spdhg_sigma_data: float | None = None
    spdhg_sigma_tv: float | None = None
    warm_start: bool = False
    checkpoint_projector: bool = True
    tv_prox_iterations: int = 10


@dataclass(frozen=True)
class ReconstructionAlgorithmRequest:
    """Resolved reconstruction inputs for a single solver run."""

    options: ReconstructionAlgorithmOptions
    geometry: Geometry
    grid: Grid
    detector: Detector
    projections: jnp.ndarray | np.ndarray
    detector_grid: tuple[jnp.ndarray, jnp.ndarray] | None
    volume_mask: jnp.ndarray | None
    views_per_batch: int
    views_per_batch_mode: str
    gather_dtype: str
    devices: Device | Sequence[Device] | None = None


@dataclass(frozen=True)
class ReconstructionResult:
    """Reconstructed volume plus normalized algorithm metadata.

    Volumes too large for the device come back as host NumPy arrays.
    """

    volume: jnp.ndarray | np.ndarray
    algorithm_config: dict[str, object]


def default_views_per_batch(method: str) -> int:
    """Views per device batch a method uses unless told otherwise.

    SPDHG's batch is its stochastic block size; the other solvers' batched
    operators launch one projector call per batch.
    """
    return {"spdhg": 16, "fista": 64, "cgls": 64}.get(str(method).lower(), 1)


def run_reconstruction_algorithm(request: ReconstructionAlgorithmRequest) -> ReconstructionResult:
    """Run the selected reconstruction method from resolved geometry and projections."""
    if request.options.method == "fbp":
        return _run_fbp_reconstruction(request)
    if request.options.method == "cgls":
        return _run_cgls_reconstruction(request)
    if request.options.method == "fista":
        return _run_fista_reconstruction(request)
    return _run_spdhg_reconstruction(request)


def _host_cone_volume(request: ReconstructionAlgorithmRequest) -> bool:
    """Whether a cone FDK volume is too large to hold on the device."""
    from tomojax.backends import device_free_memory_bytes
    from tomojax.core.geometry.cone import ConeSegments, beam_of

    if isinstance(request.geometry, ConeSegments):
        return False  # FDK refuses segmented scans with its own message.
    if beam_of(request.geometry) is None or request.detector_grid is not None:
        return False
    free = device_free_memory_bytes()
    grid = request.grid
    return free is not None and 4 * grid.nx * grid.ny * grid.nz > 0.4 * free


def _run_host_fdk(request: ReconstructionAlgorithmRequest) -> ReconstructionResult:
    """FDK in z-slabs with the volume on the host (see :func:`fdk_host`)."""
    import logging

    import numpy as np

    from tomojax.recon.fdk import FDKConfig, FDKHostConfig, fdk_host

    grid = request.grid
    logging.info(
        "The %dx%dx%d volume exceeds device memory; reconstructing in z-slabs on the host",
        grid.nx,
        grid.ny,
        grid.nz,
    )
    fdk_cfg = FDKConfig(
        filter=str(request.options.filter), views_per_batch=int(request.views_per_batch)
    )
    volume = fdk_host(
        request.geometry,
        grid,
        request.detector,
        np.asarray(request.projections, np.float32),
        config=FDKHostConfig(fdk=fdk_cfg),
    )
    if request.volume_mask is not None:
        volume *= np.asarray(request.volume_mask, np.float32)
    return ReconstructionResult(
        volume=volume,
        algorithm_config={
            "filter": str(fdk_cfg.filter),
            "views_per_batch": int(fdk_cfg.views_per_batch),
            "host_slabs": True,
        },
    )


def _run_fbp_reconstruction(request: ReconstructionAlgorithmRequest) -> ReconstructionResult:
    if _host_cone_volume(request):
        return _run_host_fdk(request)
    cfg = FBPConfig(
        filter=str(request.options.filter),
        views_per_batch=int(request.views_per_batch),
        projector_unroll=1,
        checkpoint_projector=bool(request.options.checkpoint_projector),
        gather_dtype=str(request.gather_dtype),
    )
    volume = fbp(
        request.geometry,
        request.grid,
        request.detector,
        request.projections,
        config=cfg,
        det_grid=request.detector_grid,
    )
    if request.volume_mask is not None:
        volume = volume * request.volume_mask
    return ReconstructionResult(
        volume=volume,
        algorithm_config={
            "filter": str(cfg.filter),
            "views_per_batch": int(cfg.views_per_batch),
            "projector_unroll": int(cfg.projector_unroll),
            "checkpoint_projector": bool(cfg.checkpoint_projector),
            "gather_dtype": str(cfg.gather_dtype),
        },
    )


def _run_cgls_reconstruction(request: ReconstructionAlgorithmRequest) -> ReconstructionResult:
    """Unregularised least squares; the fastest-converging solver for consistent data."""
    cfg = CGLSConfig(
        iterations=int(request.options.iterations),
        views_per_batch=int(request.views_per_batch),
        devices=request.devices,
    )
    init_x = _fbp_warm_start(request, nonnegative=False)
    volume, info = cgls(
        request.geometry,
        request.grid,
        request.detector,
        request.projections,
        init_x=init_x,
        config=cfg,
        det_grid=request.detector_grid,
    )
    if request.volume_mask is not None:
        volume = volume * request.volume_mask
    return ReconstructionResult(
        volume=volume,
        algorithm_config={
            "iterations": int(cfg.iterations),
            "effective_iterations": int(cast("int", info["effective_iterations"])),
            "termination": str(info["termination"]),
            "views_per_batch": int(cfg.views_per_batch),
            "projector_model": str(info["projector_model"]),
            "projector_backend": str(info["projector_backend"]),
            "warm_start": bool(request.options.warm_start),
            "support": "applied after the solve" if request.volume_mask is not None else None,
        },
    )


def _run_fista_reconstruction(request: ReconstructionAlgorithmRequest) -> ReconstructionResult:
    cfg = FistaConfig(
        iterations=int(request.options.iterations),
        tv_weight=float(request.options.tv_weight),
        regulariser=cast("Regulariser", str(request.options.regulariser)),
        huber_delta=float(request.options.huber_delta),
        lipschitz=(
            float(request.options.lipschitz) if request.options.lipschitz is not None else None
        ),
        views_per_batch=int(request.views_per_batch),
        projector_unroll=1,
        checkpoint_projector=bool(request.options.checkpoint_projector),
        gather_dtype=str(request.gather_dtype),
        tv_prox_iterations=int(request.options.tv_prox_iterations),
        support=request.volume_mask,
        nonnegative=bool(request.options.nonnegative),
        lower_bound=(
            float(request.options.lower_bound) if request.options.lower_bound is not None else None
        ),
        upper_bound=(
            float(request.options.upper_bound) if request.options.upper_bound is not None else None
        ),
        devices=request.devices,
    )
    volume = fista_tv(
        request.geometry,
        request.grid,
        request.detector,
        request.projections,
        config=cfg,
        det_grid=request.detector_grid,
    )[0]
    return ReconstructionResult(
        volume=volume,
        algorithm_config={
            "iterations": int(cfg.iterations),
            "tv_weight": float(cfg.tv_weight),
            "regulariser": str(cfg.regulariser),
            "huber_delta": float(cfg.huber_delta),
            "lipschitz": cfg.lipschitz,
            "views_per_batch": int(request.views_per_batch),
            "projector_unroll": int(cfg.projector_unroll),
            "checkpoint_projector": bool(cfg.checkpoint_projector),
            "gather_dtype": str(cfg.gather_dtype),
            "grad_mode": str(cfg.grad_mode),
            "tv_prox_iterations": int(cfg.tv_prox_iterations),
            "recon_rel_tol": cfg.recon_rel_tol,
            "recon_patience": int(cfg.recon_patience),
            "power_iterations": int(cfg.power_iterations),
            "support": "cylindrical" if request.volume_mask is not None else None,
            "nonnegative": bool(cfg.nonnegative),
            "lower_bound": cfg.lower_bound,
            "upper_bound": cfg.upper_bound,
        },
    )


def _run_spdhg_reconstruction(request: ReconstructionAlgorithmRequest) -> ReconstructionResult:
    cfg = SPDHGConfig(
        iterations=int(request.options.iterations),
        tv_weight=float(request.options.tv_weight),
        regulariser=cast("Regulariser", str(request.options.regulariser)),
        huber_delta=float(request.options.huber_delta),
        theta=float(request.options.theta),
        views_per_batch=int(request.views_per_batch),
        seed=int(request.options.seed),
        tau=(float(request.options.spdhg_tau) if request.options.spdhg_tau is not None else None),
        sigma_data=(
            float(request.options.spdhg_sigma_data)
            if request.options.spdhg_sigma_data is not None
            else None
        ),
        sigma_tv=(
            float(request.options.spdhg_sigma_tv)
            if request.options.spdhg_sigma_tv is not None
            else None
        ),
        projector_unroll=1,
        checkpoint_projector=bool(request.options.checkpoint_projector),
        gather_dtype=str(request.gather_dtype),
        nonnegative=bool(request.options.nonnegative),
        support=request.volume_mask if request.volume_mask is not None else None,
        log_every=1,
    )
    init_x = _fbp_warm_start(request, nonnegative=cfg.nonnegative)
    volume = spdhg_tv(
        request.geometry,
        request.grid,
        request.detector,
        request.projections,
        init_x=init_x,
        config=cfg,
        det_grid=request.detector_grid,
    )[0]
    return ReconstructionResult(
        volume=volume,
        algorithm_config={
            "iterations": int(cfg.iterations),
            "tv_weight": float(cfg.tv_weight),
            "regulariser": str(cfg.regulariser),
            "huber_delta": float(cfg.huber_delta),
            "theta": float(cfg.theta),
            "views_per_batch": int(cfg.views_per_batch),
            "seed": int(cfg.seed),
            "tau": cfg.tau,
            "sigma_data": cfg.sigma_data,
            "sigma_tv": cfg.sigma_tv,
            "projector_unroll": int(cfg.projector_unroll),
            "checkpoint_projector": bool(cfg.checkpoint_projector),
            "gather_dtype": str(cfg.gather_dtype),
            "nonnegative": bool(cfg.nonnegative),
            "support": "cylindrical" if request.volume_mask is not None else None,
            "log_every": int(cfg.log_every),
            "warm_start": bool(request.options.warm_start),
        },
    )


def _fbp_warm_start(
    request: ReconstructionAlgorithmRequest, *, nonnegative: bool
) -> jnp.ndarray | None:
    if not request.options.warm_start:
        return None
    warm_start_vpb = (
        1 if request.views_per_batch_mode == "default" else int(request.views_per_batch)
    )
    warm_start_cfg = FBPConfig(
        filter=str(request.options.filter),
        views_per_batch=warm_start_vpb,
        projector_unroll=1,
        checkpoint_projector=bool(request.options.checkpoint_projector),
        gather_dtype=str(request.gather_dtype),
    )
    init_x = fbp(
        request.geometry,
        request.grid,
        request.detector,
        request.projections,
        config=warm_start_cfg,
        det_grid=request.detector_grid,
    )
    if request.volume_mask is not None:
        init_x = init_x * request.volume_mask
    return jnp.maximum(init_x, 0.0) if nonnegative else init_x


__all__ = [
    "CGLSConfig",
    "FBPConfig",
    "FBPHostConfig",
    "FistaConfig",
    "FourierConfig",
    "ReconstructionAlgorithmOptions",
    "ReconstructionAlgorithmRequest",
    "ReconstructionResult",
    "Regulariser",
    "SPDHGConfig",
    "cgls",
    "cgls_multires",
    "clear_filter_caches",
    "default_fbp_scale",
    "default_views_per_batch",
    "fbp",
    "fbp_host",
    "fista_tv",
    "fourier_reconstruct",
    "run_parallel_fbp_direct_pallas",
    "run_reconstruction_algorithm",
    "spdhg_tv",
]
