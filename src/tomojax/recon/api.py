"""Public API for reconstruction routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import jax.numpy as jnp

from tomojax.recon._backprojection_accumulation import sum_backproject_views_chunked
from tomojax.recon._support import VolumeSupportKind, centered_volume_support
from tomojax.recon.fbp import (
    FBPConfig,
    default_fbp_scale,
    fbp,
    run_parallel_fbp_direct_pallas,
    supports_parallel_fbp_z_integer,
)
from tomojax.recon.filters import clear_filter_caches
from tomojax.recon.fista_tv import FistaConfig, fista_tv
from tomojax.recon.spdhg_tv import SPDHGConfig, spdhg_tv
from tomojax.recon.types import Regulariser

type ReconstructionAlgorithm = Literal["fbp", "fista", "spdhg"]


@dataclass(frozen=True)
class ReconstructionAlgorithmOptions:
    """User-facing solver choices after CLI/config parsing."""

    algorithm: ReconstructionAlgorithm
    filter_name: str = "ramp"
    iters: int = 50
    lambda_tv: float = 0.005
    regulariser: Regulariser = "tv"
    huber_delta: float = 1e-2
    lipschitz: float | None = None
    positivity: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None
    theta: float = 1.0
    spdhg_seed: int = 0
    spdhg_tau: float | None = None
    spdhg_sigma_data: float | None = None
    spdhg_sigma_tv: float | None = None
    warm_start: Literal["none", "fbp"] = "none"
    checkpoint_projector: bool = True
    tv_prox_iters: int = 10


@dataclass(frozen=True)
class ReconstructionAlgorithmRequest:
    """Resolved reconstruction inputs for a single solver run."""

    options: ReconstructionAlgorithmOptions
    geometry: Any
    grid: Any
    detector: Any
    projections: jnp.ndarray
    detector_grid: tuple[jnp.ndarray, jnp.ndarray] | None
    volume_mask: jnp.ndarray | None
    views_per_batch: int
    views_per_batch_mode: str
    gather_dtype: str


@dataclass(frozen=True)
class ReconstructionResult:
    """Reconstructed volume plus normalized algorithm metadata."""

    volume: jnp.ndarray
    algorithm_config: dict[str, object]


def run_reconstruction_algorithm(request: ReconstructionAlgorithmRequest) -> ReconstructionResult:
    """Run the selected reconstruction algorithm from resolved geometry and projections."""
    if request.options.algorithm == "fbp":
        return _run_fbp_reconstruction(request)
    if request.options.algorithm == "fista":
        return _run_fista_reconstruction(request)
    return _run_spdhg_reconstruction(request)


def _run_fbp_reconstruction(request: ReconstructionAlgorithmRequest) -> ReconstructionResult:
    cfg = FBPConfig(
        filter_name=str(request.options.filter_name),
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
            "filter": str(cfg.filter_name),
            "views_per_batch": int(cfg.views_per_batch),
            "projector_unroll": int(cfg.projector_unroll),
            "checkpoint_projector": bool(cfg.checkpoint_projector),
            "gather_dtype": str(cfg.gather_dtype),
        },
    )


def _run_fista_reconstruction(request: ReconstructionAlgorithmRequest) -> ReconstructionResult:
    cfg = FistaConfig(
        iters=int(request.options.iters),
        lambda_tv=float(request.options.lambda_tv),
        regulariser=cast("Regulariser", str(request.options.regulariser)),
        huber_delta=float(request.options.huber_delta),
        L=(float(request.options.lipschitz) if request.options.lipschitz is not None else None),
        views_per_batch=int(request.views_per_batch),
        projector_unroll=1,
        checkpoint_projector=bool(request.options.checkpoint_projector),
        gather_dtype=str(request.gather_dtype),
        tv_prox_iters=int(request.options.tv_prox_iters),
        support=request.volume_mask,
        positivity=bool(request.options.positivity),
        lower_bound=(
            float(request.options.lower_bound) if request.options.lower_bound is not None else None
        ),
        upper_bound=(
            float(request.options.upper_bound) if request.options.upper_bound is not None else None
        ),
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
            "iters": int(cfg.iters),
            "lambda_tv": float(cfg.lambda_tv),
            "regulariser": str(cfg.regulariser),
            "huber_delta": float(cfg.huber_delta),
            "L": cfg.L,
            "views_per_batch": int(request.views_per_batch),
            "projector_unroll": int(cfg.projector_unroll),
            "checkpoint_projector": bool(cfg.checkpoint_projector),
            "gather_dtype": str(cfg.gather_dtype),
            "grad_mode": str(cfg.grad_mode),
            "tv_prox_iters": int(cfg.tv_prox_iters),
            "recon_rel_tol": cfg.recon_rel_tol,
            "recon_patience": int(cfg.recon_patience),
            "power_iters": int(cfg.power_iters),
            "support": "cylindrical" if request.volume_mask is not None else None,
            "positivity": bool(cfg.positivity),
            "lower_bound": cfg.lower_bound,
            "upper_bound": cfg.upper_bound,
        },
    )


def _run_spdhg_reconstruction(request: ReconstructionAlgorithmRequest) -> ReconstructionResult:
    cfg = SPDHGConfig(
        iters=int(request.options.iters),
        lambda_tv=float(request.options.lambda_tv),
        regulariser=cast("Regulariser", str(request.options.regulariser)),
        huber_delta=float(request.options.huber_delta),
        theta=float(request.options.theta),
        views_per_batch=int(request.views_per_batch),
        seed=int(request.options.spdhg_seed),
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
        positivity=True,
        support=request.volume_mask if request.volume_mask is not None else None,
        log_every=1,
    )
    init_x = _spdhg_warm_start(request)
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
            "iters": int(cfg.iters),
            "lambda_tv": float(cfg.lambda_tv),
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
            "positivity": bool(cfg.positivity),
            "support": "cylindrical" if request.volume_mask is not None else None,
            "log_every": int(cfg.log_every),
            "warm_start": str(request.options.warm_start),
        },
    )


def _spdhg_warm_start(request: ReconstructionAlgorithmRequest) -> jnp.ndarray | None:
    if str(request.options.warm_start).lower() != "fbp":
        return None
    warm_start_vpb = (
        1 if request.views_per_batch_mode == "default" else int(request.views_per_batch)
    )
    warm_start_cfg = FBPConfig(
        filter_name=str(request.options.filter_name),
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
    return jnp.maximum(init_x, 0.0)


__all__ = [
    "FBPConfig",
    "FistaConfig",
    "ReconstructionAlgorithmOptions",
    "ReconstructionAlgorithmRequest",
    "ReconstructionResult",
    "Regulariser",
    "SPDHGConfig",
    "VolumeSupportKind",
    "centered_volume_support",
    "clear_filter_caches",
    "default_fbp_scale",
    "fbp",
    "fista_tv",
    "run_parallel_fbp_direct_pallas",
    "run_reconstruction_algorithm",
    "spdhg_tv",
    "sum_backproject_views_chunked",
    "supports_parallel_fbp_z_integer",
]
