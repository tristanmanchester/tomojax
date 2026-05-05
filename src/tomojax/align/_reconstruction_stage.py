from __future__ import annotations

import functools
import logging
import time
from typing import Mapping

import jax
import jax.numpy as jnp

from ..core.geometry.base import Detector, Geometry, Grid
from ..core.geometry.views import stack_view_poses
from ..recon.fista_tv_core import FistaCoreConfig, fista_tv_core_arrays
from ..recon.fista_tv import FistaConfig, fista_tv
from ..recon.spdhg_tv import SPDHGConfig, spdhg_tv
from .geometry.parametrizations import se3_from_5d
from ._observer import OuterStat
from ._results import record_reconstruction_info as _record_reconstruction_info
from .objectives.recon_layer import PoseAdjustedGeometry


def _heuristic_projection_lipschitz(
    *,
    n_views: int,
    grid: Grid,
    lambda_tv: float,
    huber_delta: float,
) -> float:
    min_voxel = max(min(float(grid.vx), float(grid.vy), float(grid.vz)), 1e-6)
    max_extent = max(
        float(grid.nx) * float(grid.vx),
        float(grid.ny) * float(grid.vy),
        float(grid.nz) * float(grid.vz),
    )
    projection_l = 1.2 * float(n_views) * max(max_extent / min_voxel, 1.0)
    regulariser_l = float(lambda_tv) * 12.0 / max(float(huber_delta), 1e-6)
    return max(projection_l + regulariser_l, 1e-6)


@functools.partial(jax.jit, static_argnames=("grid", "detector", "cfg"))
def _run_huber_fista_core_jit(
    x0: jnp.ndarray,
    T_all: jnp.ndarray,
    det_u: jnp.ndarray,
    det_v: jnp.ndarray,
    projections: jnp.ndarray,
    L_value: jnp.ndarray,
    *,
    grid: Grid,
    detector: Detector,
    cfg: FistaCoreConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    result = fista_tv_core_arrays(
        x0=x0,
        T_all=T_all,
        det_grid=(det_u, det_v),
        projections=projections,
        grid=grid,
        detector=detector,
        cfg=cfg,
        L_override=L_value,
    )
    return (
        result.x,
        result.loss,
        result.data_loss,
        result.regulariser_value,
        result.effective_iters,
    )


def _run_reconstruction_step(
    *,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    params5: jnp.ndarray,
    x: jnp.ndarray,
    cfg: object,
    L_prev: float | None,
    outer_idx: int,
    recon_algo: str,
) -> tuple[jnp.ndarray, float | None, OuterStat]:
    recon_geometry = PoseAdjustedGeometry(geometry=geometry, params5=params5)

    def _run_fista(vpb: int | None, unroll: int, gather: str, gm: str):
        fista_cfg = FistaConfig(
            iters=cfg.recon_iters,
            lambda_tv=cfg.lambda_tv,
            regulariser=cfg.regulariser,
            huber_delta=cfg.huber_delta,
            L=L_prev,
            views_per_batch=vpb,
            projector_unroll=int(unroll),
            checkpoint_projector=cfg.checkpoint_projector,
            gather_dtype=gather,
            grad_mode=gm,
            tv_prox_iters=int(cfg.tv_prox_iters),
            recon_rel_tol=cfg.recon_rel_tol,
            recon_patience=(int(cfg.recon_patience) if cfg.recon_patience is not None else 0),
        )
        return fista_tv(
            recon_geometry,
            grid,
            detector,
            projections,
            init_x=x,
            config=fista_cfg,
            det_grid=det_grid,
        )

    def _run_spdhg():
        spdhg_cfg = SPDHGConfig(
            iters=int(cfg.recon_iters),
            lambda_tv=float(cfg.lambda_tv),
            regulariser=cfg.regulariser,
            huber_delta=float(cfg.huber_delta),
            views_per_batch=max(1, int(cfg.views_per_batch)),
            seed=int(cfg.spdhg_seed) + int(outer_idx) - 1,
            projector_unroll=int(cfg.projector_unroll),
            checkpoint_projector=cfg.checkpoint_projector,
            gather_dtype=cfg.gather_dtype,
            positivity=bool(cfg.recon_positivity),
            log_every=1,
        )
        return spdhg_tv(
            recon_geometry,
            grid,
            detector,
            projections,
            init_x=x,
            config=spdhg_cfg,
            det_grid=det_grid,
        )

    def _run_huber_fista_core():
        n_views = int(projections.shape[0])
        L_core = (
            float(L_prev)
            if L_prev is not None
            else _heuristic_projection_lipschitz(
                n_views=n_views,
                grid=grid,
                lambda_tv=float(cfg.lambda_tv),
                huber_delta=float(cfg.huber_delta),
            )
        )
        if isinstance(recon_geometry, PoseAdjustedGeometry):
            nominal = stack_view_poses(recon_geometry.geometry, n_views)
            T_all = nominal @ jax.vmap(se3_from_5d)(recon_geometry.params5)
        else:
            T_all = stack_view_poses(recon_geometry, n_views)
        core_cfg = FistaCoreConfig(
            iters=int(cfg.recon_iters),
            lambda_tv=float(cfg.lambda_tv),
            regulariser="huber_tv",
            huber_delta=float(cfg.huber_delta),
            L=1.0,
            checkpoint_projector=bool(cfg.checkpoint_projector),
            projector_unroll=int(cfg.projector_unroll),
            gather_dtype=str(cfg.gather_dtype),
            views_per_batch=int(cfg.views_per_batch),
            forward_projector=str(getattr(cfg, "projector_backend", "jax")),
            backprojector=str(getattr(cfg, "projector_backend", "jax")),
            compute_iteration_loss=False,
            compute_final_data_loss=False,
            compute_final_regulariser_value=False,
        )

        x_core, loss, data_loss, regulariser_value, effective_iters = _run_huber_fista_core_jit(
            x,
            T_all,
            det_grid[0],
            det_grid[1],
            projections,
            jnp.asarray(L_core, dtype=jnp.float32),
            grid=grid,
            detector=detector,
            cfg=core_cfg,
        )
        info = {
            "loss": [],
            "effective_iters": int(effective_iters),
            "early_stop": False,
            "regulariser": "huber_tv",
            "huber_delta": float(cfg.huber_delta),
            "data_loss_computed": False,
            "regulariser_value_computed": False,
            "L": float(L_core / 1.2),
        }
        return x_core, info

    vpb0 = cfg.views_per_batch if cfg.views_per_batch > 0 else None
    recon_retry = False
    recon_start = time.perf_counter()
    if recon_algo == "fista":
        if str(cfg.regulariser) == "huber_tv":
            x_out, info_rec = _run_huber_fista_core()
        else:
            try:
                x_out, info_rec = _run_fista(
                    vpb0,
                    int(cfg.projector_unroll),
                    cfg.gather_dtype,
                    "auto",
                )
            except Exception as e:
                msg = str(e)
                is_oom = (
                    ("RESOURCE_EXHAUSTED" in msg)
                    or ("Out of memory" in msg)
                    or ("Allocator" in msg)
                )
                if not is_oom:
                    raise
                logging.warning(
                    "FISTA OOM detected; retrying with safer settings (vpb=1, unroll=1, stream)"
                )
                try:
                    recon_retry = True
                    x_out, info_rec = _run_fista(1, 1, cfg.gather_dtype, "stream")
                except Exception as e2:
                    msg2 = str(e2)
                    if (
                        ("RESOURCE_EXHAUSTED" in msg2)
                        or ("Out of memory" in msg2)
                        or ("Allocator" in msg2)
                    ):
                        logging.error(
                            "FISTA still OOM at finest level. Reduce memory pressure "
                            "(smaller problem size or lower internal batching), or "
                            "provide --recon-L to skip power-method."
                        )
                    raise
    else:
        x_out, info_rec = _run_spdhg()

    jax.block_until_ready(x_out)
    requested_backend = str(getattr(cfg, "projector_backend", "jax"))
    actual_recon_backend = (
        requested_backend
        if requested_backend == "pallas" and str(cfg.regulariser) == "huber_tv"
        else "jax"
    )
    stat: OuterStat = {
        "recon_time": time.perf_counter() - recon_start,
        "recon_retry": recon_retry,
        "recon_requested_backend": requested_backend,
        "recon_actual_backend": actual_recon_backend,
    }
    info_mapping = info_rec if isinstance(info_rec, Mapping) else {}
    L_next = _record_reconstruction_info(
        stat,
        info_rec=info_mapping,
        recon_algo=recon_algo,
        cfg=cfg,
        outer_idx=outer_idx,
        L_prev=L_prev,
    )
    return x_out, L_next, stat
