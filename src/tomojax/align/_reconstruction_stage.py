from __future__ import annotations

from collections.abc import Mapping
import functools
import logging
import time
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from tomojax.backends import estimate_views_per_batch_info
from tomojax.core.backend_policy import normalize_projector_backend
from tomojax.core.geometry.views import stack_view_poses
from tomojax.recon.fista_tv import FistaConfig, fista_tv
from tomojax.recon.fista_tv_core import FistaCoreConfig, fista_tv_core_arrays
from tomojax.recon.spdhg_tv import SPDHGConfig, spdhg_tv

from ._quality_policy import reconstruction_quality_policy
from ._results import record_reconstruction_info as _record_reconstruction_info
from .geometry.parametrizations import se3_from_5d
from .objectives.recon_layer import PoseAdjustedGeometry

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Geometry, Grid

    from ._observer import OuterStat


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


def _resolve_auto_views_per_batch(
    *,
    requested: int | None,
    n_views: int,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    gather_dtype: str,
    checkpoint_projector: bool,
    algo: str = "fista",
) -> int:
    requested_i = 0 if requested is None else int(requested)
    if requested_i > 0:
        return max(1, min(int(requested_i), int(n_views)))
    estimate = estimate_views_per_batch_info(
        n_views=int(n_views),
        grid_nxyz=(int(grid.nx), int(grid.ny), int(grid.nz)),
        det_nuv=(int(detector.nv), int(detector.nu)),
        gather_dtype=str(gather_dtype),
        projection_dtype="fp32",
        volume_dtype=str(volume.dtype),
        checkpoint_projector=bool(checkpoint_projector),
        algo=algo,
        fallback_batch=1,
    )
    return max(1, min(int(estimate.views_per_batch), int(n_views)))


def _resolve_reconstruction_projector_backend(
    *,
    requested_backend: str,
    T_all: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    gather_dtype: str,
    fallback_policy: str,
) -> tuple[str, str | None]:
    requested = normalize_projector_backend(requested_backend)
    if requested == "jax":
        return "jax", None
    try:
        backend = jax.default_backend()
    except Exception:
        backend = "unknown"
    if backend != "gpu":
        reason = f"pallas alignment reconstruction requires gpu backend; got {backend}"
        if str(fallback_policy) == "strict":
            raise RuntimeError(reason)
        return "jax", reason
    try:
        from tomojax.core.pallas_projector import (
            pallas_projector_sinogram_unsupported_reason,
        )

        reason = pallas_projector_sinogram_unsupported_reason(
            T_all,
            grid,
            detector,
            volume,
            gather_dtype=gather_dtype,
            det_grid=det_grid,
            state_mode="cached",
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
    if reason:
        if str(fallback_policy) == "strict":
            raise RuntimeError(reason)
        return "jax", reason
    return "pallas", None


def _is_oom_error_message(message: str) -> bool:
    return (
        ("RESOURCE_EXHAUSTED" in message)
        or ("Out of memory" in message)
        or ("Allocator" in message)
    )


def _finite_fraction(value: jnp.ndarray) -> float:
    array = jnp.asarray(value)
    if array.size == 0:
        return 0.0
    finite_fraction = jnp.mean(jnp.isfinite(array).astype(jnp.float32))
    return float(jax.device_get(finite_fraction))


def _reconstruction_step_stat(
    *,
    recon_start: float,
    recon_retry: bool,
    info_rec: object,
    recon_algo: str,
    cfg: object,
    outer_idx: int,
    L_prev: float | None,
) -> tuple[OuterStat, float | None]:
    info_mapping = info_rec if isinstance(info_rec, Mapping) else {}
    requested_backend = str(getattr(cfg, "projector_backend", "jax"))
    actual_recon_backend = str(info_mapping.get("actual_backend") or "jax")
    fallback_reason = info_mapping.get("fallback_reason")
    stat: OuterStat = {
        "recon_time": time.perf_counter() - recon_start,
        "recon_retry": recon_retry,
        "recon_requested_backend": requested_backend,
        "recon_actual_backend": actual_recon_backend,
        "recon_fallback_reason": str(fallback_reason) if fallback_reason else None,
        "align_profile": str(getattr(cfg, "align_profile", "lightning")),
        "quality_tier": str(getattr(cfg, "quality_tier", "")),
        "fallback_policy": str(getattr(cfg, "fallback_policy", "")),
        "regulariser": str(info_mapping.get("regulariser") or getattr(cfg, "regulariser", "")),
        "data_loss_computed": bool(info_mapping.get("data_loss_computed", False)),
        "regulariser_value_computed": bool(info_mapping.get("regulariser_value_computed", False)),
    }
    L_next = _record_reconstruction_info(
        stat,
        info_rec=info_mapping,
        recon_algo=recon_algo,
        cfg=cfg,
        outer_idx=outer_idx,
        L_prev=L_prev,
    )
    return stat, L_next


def _mark_nonfinite_reconstruction(
    stat: OuterStat,
    *,
    finite_fraction: float,
    reason: str,
) -> None:
    stat["reconstruction_failed"] = True
    stat["reconstruction_failure_reason"] = reason
    stat["reconstruction_finite_fraction"] = float(finite_fraction)


def _nonfinite_initial_reconstruction_result(
    *,
    x: jnp.ndarray,
    recon_start: float,
    recon_algo: str,
    cfg: object,
    outer_idx: int,
    L_prev: float | None,
    finite_fraction: float,
) -> tuple[jnp.ndarray, float | None, OuterStat]:
    stat, L_next = _reconstruction_step_stat(
        recon_start=recon_start,
        recon_retry=False,
        info_rec={},
        recon_algo=recon_algo,
        cfg=cfg,
        outer_idx=outer_idx,
        L_prev=L_prev,
    )
    _mark_nonfinite_reconstruction(
        stat,
        finite_fraction=finite_fraction,
        reason="nonfinite_initial_reconstruction",
    )
    return x, L_next, stat


def _retry_info_after_nonfinite_core(
    *,
    retry_info: Mapping[str, object],
    core_info: Mapping[str, object],
    core_finite_fraction: float,
    cfg: object,
) -> Mapping[str, object]:
    return {
        **dict(retry_info),
        "recon_nonfinite_retry": True,
        "nonfinite_core_finite_fraction": float(core_finite_fraction),
        "nonfinite_core_info": {
            "requested_backend": core_info.get("requested_backend"),
            "actual_backend": core_info.get("actual_backend"),
            "fallback_reason": core_info.get("fallback_reason"),
            "L": core_info.get("L"),
        },
        "fallback_reason": "huber_fista_core_nonfinite_retry_public_stream",
        "actual_backend": "jax",
        "requested_backend": str(getattr(cfg, "projector_backend", "jax")),
        "regulariser": str(getattr(cfg, "regulariser", "")),
        "data_loss_computed": True,
        "regulariser_value_computed": True,
    }


@functools.partial(jax.jit, static_argnames=("grid", "detector", "cfg"))
def _run_huber_fista_core_jit(
    x0: jnp.ndarray,
    T_all: jnp.ndarray,
    det_u: jnp.ndarray | None,
    det_v: jnp.ndarray | None,
    projections: jnp.ndarray,
    L_value: jnp.ndarray,
    *,
    grid: Grid,
    detector: Detector,
    cfg: FistaCoreConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    det_grid = None if det_u is None or det_v is None else (det_u, det_v)
    result = fista_tv_core_arrays(
        x0=x0,
        T_all=T_all,
        det_grid=det_grid,
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


def _run_reconstruction_step(  # noqa: PLR0915
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

    def _run_fista(
        vpb: int | None,
        unroll: int,
        gather: str,
        gm: str,
    ) -> tuple[jnp.ndarray, Mapping[str, object]]:
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

    def _run_spdhg() -> tuple[jnp.ndarray, Mapping[str, object]]:
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

    def _run_huber_fista_core() -> tuple[jnp.ndarray, Mapping[str, object]]:
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
        requested_backend = str(getattr(cfg, "projector_backend", "jax"))
        actual_backend, fallback_reason = _resolve_reconstruction_projector_backend(
            requested_backend=requested_backend,
            T_all=T_all,
            grid=grid,
            detector=detector,
            volume=x,
            det_grid=det_grid,
            gather_dtype=str(cfg.gather_dtype),
            fallback_policy=str(getattr(cfg, "fallback_policy", "fallback")),
        )
        quality_policy = reconstruction_quality_policy(str(getattr(cfg, "quality_tier", "fast")))
        core_cfg = FistaCoreConfig(
            iters=int(cfg.recon_iters),
            lambda_tv=float(cfg.lambda_tv),
            regulariser="huber_tv",
            huber_delta=float(cfg.huber_delta),
            L=1.0,
            checkpoint_projector=bool(cfg.checkpoint_projector),
            projector_unroll=int(cfg.projector_unroll),
            gather_dtype=str(cfg.gather_dtype),
            views_per_batch=_resolve_auto_views_per_batch(
                requested=int(cfg.views_per_batch),
                n_views=n_views,
                grid=grid,
                detector=detector,
                volume=x,
                gather_dtype=str(cfg.gather_dtype),
                checkpoint_projector=bool(cfg.checkpoint_projector),
                algo="fista",
            ),
            forward_projector=actual_backend,
            backprojector=actual_backend,
            compute_iteration_loss=bool(quality_policy.compute_iteration_loss),
            compute_final_data_loss=bool(quality_policy.compute_final_data_loss),
            compute_final_regulariser_value=bool(quality_policy.compute_final_regulariser_value),
        )

        det_grid_for_core = None if actual_backend == "pallas" else det_grid
        x_core, loss, data_loss, regulariser_value, effective_iters = _run_huber_fista_core_jit(
            x,
            T_all,
            None if det_grid_for_core is None else det_grid_for_core[0],
            None if det_grid_for_core is None else det_grid_for_core[1],
            projections,
            jnp.asarray(L_core, dtype=jnp.float32),
            grid=grid,
            detector=detector,
            cfg=core_cfg,
        )
        info = {
            "loss": loss,
            "loss_alias_only": not bool(quality_policy.compute_iteration_loss),
            "effective_iters": int(effective_iters),
            "early_stop": False,
            "regulariser": "huber_tv",
            "huber_delta": float(cfg.huber_delta),
            "data_loss_computed": bool(quality_policy.compute_final_data_loss),
            "regulariser_value_computed": bool(quality_policy.compute_final_regulariser_value),
            "quality_policy": quality_policy.to_dict(),
            "L": float(L_core / 1.2),
            "requested_backend": requested_backend,
            "actual_backend": actual_backend,
            "fallback_reason": fallback_reason,
        }
        return x_core, info

    vpb0 = _resolve_auto_views_per_batch(
        requested=int(cfg.views_per_batch),
        n_views=int(projections.shape[0]),
        grid=grid,
        detector=detector,
        volume=x,
        gather_dtype=str(cfg.gather_dtype),
        checkpoint_projector=bool(cfg.checkpoint_projector),
        algo="fista",
    )
    recon_retry = False
    recon_start = time.perf_counter()
    x_finite_fraction = _finite_fraction(x)
    if x_finite_fraction < 1.0:
        return _nonfinite_initial_reconstruction_result(
            x=x,
            recon_start=recon_start,
            recon_algo=recon_algo,
            cfg=cfg,
            outer_idx=outer_idx,
            L_prev=L_prev,
            finite_fraction=x_finite_fraction,
        )

    if recon_algo == "fista":
        if str(cfg.regulariser) == "huber_tv":
            x_out, info_rec = _run_huber_fista_core()
            jax.block_until_ready(x_out)
            core_finite_fraction = _finite_fraction(x_out)
            if core_finite_fraction < 1.0:
                logging.warning(
                    "Huber-FISTA core returned non-finite reconstruction "
                    "(finite_fraction=%.6g); retrying public streamed FISTA",
                    core_finite_fraction,
                )
                recon_retry = True
                core_info = dict(info_rec)
                x_out, info_rec = _run_fista(1, 1, "fp32", "stream")
                info_rec = _retry_info_after_nonfinite_core(
                    retry_info=info_rec,
                    core_info=core_info,
                    core_finite_fraction=core_finite_fraction,
                    cfg=cfg,
                )
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
                is_oom = _is_oom_error_message(msg)
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
                    if _is_oom_error_message(msg2):
                        logging.error(
                            "FISTA still OOM at finest level. Reduce memory pressure "
                            "(smaller problem size or lower internal batching), or "
                            "provide --recon-L to skip power-method."
                        )
                    raise
    else:
        x_out, info_rec = _run_spdhg()

    jax.block_until_ready(x_out)
    stat, L_next = _reconstruction_step_stat(
        recon_start=recon_start,
        recon_retry=recon_retry,
        info_rec=info_rec,
        recon_algo=recon_algo,
        cfg=cfg,
        outer_idx=outer_idx,
        L_prev=L_prev,
    )
    output_finite_fraction = _finite_fraction(x_out)
    stat["reconstruction_finite_fraction"] = float(output_finite_fraction)
    if isinstance(info_rec, Mapping) and bool(info_rec.get("recon_nonfinite_retry", False)):
        stat["recon_nonfinite_retry"] = True
        stat["nonfinite_core_finite_fraction"] = float(
            info_rec.get("nonfinite_core_finite_fraction", 0.0)
        )
    if output_finite_fraction < 1.0:
        _mark_nonfinite_reconstruction(
            stat,
            finite_fraction=output_finite_fraction,
            reason="nonfinite_reconstruction_after_retry"
            if recon_retry
            else "nonfinite_reconstruction",
        )
    return x_out, L_next, stat
