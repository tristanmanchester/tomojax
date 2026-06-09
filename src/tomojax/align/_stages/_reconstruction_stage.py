from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import functools
import hashlib
import logging
import time
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.align._geometry.parametrizations import se3_from_5d
from tomojax.align._objectives.recon_layer import PoseAdjustedGeometry
from tomojax.align._quality_policy import reconstruction_quality_policy
from tomojax.align._results import record_reconstruction_info as _record_reconstruction_info
from tomojax.backends import estimate_views_per_batch_info
from tomojax.core.backend_policy import normalize_projector_backend
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.pallas_resolver import resolve_pallas_callable
from tomojax.core.projector import get_detector_grid_device
from tomojax.recon.fista_tv import FistaConfig, fista_tv
from tomojax.recon.fista_tv_core import (
    FistaCoreConfig,
    effective_fista_core_backend,
    fista_tv_core_arrays,
)
from tomojax.recon.spdhg_tv import SPDHGConfig, spdhg_tv

if TYPE_CHECKING:
    from tomojax.align._observer import OuterStat
    from tomojax.core.geometry.base import Detector, Geometry, Grid


@dataclass(frozen=True, slots=True)
class _ReconstructionStepInputs:
    recon_geometry: Geometry
    grid: Grid
    detector: Detector
    projections: jnp.ndarray
    det_grid: tuple[jnp.ndarray, jnp.ndarray]
    x: jnp.ndarray
    cfg: object
    L_prev: float | None
    outer_idx: int


@dataclass(frozen=True, slots=True)
class _HuberFistaBackendPlan:
    T_all: jnp.ndarray
    requested_backend: str
    actual_backend: str
    fallback_reason: str | None
    detector_grid_folded_into_pose: bool
    detector_grid_fold_reason: str | None


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
    support_fn, reason = resolve_pallas_callable(
        "pallas_projector_sinogram_unsupported_reason",
        missing_reason="pallas_sinogram_support_check_missing",
    )
    if support_fn is not None:
        try:
            options_cls, options_reason = resolve_pallas_callable(
                "PallasProjectorOptions",
                missing_reason="pallas_projector_options_missing",
            )
            if options_cls is None:
                reason = options_reason
            else:
                options = options_cls(
                    gather_dtype=gather_dtype,
                    det_grid=det_grid,
                    state_mode="cached",
                )
                reason = support_fn(
                    T_all,
                    grid,
                    detector,
                    volume,
                    options=options,
                )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
    if reason:
        if str(fallback_policy) == "strict":
            raise RuntimeError(reason)
        return "jax", reason
    return "pallas", None


def _is_oom_error_message(message: str) -> bool:
    msg = message.lower()
    return any(term in msg for term in ("resource_exhausted", "out of memory", "allocator"))


def _rigid_detector_grid_transform(
    detector: Detector,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    *,
    atol: float = 2e-4,
) -> tuple[float, float, float, float, float, float] | None:
    if det_grid is None:
        return None
    try:
        X0, Z0 = get_detector_grid_device(detector)
        X0_host = np.asarray(X0, dtype=np.float32).reshape(-1)
        Z0_host = np.asarray(Z0, dtype=np.float32).reshape(-1)
        X_host = np.asarray(det_grid[0], dtype=np.float32).reshape(-1)
        Z_host = np.asarray(det_grid[1], dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if X_host.shape != X0_host.shape or Z_host.shape != Z0_host.shape:
        return None

    design = np.stack(
        [X0_host, Z0_host, np.ones_like(X0_host, dtype=np.float32)],
        axis=1,
    )
    try:
        x_coeffs, *_ = np.linalg.lstsq(design, X_host, rcond=None)
        z_coeffs, *_ = np.linalg.lstsq(design, Z_host, rcond=None)
    except np.linalg.LinAlgError:
        return None
    x_fit = design @ x_coeffs
    z_fit = design @ z_coeffs
    if float(np.max(np.abs(x_fit - X_host), initial=0.0)) > float(atol) or float(
        np.max(np.abs(z_fit - Z_host), initial=0.0)
    ) > float(atol):
        return None

    a, b, c = (float(v) for v in x_coeffs)
    d, e, f = (float(v) for v in z_coeffs)
    transform = np.asarray([[a, b], [d, e]], dtype=np.float32)
    gram = transform.T @ transform
    det = float(np.linalg.det(transform))
    if not (
        np.allclose(gram, np.eye(2, dtype=np.float32), atol=2e-4, rtol=2e-4)
        and abs(det - 1.0) <= 2e-4
    ):
        return None
    return a, b, c, d, e, f


def _fold_rigid_detector_grid_into_pose_stack(
    T_all: jnp.ndarray,
    detector: Detector,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> jnp.ndarray | None:
    transform = _rigid_detector_grid_transform(detector, det_grid)
    if transform is None:
        return None
    a, b, c, d, e, f = transform
    T = jnp.asarray(T_all, dtype=jnp.float32)
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    Rinv = jnp.swapaxes(R, 1, 2)
    tinv = -jnp.einsum("nij,nj->ni", Rinv, t)

    col0 = Rinv[:, :, 0] * jnp.float32(a) + Rinv[:, :, 2] * jnp.float32(d)
    col1 = Rinv[:, :, 1]
    col2 = Rinv[:, :, 0] * jnp.float32(b) + Rinv[:, :, 2] * jnp.float32(e)
    tinv_folded = tinv + Rinv[:, :, 0] * jnp.float32(c) + Rinv[:, :, 2] * jnp.float32(f)
    Rinv_folded = jnp.stack([col0, col1, col2], axis=2)
    R_folded = jnp.swapaxes(Rinv_folded, 1, 2)
    t_folded = -jnp.einsum("nij,nj->ni", R_folded, tinv_folded)

    bottom = jnp.broadcast_to(
        jnp.asarray([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32),
        (int(T.shape[0]), 1, 4),
    )
    upper = jnp.concatenate([R_folded, t_folded[:, :, None]], axis=2)
    return jnp.concatenate([upper, bottom], axis=1)


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
    if bool(info_mapping.get("recon_public_fista_fallback", False)):
        stat["recon_public_fista_fallback"] = True
    if bool(info_mapping.get("detector_grid_folded_into_pose", False)):
        stat["detector_grid_folded_into_pose"] = True
        stat["detector_grid_fold_reason"] = info_mapping.get("detector_grid_fold_reason")
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


def _skipped_fixed_volume_reconstruction_result(
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
        info_rec={
            "loss": [],
            "effective_iters": 0,
            "early_stop": False,
            "regulariser": str(getattr(cfg, "regulariser", "")),
            "fixed_volume_reconstruction_skipped": True,
        },
        recon_algo=recon_algo,
        cfg=cfg,
        outer_idx=outer_idx,
        L_prev=L_prev,
    )
    stat["fixed_volume_reconstruction_skipped"] = True
    stat["reconstruction_finite_fraction"] = float(finite_fraction)
    return x, L_next, stat


def _retry_info_after_nonfinite_core(
    *,
    retry_info: Mapping[str, object],
    core_info: Mapping[str, object],
    core_finite_fraction: float,
    cfg: object,
) -> Mapping[str, object]:
    quality_policy = reconstruction_quality_policy(str(getattr(cfg, "quality_tier", "fast")))
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
        "iteration_loss_computed": bool(quality_policy.compute_iteration_loss),
        "data_loss_computed": bool(quality_policy.compute_final_data_loss),
        "regulariser_value_computed": bool(quality_policy.compute_final_regulariser_value),
        "quality_policy": quality_policy.to_dict(),
    }


def _public_fista_info_after_core_bypass(
    *,
    public_info: Mapping[str, object],
    fallback_reason: str,
    cfg: object,
) -> Mapping[str, object]:
    quality_policy = reconstruction_quality_policy(str(getattr(cfg, "quality_tier", "fast")))
    return {
        **dict(public_info),
        "recon_public_fista_fallback": True,
        "fallback_reason": fallback_reason,
        "actual_backend": "jax",
        "requested_backend": str(getattr(cfg, "projector_backend", "jax")),
        "regulariser": str(getattr(cfg, "regulariser", "")),
        "iteration_loss_computed": bool(quality_policy.compute_iteration_loss),
        "data_loss_computed": bool(quality_policy.compute_final_data_loss),
        "regulariser_value_computed": bool(quality_policy.compute_final_regulariser_value),
        "quality_policy": quality_policy.to_dict(),
    }


@functools.partial(jax.jit, static_argnames=("grid", "detector", "cfg"))
def _run_huber_fista_core_dynamic_geometry(
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


def _run_huber_fista_core_fixed_geometry(
    x0: jnp.ndarray,
    T_all: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    projections: jnp.ndarray,
    L_value: jnp.ndarray,
    *,
    grid: Grid,
    detector: Detector,
    cfg: FistaCoreConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    runner = _cached_huber_fista_core_runner(
        _array_cache_key(T_all),
        None if det_grid is None else tuple(_array_cache_key(v) for v in det_grid),
        grid,
        detector,
        cfg,
    )
    return runner(x0, projections, L_value)


def _array_cache_key(array: jnp.ndarray) -> tuple[tuple[int, ...], str, bytes, str]:
    arr = np.asarray(array)
    contiguous = np.ascontiguousarray(arr)
    data = contiguous.tobytes()
    digest = hashlib.sha256(data).hexdigest()
    return tuple(int(v) for v in contiguous.shape), str(contiguous.dtype), data, digest


@functools.lru_cache(maxsize=16)
def _cached_huber_fista_core_runner(
    T_key: tuple[tuple[int, ...], str, bytes, str],
    det_grid_key: tuple[tuple[tuple[int, ...], str, bytes, str], ...] | None,
    grid: Grid,
    detector: Detector,
    cfg: FistaCoreConfig,
) -> object:
    T_all = _array_from_cache_key(T_key)
    det_grid = (
        None
        if det_grid_key is None
        else (_array_from_cache_key(det_grid_key[0]), _array_from_cache_key(det_grid_key[1]))
    )

    def run(
        x_init: jnp.ndarray,
        y: jnp.ndarray,
        L_current: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        result = fista_tv_core_arrays(
            x0=x_init,
            T_all=T_all,
            det_grid=det_grid,
            projections=y,
            grid=grid,
            detector=detector,
            cfg=cfg,
            L_override=L_current,
        )
        return (
            result.x,
            result.loss,
            result.data_loss,
            result.regulariser_value,
            result.effective_iters,
        )

    return jax.jit(run)


def _array_from_cache_key(key: tuple[tuple[int, ...], str, bytes, str]) -> jnp.ndarray:
    shape, dtype_name, data, _digest = key
    arr = np.frombuffer(data, dtype=np.dtype(dtype_name)).reshape(shape).copy()
    return jnp.asarray(arr)


def _run_public_fista_reconstruction(
    step: _ReconstructionStepInputs,
    *,
    views_per_batch: int | None,
    projector_unroll: int,
    gather_dtype: str,
    grad_mode: str,
) -> tuple[jnp.ndarray, Mapping[str, object]]:
    cfg = step.cfg
    fista_cfg = FistaConfig(
        iters=cfg.recon_iters,
        lambda_tv=cfg.lambda_tv,
        regulariser=cfg.regulariser,
        huber_delta=cfg.huber_delta,
        L=step.L_prev,
        views_per_batch=views_per_batch,
        projector_unroll=int(projector_unroll),
        checkpoint_projector=cfg.checkpoint_projector,
        gather_dtype=gather_dtype,
        grad_mode=grad_mode,
        tv_prox_iters=int(cfg.tv_prox_iters),
        positivity=bool(cfg.recon_positivity),
        recon_rel_tol=cfg.recon_rel_tol,
        recon_patience=(int(cfg.recon_patience) if cfg.recon_patience is not None else 0),
    )
    return fista_tv(
        step.recon_geometry,
        step.grid,
        step.detector,
        step.projections,
        init_x=step.x,
        config=fista_cfg,
        det_grid=step.det_grid,
    )


def _huber_fista_pose_stack(step: _ReconstructionStepInputs, *, n_views: int) -> jnp.ndarray:
    if isinstance(step.recon_geometry, PoseAdjustedGeometry):
        nominal = stack_view_poses(step.recon_geometry.geometry, n_views)
        return nominal @ jax.vmap(se3_from_5d)(step.recon_geometry.params5)
    return stack_view_poses(step.recon_geometry, n_views)


def _resolve_huber_fista_backend_plan(
    step: _ReconstructionStepInputs,
    *,
    T_all: jnp.ndarray,
) -> _HuberFistaBackendPlan:
    cfg = step.cfg
    requested_backend = str(getattr(cfg, "projector_backend", "jax"))
    actual_backend, fallback_reason = _resolve_reconstruction_projector_backend(
        requested_backend=requested_backend,
        T_all=T_all,
        grid=step.grid,
        detector=step.detector,
        volume=step.x,
        det_grid=step.det_grid,
        gather_dtype=str(cfg.gather_dtype),
        fallback_policy=str(getattr(cfg, "fallback_policy", "fallback")),
    )
    detector_grid_folded_into_pose = False
    detector_grid_fold_reason = None
    if (
        actual_backend == "jax"
        and fallback_reason
        and requested_backend == "pallas"
        and bool(getattr(cfg, "fold_rigid_detector_grid", True))
    ):
        T_folded = _fold_rigid_detector_grid_into_pose_stack(
            T_all,
            step.detector,
            step.det_grid,
        )
        if T_folded is not None:
            folded_backend, folded_fallback_reason = _resolve_reconstruction_projector_backend(
                requested_backend=requested_backend,
                T_all=T_folded,
                grid=step.grid,
                detector=step.detector,
                volume=step.x,
                det_grid=None,
                gather_dtype=str(cfg.gather_dtype),
                fallback_policy=str(getattr(cfg, "fallback_policy", "fallback")),
            )
            if folded_backend == "pallas":
                return _HuberFistaBackendPlan(
                    T_all=T_folded,
                    requested_backend=requested_backend,
                    actual_backend=folded_backend,
                    fallback_reason=folded_fallback_reason,
                    detector_grid_folded_into_pose=True,
                    detector_grid_fold_reason=str(fallback_reason),
                )
    return _HuberFistaBackendPlan(
        T_all=T_all,
        requested_backend=requested_backend,
        actual_backend=actual_backend,
        fallback_reason=fallback_reason,
        detector_grid_folded_into_pose=detector_grid_folded_into_pose,
        detector_grid_fold_reason=detector_grid_fold_reason,
    )


def _run_public_fista_core_bypass(
    step: _ReconstructionStepInputs,
    *,
    fallback_reason: str,
) -> tuple[jnp.ndarray, Mapping[str, object]]:
    x_public, public_info = _run_public_fista_reconstruction(
        step,
        views_per_batch=1,
        projector_unroll=1,
        gather_dtype="fp32",
        grad_mode="stream",
    )
    return x_public, _public_fista_info_after_core_bypass(
        public_info=public_info,
        fallback_reason=fallback_reason,
        cfg=step.cfg,
    )


def _huber_fista_core_config(
    step: _ReconstructionStepInputs,
    *,
    n_views: int,
    actual_backend: str,
    quality_policy: object,
) -> FistaCoreConfig:
    cfg = step.cfg
    return FistaCoreConfig(
        iters=int(cfg.recon_iters),
        lambda_tv=float(cfg.lambda_tv),
        regulariser="huber_tv",
        huber_delta=float(cfg.huber_delta),
        L=1.0,
        positivity=bool(cfg.recon_positivity),
        checkpoint_projector=bool(cfg.checkpoint_projector),
        projector_unroll=int(cfg.projector_unroll),
        gather_dtype=str(cfg.gather_dtype),
        views_per_batch=_resolve_auto_views_per_batch(
            requested=int(cfg.views_per_batch),
            n_views=n_views,
            grid=step.grid,
            detector=step.detector,
            volume=step.x,
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


def _huber_fista_core_info(
    *,
    cfg: object,
    loss: jnp.ndarray,
    effective_iters: jnp.ndarray,
    L_core: float,
    quality_policy: object,
    backend_plan: _HuberFistaBackendPlan,
    effective_backend: str | None = None,
) -> dict[str, object]:
    return {
        "loss": loss,
        "iteration_loss_computed": bool(quality_policy.compute_iteration_loss),
        "effective_iters": int(effective_iters),
        "early_stop": False,
        "regulariser": "huber_tv",
        "huber_delta": float(cfg.huber_delta),
        "data_loss_computed": bool(quality_policy.compute_final_data_loss),
        "regulariser_value_computed": bool(quality_policy.compute_final_regulariser_value),
        "quality_policy": quality_policy.to_dict(),
        "L": float(L_core),
        "requested_backend": backend_plan.requested_backend,
        "actual_backend": effective_backend or backend_plan.actual_backend,
        "fallback_reason": backend_plan.fallback_reason,
        "detector_grid_folded_into_pose": backend_plan.detector_grid_folded_into_pose,
        "detector_grid_fold_reason": backend_plan.detector_grid_fold_reason,
    }


def _run_spdhg_reconstruction(
    step: _ReconstructionStepInputs,
) -> tuple[jnp.ndarray, Mapping[str, object]]:
    cfg = step.cfg
    spdhg_cfg = SPDHGConfig(
        iters=int(cfg.recon_iters),
        lambda_tv=float(cfg.lambda_tv),
        regulariser=cfg.regulariser,
        huber_delta=float(cfg.huber_delta),
        views_per_batch=max(1, int(cfg.views_per_batch)),
        seed=int(cfg.spdhg_seed) + int(step.outer_idx) - 1,
        projector_unroll=int(cfg.projector_unroll),
        checkpoint_projector=cfg.checkpoint_projector,
        gather_dtype=cfg.gather_dtype,
        positivity=bool(cfg.recon_positivity),
        log_every=1,
    )
    return spdhg_tv(
        step.recon_geometry,
        step.grid,
        step.detector,
        step.projections,
        init_x=step.x,
        config=spdhg_cfg,
        det_grid=step.det_grid,
    )


def _run_huber_fista_core_reconstruction(
    step: _ReconstructionStepInputs,
) -> tuple[jnp.ndarray, Mapping[str, object]]:
    cfg = step.cfg
    n_views = int(step.projections.shape[0])
    L_core = (
        float(step.L_prev)
        if step.L_prev is not None
        else _heuristic_projection_lipschitz(
            n_views=n_views,
            grid=step.grid,
            lambda_tv=float(cfg.lambda_tv),
            huber_delta=float(cfg.huber_delta),
        )
    )
    backend_plan = _resolve_huber_fista_backend_plan(
        step,
        T_all=_huber_fista_pose_stack(step, n_views=n_views),
    )
    is_pose_adjusted = isinstance(step.recon_geometry, PoseAdjustedGeometry)
    if is_pose_adjusted and backend_plan.actual_backend == "pallas":
        backend_plan = _HuberFistaBackendPlan(
            T_all=backend_plan.T_all,
            requested_backend=backend_plan.requested_backend,
            actual_backend="jax",
            fallback_reason="dynamic_geometry_alignment_uses_jax_core",
            detector_grid_folded_into_pose=backend_plan.detector_grid_folded_into_pose,
            detector_grid_fold_reason=backend_plan.detector_grid_fold_reason,
        )
    if (
        backend_plan.actual_backend == "jax"
        and backend_plan.fallback_reason
        and not is_pose_adjusted
    ):
        return _run_public_fista_core_bypass(
            step,
            fallback_reason=backend_plan.fallback_reason,
        )
    quality_policy = reconstruction_quality_policy(str(getattr(cfg, "quality_tier", "fast")))
    core_cfg = _huber_fista_core_config(
        step,
        n_views=n_views,
        actual_backend=backend_plan.actual_backend,
        quality_policy=quality_policy,
    )

    det_grid_for_core = (
        None
        if backend_plan.actual_backend == "pallas" or backend_plan.detector_grid_folded_into_pose
        else step.det_grid
    )
    effective_backend = backend_plan.actual_backend
    if not is_pose_adjusted:
        effective_backend = effective_fista_core_backend(
            core_cfg,
            T_all=backend_plan.T_all,
            grid=step.grid,
            detector=step.detector,
            volume=step.x,
            det_grid=det_grid_for_core,
        )
    if isinstance(step.recon_geometry, PoseAdjustedGeometry):
        x_core, loss, _data_loss, _regulariser_value, effective_iters = (
            _run_huber_fista_core_dynamic_geometry(
                step.x,
                backend_plan.T_all,
                None if det_grid_for_core is None else det_grid_for_core[0],
                None if det_grid_for_core is None else det_grid_for_core[1],
                step.projections,
                jnp.asarray(L_core, dtype=jnp.float32),
                grid=step.grid,
                detector=step.detector,
                cfg=core_cfg,
            )
        )
    else:
        x_core, loss, _data_loss, _regulariser_value, effective_iters = (
            _run_huber_fista_core_fixed_geometry(
                step.x,
                backend_plan.T_all,
                det_grid_for_core,
                step.projections,
                jnp.asarray(L_core, dtype=jnp.float32),
                grid=step.grid,
                detector=step.detector,
                cfg=core_cfg,
            )
        )
    return x_core, _huber_fista_core_info(
        cfg=cfg,
        loss=loss,
        effective_iters=effective_iters,
        L_core=L_core,
        quality_policy=quality_policy,
        backend_plan=backend_plan,
        effective_backend=effective_backend,
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
    step = _ReconstructionStepInputs(
        recon_geometry=recon_geometry,
        grid=grid,
        detector=detector,
        projections=projections,
        det_grid=det_grid,
        x=x,
        cfg=cfg,
        L_prev=L_prev,
        outer_idx=outer_idx,
    )

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
    if int(getattr(cfg, "recon_iters", 0)) <= 0:
        return _skipped_fixed_volume_reconstruction_result(
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
            x_out, info_rec = _run_huber_fista_core_reconstruction(step)
            if bool(info_rec.get("recon_public_fista_fallback", False)):
                recon_retry = True
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
                x_out, info_rec = _run_public_fista_reconstruction(
                    step,
                    views_per_batch=1,
                    projector_unroll=1,
                    gather_dtype="fp32",
                    grad_mode="stream",
                )
                info_rec = _retry_info_after_nonfinite_core(
                    retry_info=info_rec,
                    core_info=core_info,
                    core_finite_fraction=core_finite_fraction,
                    cfg=cfg,
                )
        else:
            try:
                x_out, info_rec = _run_public_fista_reconstruction(
                    step,
                    views_per_batch=vpb0,
                    projector_unroll=int(cfg.projector_unroll),
                    gather_dtype=cfg.gather_dtype,
                    grad_mode="auto",
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
                    x_out, info_rec = _run_public_fista_reconstruction(
                        step,
                        views_per_batch=1,
                        projector_unroll=1,
                        gather_dtype=cfg.gather_dtype,
                        grad_mode="stream",
                    )
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
        x_out, info_rec = _run_spdhg_reconstruction(step)

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
