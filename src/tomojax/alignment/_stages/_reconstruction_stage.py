from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
import functools
import logging
import time
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.alignment._geometry.parametrizations import apply_pose_updates
from tomojax.alignment._objectives.fixed_volume import cuda_sinogram_fallback_reason
from tomojax.alignment._objectives.recon_layer import PoseAdjustedGeometry
from tomojax.alignment._profiles import (
    ReconstructionQualityPolicy,
    reconstruction_quality_policy,
    scaled_reconstruction_iterations,
)
from tomojax.alignment._reconstruction_config import fista_config
from tomojax.alignment._results import record_reconstruction_info as _record_reconstruction_info
from tomojax.backends import estimate_views_per_batch_info
from tomojax.backends.host_stream import host_source, should_stream
from tomojax.core.backend_policy import normalize_projector_backend
from tomojax.core.devices import shared_devices
from tomojax.core.geometry.cone import is_cone_beam
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.operator_norm import estimate_normal_norm
from tomojax.core.projector import get_detector_grid_device
from tomojax.core.volume_layout import transpose_volume
from tomojax.recon.fista_tv import fista_tv
from tomojax.recon.fista_tv_core import (
    FistaCoreConfig,
    fista_tv_core_iterates,
    initial_iterate,
)
from tomojax.recon.spdhg_tv import SPDHGConfig, spdhg_tv

if TYPE_CHECKING:
    from tomojax.alignment._config import AlignConfig
    from tomojax.alignment._observer import OuterStat
    from tomojax.core.geometry.base import Detector, Geometry, Grid


@dataclass(frozen=True, slots=True)
class _ReconstructionStepInputs:
    recon_geometry: PoseAdjustedGeometry
    grid: Grid
    detector: Detector
    projections: jnp.ndarray
    det_grid: tuple[jnp.ndarray, jnp.ndarray]
    x: jnp.ndarray
    cfg: object
    L_prev: float | None
    outer_idx: int


def _initial_projection_lipschitz(
    step: _ReconstructionStepInputs,
    poses: jnp.ndarray,
) -> float:
    cfg = step.cfg
    projection_l = float(
        estimate_normal_norm(
            poses,
            step.x,
            step.det_grid,
            None,
            grid=step.grid,
            detector=step.detector,
            batch_size=max(1, int(cfg.views_per_batch)),
            iterations=1,
            unroll=1,
            checkpoint=bool(cfg.checkpoint_projector),
            gather_dtype=str(cfg.gather_dtype),
            upper_bound=True,
            ray_integrator=getattr(cfg, "ray_integrator", "sampled"),
        )
    )
    return 1.2 * projection_l


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
) -> tuple[str, str | None]:
    requested = normalize_projector_backend(requested_backend)
    if requested == "jax":
        return "jax", None
    try:
        backend = jax.default_backend()
    except Exception:
        backend = "unknown"
    if backend != "gpu":
        return "jax", f"cuda alignment reconstruction requires gpu backend; got {backend}"
    reason = cuda_sinogram_fallback_reason(
        pose_stack=T_all,
        grid=grid,
        detector=detector,
        volume=volume,
        det_grid=det_grid,
        gather_dtype=gather_dtype,
    )
    if reason:
        return "jax", reason
    return "cuda", None


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
    # A float32 mean of all-True values can round below 1, rejecting a valid
    # reconstruction. Count failures and divide on the host: zero failures is
    # exactly 1, and a single failure remains distinguishable on large volumes.
    failures = jnp.sum(~jnp.isfinite(array), dtype=jnp.float32)
    return max(0.0, 1.0 - float(jax.device_get(failures)) / array.size)


def _reconstruction_step_stat(
    *,
    recon_start: float,
    recon_retry: bool,
    info_rec: object,
    reconstruction: str,
    cfg: object,
    outer_idx: int,
    L_prev: float | None,
) -> tuple[OuterStat, float | None]:
    info_mapping = info_rec if isinstance(info_rec, Mapping) else {}
    requested_backend = str(getattr(cfg, "backend", "jax"))
    actual_recon_backend = str(info_mapping.get("actual_backend") or "jax")
    fallback_reason = info_mapping.get("fallback_reason")
    stat: OuterStat = {
        "recon_time": time.perf_counter() - recon_start,
        "recon_retry": recon_retry,
        "recon_requested_backend": requested_backend,
        "recon_actual_backend": actual_recon_backend,
        "recon_fallback_reason": str(fallback_reason) if fallback_reason else None,
        "quality": str(getattr(cfg, "quality", "fast")),
        "quality_tier": str(getattr(cfg, "stage_quality_tier", "")),
        "regulariser": "none",
        "data_loss_computed": bool(info_mapping.get("data_loss_computed", False)),
        "regulariser_value_computed": bool(info_mapping.get("regulariser_value_computed", False)),
    }
    if bool(info_mapping.get("detector_grid_folded_into_pose", False)):
        stat["detector_grid_folded_into_pose"] = True
        stat["detector_grid_fold_reason"] = info_mapping.get("detector_grid_fold_reason")
    L_next = _record_reconstruction_info(
        stat,
        info_rec=info_mapping,
        reconstruction=reconstruction,
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


def _retry_info_after_nonfinite_core(
    *,
    retry_info: Mapping[str, object],
    core_info: Mapping[str, object],
    core_finite_fraction: float,
    cfg: object,
) -> Mapping[str, object]:
    quality_policy = reconstruction_quality_policy(str(getattr(cfg, "stage_quality_tier", "fast")))
    return {
        **dict(retry_info),
        "recon_nonfinite_retry": True,
        "nonfinite_core_finite_fraction": float(core_finite_fraction),
        "nonfinite_core_info": {
            "requested_backend": core_info.get("requested_backend"),
            "actual_backend": core_info.get("actual_backend"),
            "fallback_reason": core_info.get("fallback_reason"),
            "lipschitz": core_info.get("lipschitz"),
        },
        "fallback_reason": "fista_core_nonfinite_retry_public_stream",
        "actual_backend": "jax",
        "requested_backend": str(getattr(cfg, "backend", "jax")),
        "regulariser": "none",
        "iteration_loss_computed": bool(quality_policy.compute_iteration_loss),
        "data_loss_computed": bool(quality_policy.compute_final_data_loss),
        "regulariser_value_computed": bool(quality_policy.compute_final_regulariser_value),
        "quality_policy": quality_policy.to_dict(),
    }


_initial_core_iterate = jax.jit(initial_iterate, static_argnames="cfg")


@functools.partial(jax.jit, static_argnames=("grid", "detector", "cfg"), donate_argnums=(0, 1))
def _run_fista_core_dynamic_geometry(
    x: jnp.ndarray,
    z: jnp.ndarray,
    T_all: jnp.ndarray,
    det_u: jnp.ndarray | None,
    det_v: jnp.ndarray | None,
    projections: jnp.ndarray,
    L_value: jnp.ndarray,
    *,
    grid: Grid,
    detector: Detector,
    cfg: FistaCoreConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    det_grid = None if det_u is None or det_v is None else (det_u, det_v)
    # The two owned iterates' buffers are reused by the loop rather than copied.
    result, z = fista_tv_core_iterates(
        x,
        z,
        T_all=T_all,
        det_grid=det_grid,
        projections=projections,
        grid=grid,
        detector=detector,
        cfg=cfg,
        L_override=L_value,
        view_weights=None,
        owned=True,
    )
    return (
        result.x,
        z,
        result.loss,
        result.data_loss,
        result.regulariser_value,
        result.effective_iterations,
    )


def _run_public_fista_reconstruction(
    step: _ReconstructionStepInputs,
    *,
    views_per_batch: int | None,
    gather_dtype: str,
    grad_mode: str,
) -> tuple[jnp.ndarray, Mapping[str, object]]:
    cfg = step.cfg
    quality_policy = reconstruction_quality_policy(str(getattr(cfg, "stage_quality_tier", "fast")))
    cone = is_cone_beam(step.recon_geometry)
    fista_cfg = fista_config(
        cast("AlignConfig", cfg),
        projector_model="auto" if cone else "ray",  # cone beams: Joseph, on CUDA if there
        backend="auto" if cone else "jax",
        iterations=scaled_reconstruction_iterations(cfg.iterations, quality_policy),
        lipschitz=step.L_prev,
        views_per_batch=views_per_batch,
        checkpoint_projector=cfg.checkpoint_projector,
        gather_dtype=gather_dtype,
        grad_mode=grad_mode,
        devices=shared_devices() if cone else None,  # the run's, see tomojax.align
    )
    volume, info = fista_tv(
        cast("Geometry", step.recon_geometry),
        step.grid,
        step.detector,
        step.projections,
        init_x=step.x,
        config=fista_cfg,
        det_grid=None if cone else step.det_grid,
    )

    return jnp.asarray(volume), info


def _fista_backend(
    step: _ReconstructionStepInputs, poses: jnp.ndarray
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray] | None, dict[str, object]]:
    """The core's poses and detector grid, and a record of its backend.

    CUDA takes no detector grid, so a rigid one is folded into the poses. The
    sampled integrator keeps the JAX core, with any folded poses.
    """
    cfg = step.cfg
    requested = str(getattr(cfg, "backend", "jax"))

    def resolve(T_all: jnp.ndarray, det_grid: tuple[jnp.ndarray, jnp.ndarray] | None):
        return _resolve_reconstruction_projector_backend(
            requested_backend=requested,
            T_all=T_all,
            grid=step.grid,
            detector=step.detector,
            volume=step.x,
            det_grid=det_grid,
            gather_dtype=str(cfg.gather_dtype),
        )

    backend, reason = resolve(poses, step.det_grid)
    fold_reason = None
    if backend == "jax" and reason and requested == "cuda":
        folded = _fold_rigid_detector_grid_into_pose_stack(poses, step.detector, step.det_grid)
        if folded is not None and resolve(folded, None)[0] == "cuda":
            poses, backend, reason, fold_reason = folded, "cuda", None, str(reason)
    if backend == "cuda" and getattr(cfg, "ray_integrator", "sampled") == "sampled":
        backend, reason = "jax", "dynamic_geometry_alignment_uses_jax_core"
    det_grid = None if backend == "cuda" or fold_reason is not None else step.det_grid
    return (
        poses,
        det_grid,
        {
            "requested_backend": requested,
            "actual_backend": backend,
            "fallback_reason": reason,
            "detector_grid_folded_into_pose": fold_reason is not None,
            "detector_grid_fold_reason": fold_reason,
        },
    )


def _fista_core_config(
    step: _ReconstructionStepInputs,
    *,
    n_views: int,
    actual_backend: str,
    quality_policy: ReconstructionQualityPolicy,
) -> FistaCoreConfig:
    cfg = step.cfg
    return FistaCoreConfig(
        iterations=scaled_reconstruction_iterations(cfg.iterations, quality_policy),
        tv_weight=0.0,
        regulariser="huber_tv",
        lipschitz=1.0,
        nonnegative=bool(cfg.nonnegative),
        checkpoint_projector=bool(cfg.checkpoint_projector),
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
        ray_integrator=getattr(cfg, "ray_integrator", "sampled"),
    )


def _run_spdhg_reconstruction(
    step: _ReconstructionStepInputs,
) -> tuple[jnp.ndarray, Mapping[str, object]]:
    cfg = step.cfg
    quality_policy = reconstruction_quality_policy(str(getattr(cfg, "stage_quality_tier", "fast")))
    spdhg_cfg = SPDHGConfig(
        iterations=scaled_reconstruction_iterations(cfg.iterations, quality_policy),
        tv_weight=0.0,
        regulariser="huber_tv",
        views_per_batch=max(1, int(cfg.views_per_batch)),
        seed=int(cfg.seed) + int(step.outer_idx) - 1,
        projector_unroll=1,
        checkpoint_projector=cfg.checkpoint_projector,
        gather_dtype=cfg.gather_dtype,
        nonnegative=bool(cfg.nonnegative),
        log_every=1,
        ray_integrator=getattr(cfg, "ray_integrator", "sampled"),
    )
    return spdhg_tv(
        cast("Geometry", step.recon_geometry),
        step.grid,
        step.detector,
        step.projections,
        init_x=step.x,
        config=spdhg_cfg,
        det_grid=None if is_cone_beam(step.recon_geometry) else step.det_grid,
    )


def _run_fista_core_reconstruction(
    step: _ReconstructionStepInputs,
) -> tuple[jnp.ndarray, Mapping[str, object]]:
    """Reconstruct with a six-image reserve for iterates, gradient and TV scratch.

    Measurement-only admission can otherwise fill the device before the first
    FISTA update allocates its momentum and transpose workspace.
    """
    cfg = step.cfg
    n_views = int(step.projections.shape[0])
    geometry = step.recon_geometry
    poses = apply_pose_updates(
        stack_view_poses(geometry.geometry, n_views),
        geometry.pose_params,
        translation_frame=geometry.translation_frame,
    )
    # The bound is the operator's, before any detector grid is folded into the poses.
    L_core = (
        float(step.L_prev)
        if step.L_prev is not None
        else _initial_projection_lipschitz(step, poses)
    )
    poses, det_grid_for_core, backend = _fista_backend(step, poses)
    quality_policy = reconstruction_quality_policy(str(getattr(cfg, "stage_quality_tier", "fast")))
    core_cfg = _fista_core_config(
        step,
        n_views=n_views,
        actual_backend=str(backend["actual_backend"]),
        quality_policy=quality_policy,
    )
    stream = should_stream(step.projections, reserve_bytes=6 * step.x.nbytes)
    core_cfg = replace(core_cfg, stream_projections=stream)
    # Each owned iterate is complete before the next allocation, so their
    # workspaces never overlap the solve's.
    x = jax.block_until_ready(_initial_core_iterate(step.x, cfg=core_cfg))
    run = functools.partial(
        _run_fista_core_dynamic_geometry,
        x,
        jax.block_until_ready(jnp.array(x, copy=True)),
        poses,
        None if det_grid_for_core is None else det_grid_for_core[0],
        None if det_grid_for_core is None else det_grid_for_core[1],
    )
    source = host_source(np.asarray(step.projections)) if stream else nullcontext(step.projections)
    with source as data:
        result = run(
            data,
            jnp.asarray(L_core, dtype=jnp.float32),
            grid=step.grid,
            detector=step.detector,
            cfg=core_cfg,
        )
        # Host sources cannot be unregistered while asynchronous reads remain.
        if stream:
            jax.block_until_ready(result)
    # The final momentum is dropped with the tuple, never deleted: it may alias x.
    x_core, _momentum, loss, _data_loss, _regulariser_value, effective_iters = result
    del x, run, result, _momentum
    if core_cfg.private_storage:
        # Restored once the compiled solve has finished and released its work arrays.
        x_core = transpose_volume(jax.block_until_ready(x_core))
    return x_core, {
        "loss": loss,
        "iteration_loss_computed": bool(quality_policy.compute_iteration_loss),
        "effective_iterations": int(effective_iters),
        "early_stop": False,
        "regulariser": "none",
        "data_loss_computed": bool(quality_policy.compute_final_data_loss),
        "regulariser_value_computed": bool(quality_policy.compute_final_regulariser_value),
        "quality_policy": quality_policy.to_dict(),
        "lipschitz": float(L_core),
        **backend,
    }


def _run_reconstruction_step(
    *,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    pose_params: jnp.ndarray,
    x: jnp.ndarray,
    cfg: object,
    L_prev: float | None,
    outer_idx: int,
    reconstruction: str,
) -> tuple[jnp.ndarray, float | None, OuterStat]:
    recon_geometry = PoseAdjustedGeometry(
        geometry=geometry,
        pose_params=pose_params,
        translation_frame=getattr(cfg, "pose_translation_frame", "object"),
    )
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
    if x_finite_fraction < 1.0 or int(getattr(cfg, "iterations", 0)) <= 0:  # x is kept
        stat, L_next = _reconstruction_step_stat(
            recon_start=recon_start,
            recon_retry=False,
            info_rec={},
            reconstruction=reconstruction,
            cfg=cfg,
            outer_idx=outer_idx,
            L_prev=L_prev,
        )
        if x_finite_fraction < 1.0:
            _mark_nonfinite_reconstruction(
                stat, finite_fraction=x_finite_fraction, reason="nonfinite_initial_reconstruction"
            )
        else:
            stat.update(
                fixed_volume_reconstruction_skipped=True, reconstruction_finite_fraction=1.0
            )
        return x, L_next, stat

    if reconstruction == "fista":
        reference = getattr(cfg, "quality", "fast") == "reference" and cfg.backend == "jax"
        if is_cone_beam(step.recon_geometry) or reference:
            # The reference JAX policy retains the public solver's power-method
            # bound. Cone beams also use it to model diverging rays.
            x_out, info_rec = _run_public_fista_reconstruction(
                step,
                views_per_batch=vpb0,
                gather_dtype=cfg.gather_dtype,
                grad_mode="auto",
            )
        else:
            x_out, info_rec = _run_fista_core_reconstruction(step)
            jax.block_until_ready(x_out)
            core_finite_fraction = _finite_fraction(x_out)
            if core_finite_fraction < 1.0:
                logging.warning(
                    "FISTA core returned non-finite reconstruction "
                    "(finite_fraction=%.6g); retrying public streamed FISTA",
                    core_finite_fraction,
                )
                recon_retry = True
                core_info = dict(info_rec)
                x_out, info_rec = _run_public_fista_reconstruction(
                    step,
                    views_per_batch=1,
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
        x_out, info_rec = _run_spdhg_reconstruction(step)

    jax.block_until_ready(x_out)
    stat, L_next = _reconstruction_step_stat(
        recon_start=recon_start,
        recon_retry=recon_retry,
        info_rec=info_rec,
        reconstruction=reconstruction,
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
