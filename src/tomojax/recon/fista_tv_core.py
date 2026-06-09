"""Array-level differentiable FISTA reconstruction core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.backend_policy import normalize_projector_backend
from tomojax.core.pallas_resolver import resolve_pallas_callable, resolve_pallas_module
from tomojax.core.projector import (
    forward_project_view_T,
    get_detector_grid_device,
    sum_backproject_views_T,
)
from tomojax.recon._tv_ops import huber_tv_grad, huber_tv_value, isotropic_tv_value

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomojax.core.geometry import Detector, Grid
    from tomojax.recon.types import Regulariser


@dataclass(frozen=True, slots=True)
class FistaCoreConfig:
    """Configuration for the array-level FISTA core."""

    iters: int = 10
    lambda_tv: float = 0.005
    regulariser: Regulariser = "huber_tv"
    huber_delta: float = 1e-2
    L: float = 100.0
    positivity: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None
    checkpoint_projector: bool = True
    projector_unroll: int = 1
    gather_dtype: str = "fp32"
    views_per_batch: int = 1
    support: jnp.ndarray | None = None
    forward_projector: str = "jax"
    backprojector: str = "jax"
    pallas_tile_shape: tuple[int, int] = (16, 4)
    pallas_num_warps: int = 1
    pallas_step_size_scale: float = 1.0
    pallas_unroll: int | None = 1
    compute_iteration_loss: bool = True
    compute_final_data_loss: bool = True
    compute_final_regulariser_value: bool = True

    def __post_init__(self) -> None:
        normalize_projector_backend(self.forward_projector)
        normalize_projector_backend(self.backprojector)


@dataclass(frozen=True, slots=True)
class FistaCoreResult:
    """Array-level FISTA result and diagnostics."""

    x: jnp.ndarray
    loss: jnp.ndarray
    data_loss: jnp.ndarray
    regulariser_value: jnp.ndarray
    effective_iters: jnp.ndarray
    status: str

    def info(self) -> dict[str, object]:
        """Return JAX-friendly result metadata."""
        return {
            "loss": self.loss,
            "data_loss": self.data_loss,
            "regulariser_value": self.regulariser_value,
            "effective_iters": self.effective_iters,
            "status": self.status,
        }

    def python_info(self) -> dict[str, object]:
        """Return Python scalar/list metadata for JSON reports."""
        return {
            "loss": [float(v) for v in list(self.loss)],
            "data_loss": float(self.data_loss),
            "regulariser_value": float(self.regulariser_value),
            "effective_iters": int(self.effective_iters),
            "status": self.status,
        }


def fista_tv_core_arrays(
    *,
    x0: jnp.ndarray,
    T_all: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    projections: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    cfg: FistaCoreConfig,
    L_override: jnp.ndarray | float | None = None,
    view_weights: jnp.ndarray | None = None,
) -> FistaCoreResult:
    """Array-level unrolled FISTA/HUBER-TV core.

    This is intentionally stricter than the public `fista_tv` API: arrays in,
    arrays out, no Python `Geometry` object construction, and a static iteration
    count suitable for differentiating tiny bilevel reference problems.
    """
    x_init = _project_constraints(jnp.asarray(x0, dtype=jnp.float32), cfg)
    z_init = x_init
    t_init = jnp.float32(1.0)
    L_raw = cfg.L if L_override is None else L_override
    L = jnp.maximum(jnp.asarray(L_raw, dtype=jnp.float32), jnp.float32(1e-6))
    lam = jnp.asarray(cfg.lambda_tv, dtype=jnp.float32)
    weights = _sqrt_view_weights(projections, view_weights)
    use_fast_fixed_geometry = _use_fused_pallas_for_fixed_geometry_fista(
        cfg,
        T_all=T_all,
        grid=grid,
        detector=detector,
        volume=x_init,
        det_grid=det_grid,
    )
    data_forward_projector, data_backprojector = _resolve_fista_core_data_projectors(
        cfg,
        T_all=T_all,
        grid=grid,
        detector=detector,
        volume=x_init,
        det_grid=det_grid,
    )

    def data_loss_fn(x: jnp.ndarray) -> jnp.ndarray:
        masked = _apply_support(x, cfg.support)
        if use_fast_fixed_geometry:
            loss, _grad = _projection_loss_and_explicit_grad(
                T_all=T_all,
                grid=grid,
                detector=detector,
                volume=masked,
                det_grid=det_grid,
                projections=projections,
                weights=weights,
                checkpoint_projector=cfg.checkpoint_projector,
                projector_unroll=cfg.projector_unroll,
                gather_dtype=cfg.gather_dtype,
                views_per_batch=cfg.views_per_batch,
                forward_projector=data_forward_projector,
                backprojector=data_backprojector,
                pallas_tile_shape=cfg.pallas_tile_shape,
                pallas_num_warps=cfg.pallas_num_warps,
                pallas_step_size_scale=cfg.pallas_step_size_scale,
                pallas_unroll=cfg.pallas_unroll,
                compute_loss=True,
            )
            return loss
        return _projection_loss(
            T_all=T_all,
            grid=grid,
            detector=detector,
            volume=masked,
            det_grid=det_grid,
            projections=projections,
            weights=weights,
            checkpoint_projector=cfg.checkpoint_projector,
            projector_unroll=cfg.projector_unroll,
            gather_dtype=cfg.gather_dtype,
            views_per_batch=cfg.views_per_batch,
            forward_projector=data_forward_projector,
            pallas_tile_shape=cfg.pallas_tile_shape,
            pallas_num_warps=cfg.pallas_num_warps,
            pallas_step_size_scale=cfg.pallas_step_size_scale,
            pallas_unroll=cfg.pallas_unroll,
        )

    def data_loss_and_grad_fn(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        masked = _apply_support(x, cfg.support)
        loss, grad = _projection_loss_and_explicit_grad(
            T_all=T_all,
            grid=grid,
            detector=detector,
            volume=masked,
            det_grid=det_grid,
            projections=projections,
            weights=weights,
            checkpoint_projector=cfg.checkpoint_projector,
            projector_unroll=cfg.projector_unroll,
            gather_dtype=cfg.gather_dtype,
            views_per_batch=cfg.views_per_batch,
            forward_projector=data_forward_projector,
            backprojector=data_backprojector,
            pallas_tile_shape=cfg.pallas_tile_shape,
            pallas_num_warps=cfg.pallas_num_warps,
            pallas_step_size_scale=cfg.pallas_step_size_scale,
            pallas_unroll=cfg.pallas_unroll,
            compute_loss=bool(cfg.compute_iteration_loss or cfg.compute_final_data_loss),
        )
        if cfg.support is not None:
            grad = grad * jnp.asarray(cfg.support, dtype=grad.dtype)
        return loss, grad

    def regulariser_value(x: jnp.ndarray) -> jnp.ndarray:
        if cfg.regulariser == "huber_tv":
            return huber_tv_value(x, float(cfg.huber_delta))
        return isotropic_tv_value(x)

    def body(
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        k: jnp.ndarray,
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], None]:
        x_prev, z_prev, t_prev, loss_arr, _last_data_loss = carry
        data_loss, grad = data_loss_and_grad_fn(z_prev)
        if cfg.regulariser == "huber_tv" and float(cfg.lambda_tv) != 0.0:
            grad = grad + lam * huber_tv_grad(z_prev, float(cfg.huber_delta))
        step = z_prev - grad / L
        # The differentiable reference path intentionally uses gradient FISTA for
        # smoothed regularisers. Exact TV belongs to the public reconstruction
        # adapter or a future nonsmooth implicit path, not the bilevel hot path.
        x_next = _project_constraints(step, cfg)
        t_next = jnp.float32(0.5) * (
            jnp.float32(1.0) + jnp.sqrt(jnp.float32(1.0) + 4.0 * t_prev * t_prev)
        )
        z_next = x_next + ((t_prev - jnp.float32(1.0)) / t_next) * (x_next - x_prev)
        z_next = _project_constraints(z_next, cfg)
        if cfg.compute_iteration_loss:
            loss_arr = loss_arr.at[k].set(
                (data_loss + lam * regulariser_value(z_prev)).astype(jnp.float32)
            )
        return (x_next, z_next, t_next, loss_arr, data_loss.astype(jnp.float32)), None

    n_iters = int(cfg.iters)
    loss0 = jnp.zeros((n_iters,), dtype=jnp.float32)
    (x_final, _, _, loss, last_data_loss), _ = jax.lax.scan(
        body,
        (x_init, z_init, t_init, loss0, jnp.asarray(0.0, dtype=jnp.float32)),
        jnp.arange(n_iters, dtype=jnp.int32),
    )
    data_final = (
        data_loss_fn(x_final)
        if cfg.compute_final_data_loss
        else (loss[-1] if cfg.compute_iteration_loss else last_data_loss)
    )
    reg_final = (
        regulariser_value(x_final)
        if cfg.compute_final_regulariser_value
        else jnp.zeros((), dtype=jnp.float32)
    )
    return FistaCoreResult(
        x=x_final,
        loss=loss,
        data_loss=data_final,
        regulariser_value=reg_final,
        effective_iters=jnp.asarray(n_iters, dtype=jnp.int32),
        status="ok",
    )


def projection_loss_arrays(
    *,
    T_all: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    projections: jnp.ndarray,
    cfg: FistaCoreConfig,
    view_weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Return the weighted projection data term for differentiable recon callers."""
    masked = _apply_support(jnp.asarray(volume, dtype=jnp.float32), cfg.support)
    return _projection_loss(
        T_all=T_all,
        grid=grid,
        detector=detector,
        volume=masked,
        det_grid=det_grid,
        projections=projections,
        weights=_sqrt_view_weights(projections, view_weights),
        checkpoint_projector=bool(cfg.checkpoint_projector),
        projector_unroll=int(cfg.projector_unroll),
        gather_dtype=str(cfg.gather_dtype),
        views_per_batch=int(cfg.views_per_batch),
        forward_projector=str(cfg.forward_projector),
        pallas_tile_shape=cfg.pallas_tile_shape,
        pallas_num_warps=int(cfg.pallas_num_warps),
        pallas_step_size_scale=float(cfg.pallas_step_size_scale),
        pallas_unroll=cfg.pallas_unroll,
    )


def regulariser_value_arrays(volume: jnp.ndarray, cfg: FistaCoreConfig) -> jnp.ndarray:
    """Return the reconstruction regulariser value selected by ``cfg``."""
    if cfg.regulariser == "huber_tv":
        return huber_tv_value(volume, float(cfg.huber_delta))
    return isotropic_tv_value(volume)


def fista_objective_arrays(
    *,
    T_all: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    projections: jnp.ndarray,
    cfg: FistaCoreConfig,
    view_weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Return the differentiable FISTA objective value for a candidate volume."""
    data = projection_loss_arrays(
        T_all=T_all,
        grid=grid,
        detector=detector,
        volume=volume,
        det_grid=det_grid,
        projections=projections,
        cfg=cfg,
        view_weights=view_weights,
    )
    if cfg.lambda_tv == 0.0:
        return data
    return data + jnp.asarray(cfg.lambda_tv, dtype=jnp.float32) * regulariser_value_arrays(
        volume,
        cfg,
    )


def _project_stack(
    *,
    T_all: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    checkpoint_projector: bool,
    projector_unroll: int,
    gather_dtype: str,
    views_per_batch: int = 1,
    forward_projector: str = "jax",
    pallas_tile_shape: tuple[int, int] = (16, 4),
    pallas_num_warps: int = 1,
    pallas_step_size_scale: float = 1.0,
    pallas_unroll: int | None = 1,
) -> jnp.ndarray:
    n_views = int(T_all.shape[0])
    if n_views == 0:
        return jnp.zeros((0, detector.nv, detector.nu), dtype=jnp.float32)
    b = _chunk_size(n_views, views_per_batch)
    num_chunks = (n_views + b - 1) // b

    def body(
        out: jnp.ndarray,
        i: jnp.ndarray,
    ) -> tuple[jnp.ndarray, None]:
        start_shifted, _valid_mask, view_idx = _chunk_schedule(i, n_views=n_views, chunk_size=b)
        T_chunk = jax.lax.dynamic_slice(T_all, (start_shifted, 0, 0), (b, 4, 4))
        pred = _project_chunk(
            T_chunk=T_chunk,
            grid=grid,
            detector=detector,
            volume=volume,
            det_grid=det_grid,
            checkpoint_projector=checkpoint_projector,
            projector_unroll=projector_unroll,
            gather_dtype=gather_dtype,
            forward_projector=forward_projector,
            pallas_tile_shape=pallas_tile_shape,
            pallas_num_warps=pallas_num_warps,
            pallas_step_size_scale=pallas_step_size_scale,
            pallas_unroll=pallas_unroll,
        )
        return out.at[view_idx].set(pred), None

    init = jnp.zeros((n_views, detector.nv, detector.nu), dtype=jnp.float32)
    out, _ = jax.lax.scan(body, init, jnp.arange(num_chunks, dtype=jnp.int32))
    return out


def _project_chunk(
    *,
    T_chunk: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    checkpoint_projector: bool,
    projector_unroll: int,
    gather_dtype: str,
    forward_projector: str,
    pallas_tile_shape: tuple[int, int],
    pallas_num_warps: int,
    pallas_step_size_scale: float,
    pallas_unroll: int | None,
) -> jnp.ndarray:
    if forward_projector == "jax":
        return jax.vmap(
            lambda T: forward_project_view_T(
                T,
                grid,
                detector,
                volume,
                use_checkpoint=checkpoint_projector,
                unroll=int(projector_unroll),
                gather_dtype=gather_dtype,
                det_grid=det_grid,
            )
        )(T_chunk)
    if forward_projector == "pallas":
        pallas_project, fallback_reason = resolve_pallas_callable(
            "forward_project_views_T_pallas",
            missing_reason="pallas_batched_callable_missing",
        )
        if pallas_project is None:
            raise RuntimeError(fallback_reason or "pallas_projector_unavailable")
        return pallas_project(
            T_chunk,
            grid,
            detector,
            volume,
            options=_pallas_projector_options(
                step_size=float(grid.vy) * float(pallas_step_size_scale),
                unroll=pallas_unroll,
                gather_dtype=gather_dtype,
                det_grid=det_grid,
                tile_shape=tuple(pallas_tile_shape),
                num_warps=int(pallas_num_warps),
                kernel_variant="auto",
                layout_variant="detector_vu",
                state_mode="cached",
            ),
        )
    raise ValueError("FistaCoreConfig.forward_projector must be one of 'jax' or 'pallas'")


def _projection_loss(
    *,
    T_all: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    projections: jnp.ndarray,
    weights: jnp.ndarray,
    checkpoint_projector: bool,
    projector_unroll: int,
    gather_dtype: str,
    views_per_batch: int,
    forward_projector: str,
    pallas_tile_shape: tuple[int, int],
    pallas_num_warps: int,
    pallas_step_size_scale: float,
    pallas_unroll: int | None,
) -> jnp.ndarray:
    n_views = int(T_all.shape[0])
    if n_views == 0:
        return jnp.asarray(0.0, dtype=jnp.float32)
    b = _chunk_size(n_views, views_per_batch)
    num_chunks = (n_views + b - 1) // b
    nv = int(projections.shape[1])
    nu = int(projections.shape[2])

    def body(
        loss_acc: jnp.ndarray,
        i: jnp.ndarray,
    ) -> tuple[jnp.ndarray, None]:
        start_shifted, valid_mask, _view_idx = _chunk_schedule(i, n_views=n_views, chunk_size=b)
        T_chunk = jax.lax.dynamic_slice(T_all, (start_shifted, 0, 0), (b, 4, 4))
        y_chunk = jax.lax.dynamic_slice(projections, (start_shifted, 0, 0), (b, nv, nu))
        w_chunk = jax.lax.dynamic_slice(weights, (start_shifted, 0, 0), (b, 1, 1))
        pred = _project_chunk(
            T_chunk=T_chunk,
            grid=grid,
            detector=detector,
            volume=volume,
            det_grid=det_grid,
            checkpoint_projector=checkpoint_projector,
            projector_unroll=projector_unroll,
            gather_dtype=gather_dtype,
            forward_projector=forward_projector,
            pallas_tile_shape=pallas_tile_shape,
            pallas_num_warps=pallas_num_warps,
            pallas_step_size_scale=pallas_step_size_scale,
            pallas_unroll=pallas_unroll,
        )
        resid = (pred - y_chunk).astype(jnp.float32) * w_chunk
        resid = resid * valid_mask[:, None, None]
        return loss_acc + jnp.float32(0.5) * jnp.vdot(resid, resid).real, None

    loss, _ = jax.lax.scan(
        body,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.arange(num_chunks, dtype=jnp.int32),
    )
    return loss


def _sqrt_view_weights(
    projections: jnp.ndarray,
    view_weights: jnp.ndarray | None,
) -> jnp.ndarray:
    n_views = int(projections.shape[0])
    weights = (
        jnp.ones((n_views,), dtype=jnp.float32)
        if view_weights is None
        else jnp.asarray(view_weights, dtype=jnp.float32).reshape((n_views,))
    )
    return jnp.sqrt(jnp.maximum(weights, jnp.float32(0.0)))[:, None, None]


def _projection_loss_and_explicit_grad(
    *,
    T_all: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    projections: jnp.ndarray,
    weights: jnp.ndarray,
    checkpoint_projector: bool,
    projector_unroll: int,
    gather_dtype: str,
    views_per_batch: int,
    forward_projector: str,
    backprojector: str,
    pallas_tile_shape: tuple[int, int],
    pallas_num_warps: int,
    pallas_step_size_scale: float = 1.0,
    pallas_unroll: int | None = 1,
    compute_loss: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    n_views = int(T_all.shape[0])
    if n_views == 0:
        return jnp.asarray(0.0, dtype=jnp.float32), jnp.zeros_like(volume)
    if forward_projector == "pallas" and backprojector == "pallas":
        if _fixed_geometry_is_available(T_all):
            loss_grad_fn, fallback_reason = resolve_pallas_callable(
                "forward_project_loss_and_grad_T_pallas",
                missing_reason="pallas_loss_grad_callable_missing",
            )
            if loss_grad_fn is None:
                raise RuntimeError(fallback_reason or "pallas_projector_unavailable")
            tuned_tile_shape, tuned_num_warps = _pallas_loss_grad_full_stack_tiling(
                detector,
                requested_tile_shape=pallas_tile_shape,
                requested_num_warps=pallas_num_warps,
            )
            return loss_grad_fn(
                T_all,
                grid,
                detector,
                volume,
                projections,
                weights=weights,
                compute_loss=bool(compute_loss),
                options=_pallas_projector_options(
                    step_size=float(grid.vy) * float(pallas_step_size_scale),
                    unroll=pallas_unroll,
                    gather_dtype=gather_dtype,
                    det_grid=det_grid,
                    tile_shape=tuned_tile_shape,
                    num_warps=tuned_num_warps,
                    kernel_variant="auto",
                    layout_variant="detector_vu",
                    state_mode="cached",
                ),
            )
        forward_projector = "jax"
        backprojector = "jax"
    b = _chunk_size(n_views, views_per_batch)
    num_chunks = (n_views + b - 1) // b
    nv = int(projections.shape[1])
    nu = int(projections.shape[2])
    backproject_fn = _resolve_sum_backproject_views(backprojector)

    def body(
        carry: tuple[jnp.ndarray, jnp.ndarray],
        i: jnp.ndarray,
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], None]:
        loss_acc, grad_acc = carry
        start_shifted, valid_mask, _view_idx = _chunk_schedule(i, n_views=n_views, chunk_size=b)
        T_chunk = jax.lax.dynamic_slice(T_all, (start_shifted, 0, 0), (b, 4, 4))
        y_chunk = jax.lax.dynamic_slice(projections, (start_shifted, 0, 0), (b, nv, nu))
        w_chunk = jax.lax.dynamic_slice(weights, (start_shifted, 0, 0), (b, 1, 1))
        valid = valid_mask[:, None, None]
        if forward_projector == "pallas" and backprojector == "pallas":
            loss_grad_fn, fallback_reason = resolve_pallas_callable(
                "forward_project_loss_and_grad_T_pallas",
                missing_reason="pallas_loss_grad_callable_missing",
            )
            if loss_grad_fn is None:
                raise RuntimeError(fallback_reason or "pallas_projector_unavailable")
            loss_batch, grad_batch = loss_grad_fn(
                T_chunk,
                grid,
                detector,
                volume,
                y_chunk,
                weights=w_chunk * valid,
                compute_loss=bool(compute_loss),
                options=_pallas_projector_options(
                    step_size=float(grid.vy) * float(pallas_step_size_scale),
                    unroll=pallas_unroll,
                    gather_dtype=gather_dtype,
                    det_grid=det_grid,
                    tile_shape=tuple(pallas_tile_shape),
                    num_warps=int(pallas_num_warps),
                    kernel_variant="auto",
                    layout_variant="detector_vu",
                    state_mode="cached",
                ),
            )
        else:
            pred = _project_chunk(
                T_chunk=T_chunk,
                grid=grid,
                detector=detector,
                volume=volume,
                det_grid=det_grid,
                checkpoint_projector=checkpoint_projector,
                projector_unroll=projector_unroll,
                gather_dtype=gather_dtype,
                forward_projector=forward_projector,
                pallas_tile_shape=pallas_tile_shape,
                pallas_num_warps=pallas_num_warps,
                pallas_step_size_scale=pallas_step_size_scale,
                pallas_unroll=pallas_unroll,
            )
            raw_resid = (pred - y_chunk).astype(jnp.float32)
            weighted_resid = raw_resid * w_chunk * valid
            loss_batch = (
                jnp.float32(0.5) * jnp.vdot(weighted_resid, weighted_resid).real
                if compute_loss
                else jnp.asarray(0.0, dtype=jnp.float32)
            )
            grad_resid = raw_resid * (w_chunk * w_chunk) * valid
            grad_batch = backproject_fn(
                T_chunk,
                grid,
                detector,
                grad_resid,
                unroll=1 if backprojector == "pallas" else int(projector_unroll),
                gather_dtype=gather_dtype,
                det_grid=det_grid,
            )
        return (loss_acc + loss_batch, grad_acc + grad_batch), None

    init = (jnp.asarray(0.0, dtype=jnp.float32), jnp.zeros_like(volume))
    (loss, grad), _ = jax.lax.scan(
        body,
        init,
        jnp.arange(num_chunks, dtype=jnp.int32),
    )
    return loss, grad


def _resolve_sum_backproject_views(backprojector: str) -> Callable[..., jnp.ndarray]:
    if backprojector == "jax":
        return sum_backproject_views_T
    if backprojector == "pallas":
        backproject_fn, fallback_reason = resolve_pallas_callable(
            "sum_backproject_views_T_pallas",
            missing_reason="pallas_sum_backproject_callable_missing",
        )
        if backproject_fn is None:
            raise RuntimeError(fallback_reason or "pallas_projector_unavailable")
        return backproject_fn
    raise ValueError("FistaCoreConfig.backprojector must be one of 'jax' or 'pallas'")


def _pallas_projector_options(**kwargs: object) -> object:
    capability = resolve_pallas_module()
    if capability.module is None:
        raise RuntimeError(capability.unavailable_reason or "pallas_projector_unavailable")
    options_cls = getattr(capability.module, "PallasProjectorOptions", None)
    if options_cls is None or not callable(options_cls):
        raise RuntimeError("pallas_projector_options_missing")
    return options_cls(**kwargs)


def _pallas_loss_grad_full_stack_tiling(
    detector: Detector,
    *,
    requested_tile_shape: tuple[int, int],
    requested_num_warps: int,
) -> tuple[tuple[int, int], int]:
    if tuple(requested_tile_shape) not in {(8, 8), (16, 4)} or int(requested_num_warps) != 1:
        return tuple(requested_tile_shape), int(requested_num_warps)
    det_shape = (int(detector.nv), int(detector.nu))
    if det_shape == (64, 64):
        return (64, 1), 2
    if det_shape == (96, 96):
        return (32, 2), 2
    if det_shape == (80, 64):
        return (16, 4), 2
    return tuple(requested_tile_shape), int(requested_num_warps)


def _use_fused_pallas_for_fixed_geometry_fista(
    cfg: FistaCoreConfig,
    *,
    T_all: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> bool:
    if cfg.support is not None or not _det_grid_allows_fused_pallas(detector, det_grid):
        return False
    if jax.default_backend() != "gpu":
        return False
    if not _fixed_geometry_is_available(T_all):
        return False
    if not (
        int(T_all.shape[0]) >= 16
        and min(int(detector.nv), int(detector.nu)) >= 64
        and min(int(grid.nx), int(grid.ny), int(grid.nz)) >= 64
    ):
        return False
    return _fused_pallas_loss_grad_is_available(
        cfg,
        T_all=T_all,
        grid=grid,
        detector=detector,
        volume=volume,
        det_grid=det_grid,
    )


def _fused_pallas_loss_grad_is_available(
    cfg: FistaCoreConfig,
    *,
    T_all: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> bool:
    loss_grad_fn, _fallback_reason = resolve_pallas_callable(
        "forward_project_loss_and_grad_T_pallas",
        missing_reason="pallas_loss_grad_callable_missing",
    )
    if loss_grad_fn is None:
        return False
    support_fn, _support_missing_reason = resolve_pallas_callable(
        "pallas_projector_sinogram_unsupported_reason",
        missing_reason="pallas_sinogram_support_check_missing",
    )
    if support_fn is None:
        return False
    tuned_tile_shape, tuned_num_warps = _pallas_loss_grad_full_stack_tiling(
        detector,
        requested_tile_shape=cfg.pallas_tile_shape,
        requested_num_warps=cfg.pallas_num_warps,
    )
    try:
        options = _pallas_projector_options(
            step_size=float(grid.vy) * float(cfg.pallas_step_size_scale),
            unroll=cfg.pallas_unroll,
            gather_dtype=cfg.gather_dtype,
            det_grid=det_grid,
            tile_shape=tuned_tile_shape,
            num_warps=tuned_num_warps,
            kernel_variant="auto",
            layout_variant="detector_vu",
            state_mode="cached",
        )
    except Exception:
        return False
    try:
        reason = support_fn(
            T_all,
            grid,
            detector,
            volume,
            options=options,
        )
    except Exception:
        return False
    return not reason


def _det_grid_allows_fused_pallas(
    detector: Detector,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> bool:
    if det_grid is None:
        return True
    try:
        expected_u, expected_v = get_detector_grid_device(detector)
        actual_u = np.asarray(det_grid[0], dtype=np.float32)
        actual_v = np.asarray(det_grid[1], dtype=np.float32)
        expected_u_host = np.asarray(expected_u, dtype=np.float32)
        expected_v_host = np.asarray(expected_v, dtype=np.float32)
    except Exception:
        return False
    return bool(
        np.array_equal(actual_u, expected_u_host) and np.array_equal(actual_v, expected_v_host)
    )


def effective_fista_core_backend(
    cfg: FistaCoreConfig,
    *,
    T_all: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> str:
    """Return the effective data projector backend selected by the core."""
    forward_projector, backprojector = _resolve_fista_core_data_projectors(
        cfg,
        T_all=T_all,
        grid=grid,
        detector=detector,
        volume=volume,
        det_grid=det_grid,
    )
    return "pallas" if forward_projector == "pallas" and backprojector == "pallas" else "jax"


def _resolve_fista_core_data_projectors(
    cfg: FistaCoreConfig,
    *,
    T_all: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> tuple[str, str]:
    fixed_geometry_available = _fixed_geometry_is_available(T_all)
    # TomoJAX treats concrete fixed-geometry GPU reconstruction as the fast path:
    # use the fused Pallas projector even when legacy config fields still say
    # "jax". Traced geometry keeps the differentiable alignment path below.
    if _use_fused_pallas_for_fixed_geometry_fista(
        cfg,
        T_all=T_all,
        grid=grid,
        detector=detector,
        volume=volume,
        det_grid=det_grid,
    ):
        return "pallas", "pallas"
    if (
        cfg.forward_projector == "pallas"
        and cfg.backprojector == "pallas"
        and not fixed_geometry_available
    ):
        return "jax", "jax"
    return cfg.forward_projector, cfg.backprojector


def _fixed_geometry_is_available(T_all: jnp.ndarray) -> bool:
    try:
        arr = np.asarray(T_all, dtype=np.float32)
    except Exception:
        return False
    return arr.ndim == 3 and arr.shape[1:] == (4, 4) and bool(np.isfinite(arr).all())


def _chunk_size(n_views: int, views_per_batch: int | None) -> int:
    b = int(views_per_batch) if views_per_batch is not None and int(views_per_batch) > 0 else 1
    return max(1, min(int(b), int(n_views)))


def _chunk_schedule(
    i: jnp.ndarray,
    *,
    n_views: int,
    chunk_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    i = jnp.asarray(i, dtype=jnp.int32)
    b = jnp.int32(chunk_size)
    start = i * b
    remaining = jnp.maximum(0, jnp.int32(n_views) - start)
    valid = jnp.minimum(b, remaining)
    shift = b - valid
    start_shifted = jnp.maximum(0, start - shift)
    idx = jnp.arange(chunk_size, dtype=jnp.int32)
    valid_mask = (idx >= (b - valid)).astype(jnp.float32)
    return start_shifted, valid_mask, start_shifted + idx


def _apply_support(x: jnp.ndarray, support: jnp.ndarray | None) -> jnp.ndarray:
    if support is None:
        return x
    return x * jnp.asarray(support, dtype=x.dtype)


def _project_constraints(x: jnp.ndarray, cfg: FistaCoreConfig) -> jnp.ndarray:
    out = x
    lower = cfg.lower_bound
    if cfg.positivity:
        lower = 0.0 if lower is None else max(0.0, float(lower))
    if lower is not None:
        out = jnp.maximum(out, jnp.asarray(lower, dtype=out.dtype))
    if cfg.upper_bound is not None:
        out = jnp.minimum(out, jnp.asarray(cfg.upper_bound, dtype=out.dtype))
    return out
