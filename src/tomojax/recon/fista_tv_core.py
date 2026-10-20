"""Array-level differentiable FISTA reconstruction core."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.backends.host_stream import read_views
from tomojax.core.backend_policy import normalize_projector_backend
from tomojax.core.cuda_projector_resolver import resolve_cuda_callable, resolve_cuda_module
from tomojax.core.projector import (
    RAY_INTEGRATORS,
    forward_project_view_T,
    sum_backproject_views_T,
)
from tomojax.core.view_chunks import chunk_schedule, chunk_size
from tomojax.recon._tv_ops import huber_tv_grad, huber_tv_value, isotropic_tv_value

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomojax.core.geometry import Detector, Grid
    from tomojax.recon.types import Regulariser


@dataclass(frozen=True, slots=True)
class FistaCoreConfig:
    """Configuration for the array-level FISTA core.

    A CUDA sampling step is preserved if dynamic poses or detector coordinates
    require a JAX fallback.
    """

    iterations: int = 10
    tv_weight: float = 0.005
    regulariser: Regulariser = "huber_tv"
    huber_delta: float = 1e-2
    lipschitz: float = 100.0
    nonnegative: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None
    checkpoint_projector: bool = True
    projector_unroll: int = 1
    gather_dtype: str = "fp32"
    views_per_batch: int = 1
    support: jnp.ndarray | None = None
    forward_projector: str = "jax"
    backprojector: str = "jax"
    cuda_tile_shape: tuple[int, int] = (16, 4)
    cuda_num_warps: int = 1
    cuda_step_size_scale: float = 1.0
    cuda_unroll: int | None = 1
    ray_integrator: str = "sampled"
    compute_iteration_loss: bool = True
    compute_final_data_loss: bool = True
    compute_final_regulariser_value: bool = True
    # Projections are a live backends.host_stream source key, not a device stack.
    stream_projections: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.lipschitz) or self.lipschitz <= 0:
            raise ValueError("FistaCoreConfig.lipschitz must be finite and positive")
        normalize_projector_backend(self.forward_projector)
        normalize_projector_backend(self.backprojector)
        if self.ray_integrator not in RAY_INTEGRATORS:
            raise ValueError("ray_integrator must be 'sampled' or 'exact'")


@dataclass(frozen=True, slots=True)
class FistaCoreResult[Array: jnp.ndarray | np.ndarray]:
    """Array-level FISTA result and diagnostics."""

    x: Array
    loss: jnp.ndarray
    data_loss: jnp.ndarray
    regulariser_value: jnp.ndarray
    effective_iterations: jnp.ndarray
    status: str

    def info(self) -> dict[str, object]:
        """Return JAX-friendly result metadata."""
        return {
            "loss": self.loss,
            "data_loss": self.data_loss,
            "regulariser_value": self.regulariser_value,
            "effective_iterations": self.effective_iterations,
            "status": self.status,
        }

    def python_info(self) -> dict[str, object]:
        """Return Python scalar/list metadata for JSON reports."""
        return {
            "loss": [float(v) for v in list(self.loss)],
            "data_loss": float(self.data_loss),
            "regulariser_value": float(self.regulariser_value),
            "effective_iterations": int(self.effective_iterations),
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
) -> FistaCoreResult[jnp.ndarray]:
    """Array-level unrolled FISTA/HUBER-TV core.

    This is intentionally stricter than the public `fista_tv` API: arrays in,
    arrays out, no Python `Geometry` object construction, and a static iteration
    count suitable for differentiating tiny bilevel reference problems.
    ``L_override`` must be positive and bound the full objective gradient.
    Nonzero regularization requires Huber-TV; nonsmooth TV belongs to ``fista_tv``.
    With ``cfg.stream_projections``, ``projections`` is a registered host-source
    key. Its lifetime must include completion of all returned device arrays;
    the reconstruction remains differentiable in the volume, not the host data.
    If final data diagnostics are disabled, ``data_loss`` is the last evaluated
    data term, or zero when none was computed; it never includes regularization.
    """
    if cfg.tv_weight != 0.0 and cfg.regulariser != "huber_tv":
        raise ValueError("fista_tv_core_arrays requires huber_tv or tv_weight=0")
    x_init = _project_constraints(jnp.asarray(x0, dtype=jnp.float32), cfg)
    L = jnp.asarray(cfg.lipschitz if L_override is None else L_override, dtype=jnp.float32)
    lam = jnp.asarray(cfg.tv_weight, dtype=jnp.float32)
    weights = _view_weights(int(T_all.shape[0]), view_weights)
    data_forward_projector, data_backprojector = _resolve_fista_core_data_projectors(
        cfg,
        T_all=T_all,
        det_grid=det_grid,
    )

    sampled_step = _requested_sampled_step(grid, cfg)

    def data_loss_fn(x: jnp.ndarray) -> jnp.ndarray:
        return _projection_loss(
            T_all=T_all,
            grid=grid,
            detector=detector,
            volume=_apply_support(x, cfg.support),
            det_grid=det_grid,
            projections=projections,
            stream_projections=cfg.stream_projections,
            weights=weights,
            checkpoint_projector=cfg.checkpoint_projector,
            projector_unroll=cfg.projector_unroll,
            gather_dtype=cfg.gather_dtype,
            views_per_batch=cfg.views_per_batch,
            forward_projector=data_forward_projector,
            cuda_tile_shape=cfg.cuda_tile_shape,
            cuda_num_warps=cfg.cuda_num_warps,
            cuda_step_size_scale=cfg.cuda_step_size_scale,
            cuda_unroll=cfg.cuda_unroll,
            ray_integrator=cfg.ray_integrator,
            sampled_step_size=sampled_step,
        )

    def data_loss_and_grad_fn(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        loss, grad = _projection_loss_and_explicit_grad(
            T_all=T_all,
            grid=grid,
            detector=detector,
            volume=_apply_support(x, cfg.support),
            det_grid=det_grid,
            projections=projections,
            stream_projections=cfg.stream_projections,
            weights=weights,
            checkpoint_projector=cfg.checkpoint_projector,
            projector_unroll=cfg.projector_unroll,
            gather_dtype=cfg.gather_dtype,
            views_per_batch=cfg.views_per_batch,
            forward_projector=data_forward_projector,
            backprojector=data_backprojector,
            cuda_tile_shape=cfg.cuda_tile_shape,
            cuda_num_warps=cfg.cuda_num_warps,
            cuda_step_size_scale=cfg.cuda_step_size_scale,
            cuda_unroll=cfg.cuda_unroll,
            ray_integrator=cfg.ray_integrator,
            sampled_step_size=sampled_step,
            compute_loss=bool(cfg.compute_iteration_loss or cfg.compute_final_data_loss),
        )
        return loss, _apply_support(grad, cfg.support)

    def body(
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        k: jnp.ndarray,
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], None]:
        x_prev, z_prev, t_prev, loss_arr, _last_data_loss = carry
        data_loss, grad = data_loss_and_grad_fn(z_prev)
        if cfg.regulariser == "huber_tv" and float(cfg.tv_weight) != 0.0:
            grad = grad + lam * huber_tv_grad(z_prev, float(cfg.huber_delta))
        # The differentiable reference path intentionally uses gradient FISTA for
        # smoothed regularisers. Exact TV belongs to the public reconstruction
        # adapter or a future nonsmooth implicit path, not the bilevel hot path.
        x_next = _project_constraints(z_prev - grad / L, cfg)
        t_next = jnp.float32(0.5) * (
            jnp.float32(1.0) + jnp.sqrt(jnp.float32(1.0) + 4.0 * t_prev * t_prev)
        )
        z_next = x_next + ((t_prev - jnp.float32(1.0)) / t_next) * (x_next - x_prev)
        z_next = _project_constraints(z_next, cfg)
        if cfg.compute_iteration_loss and cfg.iterations > 0:
            objective = data_loss
            if cfg.tv_weight != 0.0:
                objective = objective + lam * regulariser_value_arrays(z_prev, cfg)
            loss_arr = loss_arr.at[k].set(objective.astype(jnp.float32))
        return (x_next, z_next, t_next, loss_arr, data_loss.astype(jnp.float32)), None

    n_iters = int(cfg.iterations)
    loss0 = jnp.zeros((n_iters,), dtype=jnp.float32)
    (x_final, _, _, loss, last_data_loss), _ = jax.lax.scan(
        body,
        (x_init, x_init, jnp.float32(1.0), loss0, jnp.float32(0.0)),
        jnp.arange(n_iters, dtype=jnp.int32),
    )
    data_final = data_loss_fn(x_final) if cfg.compute_final_data_loss else last_data_loss
    reg_final = (
        regulariser_value_arrays(x_final, cfg)
        if cfg.compute_final_regulariser_value
        else jnp.zeros((), dtype=jnp.float32)
    )
    return FistaCoreResult(
        x=x_final,
        loss=loss,
        data_loss=data_final,
        regulariser_value=reg_final,
        effective_iterations=jnp.asarray(n_iters, dtype=jnp.int32),
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
    forward_projector, _ = _resolve_fista_core_data_projectors(cfg, T_all=T_all, det_grid=det_grid)
    return _projection_loss(
        T_all=T_all,
        grid=grid,
        detector=detector,
        volume=masked,
        det_grid=det_grid,
        projections=projections,
        weights=_view_weights(int(T_all.shape[0]), view_weights),
        stream_projections=cfg.stream_projections,
        checkpoint_projector=bool(cfg.checkpoint_projector),
        projector_unroll=int(cfg.projector_unroll),
        gather_dtype=str(cfg.gather_dtype),
        views_per_batch=int(cfg.views_per_batch),
        forward_projector=forward_projector,
        cuda_tile_shape=cfg.cuda_tile_shape,
        cuda_num_warps=int(cfg.cuda_num_warps),
        cuda_step_size_scale=float(cfg.cuda_step_size_scale),
        cuda_unroll=cfg.cuda_unroll,
        ray_integrator=cfg.ray_integrator,
        sampled_step_size=_requested_sampled_step(grid, cfg),
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
    if cfg.tv_weight == 0.0:
        return data
    return data + jnp.asarray(cfg.tv_weight, dtype=jnp.float32) * regulariser_value_arrays(
        volume,
        cfg,
    )


def _requested_sampled_step(grid: Grid, cfg: FistaCoreConfig) -> float | None:
    # Backend fallback must not change the requested discretization.
    return grid.vy * cfg.cuda_step_size_scale if cfg.forward_projector == "cuda" else None


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
    cuda_tile_shape: tuple[int, int] = (16, 4),
    cuda_num_warps: int = 1,
    cuda_step_size_scale: float = 1.0,
    cuda_unroll: int | None = 1,
    ray_integrator: str = "sampled",
) -> jnp.ndarray:
    n_views = int(T_all.shape[0])
    if n_views == 0:
        return jnp.zeros((0, detector.nv, detector.nu), dtype=jnp.float32)
    b = chunk_size(n_views, views_per_batch, default=1)
    num_chunks = (n_views + b - 1) // b

    def body(
        out: jnp.ndarray,
        i: jnp.ndarray,
    ) -> tuple[jnp.ndarray, None]:
        start_shifted, _valid_mask, view_idx = chunk_schedule(i, n_views=n_views, chunk_size=b)
        T_chunk = jax.lax.dynamic_slice_in_dim(T_all, start_shifted, b)
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
            cuda_tile_shape=cuda_tile_shape,
            cuda_num_warps=cuda_num_warps,
            cuda_step_size_scale=cuda_step_size_scale,
            cuda_unroll=cuda_unroll,
            ray_integrator=ray_integrator,
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
    cuda_tile_shape: tuple[int, int],
    cuda_num_warps: int,
    cuda_step_size_scale: float,
    cuda_unroll: int | None,
    ray_integrator: str = "sampled",
    sampled_step_size: float | None = None,
) -> jnp.ndarray:
    if ray_integrator == "exact":
        from tomojax.core.trilinear import exact_forward

        return exact_forward(
            T_chunk, grid, detector, volume, backend=forward_projector, det_grid=det_grid
        )
    if ray_integrator in {"joseph", "joseph_cubic"}:
        from tomojax.core.joseph import forward_project_planes, plane_coefficients

        return forward_project_planes(
            plane_coefficients(T_chunk, grid, detector, det_grid),
            volume,
            grid,
            detector,
            backend=forward_projector,
            interpolation="cubic" if ray_integrator == "joseph_cubic" else "linear",
        )
    if forward_projector == "jax":
        return jax.vmap(
            lambda T: forward_project_view_T(
                T,
                grid,
                detector,
                volume,
                step_size=sampled_step_size,
                use_checkpoint=checkpoint_projector,
                unroll=int(projector_unroll),
                gather_dtype=gather_dtype,
                det_grid=det_grid,
            )
        )(T_chunk)
    if forward_projector == "cuda":
        cuda_project, fallback_reason = resolve_cuda_callable(
            "forward_project_views_T_cuda",
            missing_reason="cuda_batched_callable_missing",
        )
        if cuda_project is None:
            raise RuntimeError(fallback_reason or "cuda_projector_unavailable")
        return cuda_project(
            T_chunk,
            grid,
            detector,
            volume,
            options=_cuda_projector_options(
                step_size=float(grid.vy) * float(cuda_step_size_scale),
                unroll=cuda_unroll,
                gather_dtype=gather_dtype,
                det_grid=det_grid,
                tile_shape=tuple(cuda_tile_shape),
                num_warps=int(cuda_num_warps),
                kernel_variant="auto",
                layout_variant="detector_vu",
                state_mode="cached",
            ),
        )
    raise ValueError("FistaCoreConfig.forward_projector must be one of 'jax' or 'cuda'")


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
    cuda_tile_shape: tuple[int, int],
    cuda_num_warps: int,
    cuda_step_size_scale: float,
    cuda_unroll: int | None,
    ray_integrator: str = "sampled",
    sampled_step_size: float | None = None,
    stream_projections: bool = False,
) -> jnp.ndarray:
    n_views = int(T_all.shape[0])
    if n_views == 0:
        return jnp.asarray(0.0, dtype=jnp.float32)
    b = chunk_size(n_views, views_per_batch, default=1)
    num_chunks = (n_views + b - 1) // b

    def body(
        loss_acc: jnp.ndarray,
        i: jnp.ndarray,
    ) -> tuple[jnp.ndarray, None]:
        start_shifted, valid_mask, _view_idx = chunk_schedule(i, n_views=n_views, chunk_size=b)
        T_chunk = jax.lax.dynamic_slice_in_dim(T_all, start_shifted, b)
        y_chunk = (
            read_views(projections, start_shifted, (b, detector.nv, detector.nu))
            if stream_projections
            else jax.lax.dynamic_slice_in_dim(projections, start_shifted, b)
        )
        w_chunk = jax.lax.dynamic_slice_in_dim(weights, start_shifted, b)
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
            cuda_tile_shape=cuda_tile_shape,
            cuda_num_warps=cuda_num_warps,
            cuda_step_size_scale=cuda_step_size_scale,
            cuda_unroll=cuda_unroll,
            ray_integrator=ray_integrator,
            sampled_step_size=sampled_step_size,
        )
        resid = (pred - y_chunk).astype(jnp.float32) * valid_mask[:, None, None]
        return loss_acc + jnp.float32(0.5) * jnp.vdot(resid, resid * w_chunk).real, None

    loss, _ = jax.lax.scan(
        body,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.arange(num_chunks, dtype=jnp.int32),
    )
    return loss


def _view_weights(
    n_views: int,
    view_weights: jnp.ndarray | None,
) -> jnp.ndarray:
    weights = (
        jnp.ones((n_views,), dtype=jnp.float32)
        if view_weights is None
        else jnp.asarray(view_weights, dtype=jnp.float32).reshape((n_views,))
    )
    # Keep precision weights, not their square roots: a zero weight can come
    # from a smooth squared parameter, whose derivatives must stay finite.
    # Choose the nonnegative-side derivative at zero for that parameterization.
    return jnp.where(weights < 0, jnp.float32(0.0), weights)[:, None, None]


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
    cuda_tile_shape: tuple[int, int],
    cuda_num_warps: int,
    cuda_step_size_scale: float = 1.0,
    cuda_unroll: int | None = 1,
    ray_integrator: str = "sampled",
    sampled_step_size: float | None = None,
    compute_loss: bool = True,
    stream_projections: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if sampled_step_size is None and forward_projector == "cuda":
        sampled_step_size = float(grid.vy) * float(cuda_step_size_scale)
    n_views = int(T_all.shape[0])
    if n_views == 0:
        return jnp.asarray(0.0, dtype=jnp.float32), jnp.zeros_like(volume)
    if (
        ray_integrator == "sampled"
        and forward_projector == "cuda"
        and backprojector == "cuda"
        and not stream_projections
    ):
        if _fixed_geometry_is_available(T_all, det_grid):
            loss_grad_fn, fallback_reason = resolve_cuda_callable(
                "forward_project_loss_and_grad_T_cuda",
                missing_reason="cuda_loss_grad_callable_missing",
            )
            if loss_grad_fn is None:
                raise RuntimeError(fallback_reason or "cuda_projector_unavailable")
            return loss_grad_fn(
                T_all,
                grid,
                detector,
                volume,
                projections,
                weights=jnp.sqrt(weights),
                compute_loss=bool(compute_loss),
                options=_cuda_projector_options(
                    step_size=float(grid.vy) * float(cuda_step_size_scale),
                    unroll=cuda_unroll,
                    gather_dtype=gather_dtype,
                    det_grid=det_grid,
                    tile_shape=cuda_tile_shape,
                    num_warps=cuda_num_warps,
                    kernel_variant="auto",
                    layout_variant="detector_vu",
                    state_mode="cached",
                ),
            )
        forward_projector = "jax"
        backprojector = "jax"
    b = chunk_size(n_views, views_per_batch, default=1)
    num_chunks = (n_views + b - 1) // b

    def body(
        carry: tuple[jnp.ndarray, jnp.ndarray],
        i: jnp.ndarray,
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], None]:
        loss_acc, grad_acc = carry
        start_shifted, valid_mask, _view_idx = chunk_schedule(i, n_views=n_views, chunk_size=b)
        T_chunk = jax.lax.dynamic_slice_in_dim(T_all, start_shifted, b)
        y_chunk = (
            read_views(projections, start_shifted, (b, detector.nv, detector.nu))
            if stream_projections
            else jax.lax.dynamic_slice_in_dim(projections, start_shifted, b)
        )
        w_chunk = jax.lax.dynamic_slice_in_dim(weights, start_shifted, b)
        valid = valid_mask[:, None, None]
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
            cuda_tile_shape=cuda_tile_shape,
            cuda_num_warps=cuda_num_warps,
            cuda_step_size_scale=cuda_step_size_scale,
            cuda_unroll=cuda_unroll,
            ray_integrator=ray_integrator,
            sampled_step_size=sampled_step_size,
        )
        raw_resid = (pred - y_chunk).astype(jnp.float32) * valid
        grad_resid = raw_resid * w_chunk
        loss_batch = (
            jnp.float32(0.5) * jnp.vdot(raw_resid, grad_resid).real
            if compute_loss
            else jnp.asarray(0.0, dtype=jnp.float32)
        )
        grad_acc = _accumulate_backprojection(
            T_chunk,
            grad_resid,
            grad_acc,
            grid=grid,
            detector=detector,
            det_grid=det_grid,
            backprojector=backprojector,
            ray_integrator=ray_integrator,
            sampled_step_size=sampled_step_size,
            projector_unroll=projector_unroll,
            gather_dtype=gather_dtype,
        )
        return (loss_acc + loss_batch, grad_acc), None

    init = (jnp.asarray(0.0, dtype=jnp.float32), jnp.zeros_like(volume))
    (loss, grad), _ = jax.lax.scan(
        body,
        init,
        jnp.arange(num_chunks, dtype=jnp.int32),
    )
    return loss, grad


def _accumulate_backprojection(
    poses: jnp.ndarray,
    residual: jnp.ndarray,
    accumulate: jnp.ndarray,
    *,
    grid: Grid,
    detector: Detector,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    backprojector: str,
    ray_integrator: str,
    sampled_step_size: float | None,
    projector_unroll: int,
    gather_dtype: str,
) -> jnp.ndarray:
    """Accumulate the selected forward model's exact transpose, not a surrogate."""
    if ray_integrator in {"joseph", "joseph_cubic"}:
        from tomojax.core.joseph import plane_coefficients, sum_backproject_planes

        return sum_backproject_planes(
            plane_coefficients(poses, grid, detector, det_grid),
            residual,
            grid,
            detector,
            backend=backprojector,
            interpolation="cubic" if ray_integrator == "joseph_cubic" else "linear",
            accumulate=accumulate,
        )
    if ray_integrator == "exact":
        from tomojax.core.trilinear import exact_adjoint

        batch = exact_adjoint(
            poses, grid, detector, residual, backend=backprojector, det_grid=det_grid
        )
    else:
        batch = _resolve_sum_backproject_views(backprojector)(
            poses,
            grid,
            detector,
            residual,
            # Transpose the requested sampling, including JAX fallback.
            step_size=sampled_step_size,
            unroll=1 if backprojector == "cuda" else int(projector_unroll),
            gather_dtype=gather_dtype,
            det_grid=det_grid,
        )
    return accumulate + batch


def _resolve_sum_backproject_views(backprojector: str) -> Callable[..., jnp.ndarray]:
    if backprojector == "jax":
        return sum_backproject_views_T
    if backprojector == "cuda":
        backproject_fn, fallback_reason = resolve_cuda_callable(
            "sum_backproject_views_T_cuda",
            missing_reason="cuda_sum_backproject_callable_missing",
        )
        if backproject_fn is None:
            raise RuntimeError(fallback_reason or "cuda_projector_unavailable")
        return backproject_fn
    raise ValueError("FistaCoreConfig.backprojector must be one of 'jax' or 'cuda'")


def _cuda_projector_options(**kwargs: object) -> object:
    capability = resolve_cuda_module()
    if capability.module is None:
        raise RuntimeError(capability.unavailable_reason or "cuda_projector_unavailable")
    options_cls = getattr(capability.module, "CudaProjectorOptions", None)
    if options_cls is None or not callable(options_cls):
        raise RuntimeError("cuda_projector_options_missing")
    return options_cls(**kwargs)


def effective_fista_core_backend(
    cfg: FistaCoreConfig,
    *,
    T_all: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> str:
    """Return the effective data projector backend selected by the core."""
    forward_projector, backprojector = _resolve_fista_core_data_projectors(
        cfg,
        T_all=T_all,
        det_grid=det_grid,
    )
    return "cuda" if forward_projector == "cuda" and backprojector == "cuda" else "jax"


def _resolve_fista_core_data_projectors(
    cfg: FistaCoreConfig,
    *,
    T_all: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> tuple[str, str]:
    # Explicit backend requests must not change with benchmark dimensions.
    if cfg.ray_integrator in {"exact", "joseph", "joseph_cubic"}:
        # Exact and Joseph CUDA operators consume dynamic poses directly. They expose a
        # matched explicit adjoint, not automatic higher derivatives; use the
        # JAX backend when differentiating this reconstruction core.
        return cfg.forward_projector, cfg.backprojector
    fixed_geometry_available = _fixed_geometry_is_available(T_all, det_grid)
    if (
        cfg.forward_projector == "cuda"
        and cfg.backprojector == "cuda"
        and not fixed_geometry_available
    ):
        return "jax", "jax"
    return cfg.forward_projector, cfg.backprojector


def _fixed_geometry_is_available(
    T_all: jnp.ndarray, det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None
) -> bool:
    try:
        arr = np.asarray(T_all, dtype=np.float32)
        # Detector calibration can be traced even when the poses are constants.
        for coord in det_grid or ():
            np.asarray(coord)
    except Exception:
        return False
    return arr.ndim == 3 and arr.shape[1:] == (4, 4) and bool(np.isfinite(arr).all())


def _apply_support(x: jnp.ndarray, support: jnp.ndarray | None) -> jnp.ndarray:
    if support is None:
        return x
    return x * jnp.asarray(support, dtype=x.dtype)


def _project_constraints(x: jnp.ndarray, cfg: FistaCoreConfig) -> jnp.ndarray:
    out = x
    lower = cfg.lower_bound
    if cfg.nonnegative:
        lower = 0.0 if lower is None else max(0.0, float(lower))
    if lower is not None:
        out = jnp.maximum(out, jnp.asarray(lower, dtype=out.dtype))
    if cfg.upper_bound is not None:
        out = jnp.minimum(out, jnp.asarray(cfg.upper_bound, dtype=out.dtype))
    return out
