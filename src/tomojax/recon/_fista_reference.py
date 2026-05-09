"""JAX reference FISTA preview reconstruction."""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import csv
from dataclasses import dataclass
import time
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from tomojax.core.projector import forward_project_view_T
from tomojax.forward import (
    CoreProjectionGeometry,
    ResidualFilterConfig,
    apply_residual_filter,
    apply_residual_filter_schedule,
    core_projection_geometry_from_state,
    pseudo_huber_loss,
    residual_loss,
)
from tomojax.geometry import CORE_X_AXIS, CORE_Y_AXIS
from tomojax.recon._backprojection_accumulation import sum_backproject_views_chunked

if TYPE_CHECKING:
    from pathlib import Path

    from tomojax.geometry import GeometryState


@dataclass(frozen=True)
class ReferenceFISTAConfig:
    iterations: int = 8
    step_size: float = 1e-2
    tv_weight: float = 0.0
    tv_delta: float = 1e-3
    residual_sigma: float = 1.0
    residual_delta: float = 1.0
    residual_loss_mode: str = "pseudo_huber"
    residual_filters: tuple[ResidualFilterConfig, ...] = (ResidualFilterConfig(),)
    non_negative: bool = True
    center_l2_weight: float = 0.0
    views_per_batch: int = 1


@dataclass(frozen=True)
class ReferenceFISTATraceRow:
    iteration: int
    loss: float
    data_loss: float
    regulariser: float
    step_size: float
    wall_time_s: float
    backend: str


@dataclass(frozen=True)
class ReferenceFISTAResult:
    volume: jax.Array
    trace: tuple[ReferenceFISTATraceRow, ...]
    config: ReferenceFISTAConfig


def fista_reconstruct_reference(
    projections: jax.Array,
    geometry: GeometryState,
    *,
    initial_volume: jax.Array | None = None,
    volume_support: jax.Array | None = None,
    mask: jax.Array | None = None,
    config: ReferenceFISTAConfig | None = None,
) -> ReferenceFISTAResult:
    """Run the v2 preview FISTA path through the core explicit adjoint."""
    cfg = config or ReferenceFISTAConfig()
    observed = jnp.asarray(projections, dtype=jnp.float32)
    if observed.ndim != 3:
        raise ValueError("projections must have shape (views, rows, cols)")

    if initial_volume is None:
        volume = jnp.zeros(
            (observed.shape[1], observed.shape[1], observed.shape[2]),
            dtype=jnp.float32,
        )
    else:
        volume = jnp.asarray(initial_volume, dtype=jnp.float32)
        if volume.ndim != 3:
            raise ValueError("initial_volume must be 3D")
    volume_shape = (int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2]))
    support = _support_or_none(volume_support, shape=volume_shape)
    volume = _project_constraints(volume, support=support, config=cfg)
    start = time.perf_counter()
    core = core_projection_geometry_from_state(
        volume_shape,
        geometry,
        detector_shape=(int(observed.shape[1]), int(observed.shape[2])),
    )
    y = volume
    t = jnp.asarray(1.0, dtype=jnp.float32)
    step_size = jnp.asarray(cfg.step_size, dtype=jnp.float32)
    trace: list[ReferenceFISTATraceRow] = []
    for iteration in range(max(0, int(cfg.iterations))):
        loss_value, data_value, regulariser_value, gradient = _loss_and_explicit_gradient(
            y,
            observed,
            core=core,
            mask=mask,
            config=cfg,
        )
        candidate = y - step_size * gradient
        candidate = _project_constraints(candidate, support=support, config=cfg)
        next_t = (1.0 + jnp.sqrt(1.0 + 4.0 * t * t)) / 2.0
        momentum = (t - 1.0) / next_t
        y = _project_constraints(
            candidate + momentum * (candidate - volume),
            support=support,
            config=cfg,
        )
        volume = candidate
        t = next_t
        trace.append(
            ReferenceFISTATraceRow(
                iteration=iteration,
                loss=float(loss_value),
                data_loss=float(data_value),
                regulariser=float(regulariser_value),
                step_size=float(step_size),
                wall_time_s=time.perf_counter() - start,
                backend="core_trilinear_ray_explicit_adjoint",
            )
        )
    return ReferenceFISTAResult(volume=volume.astype(jnp.float32), trace=tuple(trace), config=cfg)


def write_fista_trace_csv(result: ReferenceFISTAResult, path: str | Path) -> Path:
    """Write the FISTA trace artifact as CSV."""
    from pathlib import Path

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "iteration",
                "loss",
                "data_loss",
                "regulariser",
                "step_size",
                "wall_time_s",
                "backend",
            ],
        )
        writer.writeheader()
        for row in result.trace:
            writer.writerow(
                {
                    "iteration": row.iteration,
                    "loss": row.loss,
                    "data_loss": row.data_loss,
                    "regulariser": row.regulariser,
                    "step_size": row.step_size,
                    "wall_time_s": row.wall_time_s,
                    "backend": row.backend,
                }
            )
    return output_path


def _loss_and_explicit_gradient(
    volume: jax.Array,
    observed: jax.Array,
    *,
    core: CoreProjectionGeometry,
    mask: jax.Array | None,
    config: ReferenceFISTAConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    data_loss, data_gradient = _projection_loss_and_explicit_gradient_chunked(
        volume,
        observed,
        core=core,
        mask=mask,
        config=config,
    )
    tv_value, tv_gradient = _smoothed_tv_value_and_grad(
        volume,
        delta=config.tv_delta,
    )
    center_value, center_gradient = _center_l2_value_and_grad(volume)
    scaled_tv = jnp.asarray(config.tv_weight, dtype=jnp.float32) * tv_value
    scaled_center = jnp.asarray(config.center_l2_weight, dtype=jnp.float32) * center_value
    scaled_regulariser = scaled_tv + scaled_center
    gradient = (
        data_gradient
        + jnp.asarray(config.tv_weight, dtype=jnp.float32) * tv_gradient
        + jnp.asarray(config.center_l2_weight, dtype=jnp.float32) * center_gradient
    )
    return data_loss + scaled_regulariser, data_loss, scaled_regulariser, gradient


def _projection_loss_and_explicit_gradient_chunked(
    volume: jax.Array,
    observed: jax.Array,
    *,
    core: CoreProjectionGeometry,
    mask: jax.Array | None,
    config: ReferenceFISTAConfig,
) -> tuple[jax.Array, jax.Array]:
    obs = jnp.asarray(observed, dtype=jnp.float32)
    n_views = int(obs.shape[0])
    if n_views == 0:
        return jnp.asarray(0.0, dtype=jnp.float32), jnp.zeros_like(volume)
    b = _chunk_size(n_views, config.views_per_batch)
    num_chunks = (n_views + b - 1) // b
    normalizer = jnp.asarray(obs.size, dtype=jnp.float32)

    def body(
        carry: tuple[jax.Array, jax.Array],
        chunk_index: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        loss_acc, grad_acc = carry
        start, valid, _view_idx = _chunk_schedule(chunk_index, n_views=n_views, chunk_size=b)
        t_chunk = jax.lax.dynamic_slice(core.t_all, (start, 0, 0), (b, 4, 4))
        obs_chunk = jax.lax.dynamic_slice(
            obs,
            (start, 0, 0),
            (b, core.detector.nv, core.detector.nu),
        )
        mask_chunk = (
            None
            if mask is None
            else jax.lax.dynamic_slice(
                jnp.asarray(mask, dtype=jnp.float32),
                (start, 0, 0),
                (b, core.detector.nv, core.detector.nu),
            )
        )
        predicted = _project_views(volume, core=core, t_all=t_chunk)
        whitened = (predicted - obs_chunk) / jnp.asarray(
            config.residual_sigma,
            dtype=jnp.float32,
        )
        filtered = apply_residual_filter_schedule(
            whitened,
            config.residual_filters,
            mask=mask_chunk,
        ).residual
        filtered = filtered * valid[:, None, None]
        data = residual_loss(
            filtered,
            jnp.zeros_like(filtered),
            mask=None,
            sigma=1.0,
            delta=config.residual_delta,
            mode="l2" if config.residual_loss_mode == "l2" else "pseudo_huber",
        )
        loss_map = (
            pseudo_huber_loss(filtered, delta=config.residual_delta)
            if config.residual_loss_mode != "l2"
            else jnp.asarray(0.5, dtype=jnp.float32) * filtered * filtered
        )
        loss_chunk = jnp.sum(loss_map) / jnp.maximum(normalizer, 1.0)
        data_grad_filtered = filtered * data.weights / jnp.maximum(normalizer, 1.0)
        data_grad_projection = _residual_filter_adjoint(
            data_grad_filtered,
            config.residual_filters,
            mask=mask_chunk,
        ) / jnp.asarray(config.residual_sigma, dtype=jnp.float32)
        data_grad_projection = data_grad_projection * valid[:, None, None]
        grad_chunk = sum_backproject_views_chunked(
            _core_with_views(core, t_chunk),
            data_grad_projection,
        )
        return (loss_acc + loss_chunk, grad_acc + grad_chunk), None

    init = (jnp.asarray(0.0, dtype=jnp.float32), jnp.zeros_like(volume))
    (loss, gradient), _ = jax.lax.scan(
        body,
        init,
        jnp.arange(num_chunks, dtype=jnp.int32),
    )
    return loss, gradient


def _residual_filter_adjoint(
    filtered_gradient: jax.Array,
    configs: tuple[ResidualFilterConfig, ...],
    *,
    mask: jax.Array | None,
) -> jax.Array:
    grad = jnp.asarray(filtered_gradient, dtype=jnp.float32)
    if mask is not None:
        mask_arr = jnp.asarray(mask, dtype=jnp.float32)
        if mask_arr.shape != grad.shape:
            raise ValueError("mask must match residual shape")
        grad = grad * mask_arr
    components = tuple(apply_residual_filter(grad, config, mask=None) for config in configs)
    return jnp.sum(jnp.stack(components, axis=0), axis=0)


def _project_views(
    volume: jax.Array,
    *,
    core: CoreProjectionGeometry,
    t_all: jax.Array,
) -> jax.Array:
    def project_one(t_view: jax.Array) -> jax.Array:
        return forward_project_view_T(
            t_view,
            core.grid,
            core.detector,
            volume,
            step_size=core.step_size,
            n_steps=core.n_steps,
            use_checkpoint=core.checkpoint_projector,
            unroll=core.projector_unroll,
            gather_dtype=core.gather_dtype,
            det_grid=core.det_grid,
        )

    return jax.vmap(project_one)(t_all).astype(jnp.float32)


def _core_with_views(core: CoreProjectionGeometry, t_all: jax.Array) -> CoreProjectionGeometry:
    return CoreProjectionGeometry(
        grid=core.grid,
        detector=core.detector,
        t_all=t_all,
        det_grid=core.det_grid,
        operator=core.operator,
        step_size=core.step_size,
        n_steps=core.n_steps,
        gather_dtype=core.gather_dtype,
        checkpoint_projector=core.checkpoint_projector,
        projector_unroll=core.projector_unroll,
        detector_roll_rad=core.detector_roll_rad,
        axis_rot_x_rad=core.axis_rot_x_rad,
        axis_rot_y_rad=core.axis_rot_y_rad,
        alpha_rad_max_abs=core.alpha_rad_max_abs,
        beta_rad_max_abs=core.beta_rad_max_abs,
        acquisition_model=core.acquisition_model,
        laminography_tilt_rad=core.laminography_tilt_rad,
        laminography_tilt_about=core.laminography_tilt_about,
    )


def _chunk_size(n_views: int, views_per_batch: int | None) -> int:
    b = (
        int(views_per_batch)
        if views_per_batch is not None and int(views_per_batch) > 0
        else int(n_views)
    )
    return max(1, min(int(b), int(n_views)))


def _chunk_schedule(
    chunk_index: jax.Array,
    *,
    n_views: int,
    chunk_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    b = jnp.asarray(chunk_size, dtype=jnp.int32)
    start = jnp.asarray(chunk_index, dtype=jnp.int32) * b
    remaining = jnp.maximum(0, jnp.asarray(n_views, dtype=jnp.int32) - start)
    valid = jnp.minimum(b, remaining)
    shift = b - valid
    start_shifted = jnp.maximum(0, start - shift)
    idx = jnp.arange(chunk_size, dtype=jnp.int32)
    valid_mask = (idx >= (b - valid)).astype(jnp.float32)
    return start_shifted, valid_mask, start_shifted + idx


def _support_or_none(
    volume_support: jax.Array | None, *, shape: tuple[int, int, int]
) -> jax.Array | None:
    if volume_support is None:
        return None
    support = jnp.asarray(volume_support, dtype=jnp.float32)
    if support.shape != shape:
        raise ValueError(f"volume_support must have shape {shape!r}")
    return support


def _project_constraints(
    volume: jax.Array,
    *,
    support: jax.Array | None,
    config: ReferenceFISTAConfig,
) -> jax.Array:
    projected = jnp.asarray(volume, dtype=jnp.float32)
    if config.non_negative:
        projected = jnp.maximum(projected, 0.0)
    if support is not None:
        projected = projected * support
    return projected


def _smoothed_tv_value_and_grad(
    volume: jax.Array,
    *,
    delta: float,
) -> tuple[jax.Array, jax.Array]:
    return jax.value_and_grad(_smoothed_tv)(volume, delta=delta)


def _smoothed_tv(volume: jax.Array, *, delta: float) -> jax.Array:
    vol = jnp.asarray(volume, dtype=jnp.float32)
    d = jnp.asarray(delta, dtype=jnp.float32)
    dx = vol[1:, :, :] - vol[:-1, :, :]
    dy = vol[:, 1:, :] - vol[:, :-1, :]
    dz = vol[:, :, 1:] - vol[:, :, :-1]
    return (
        jnp.sum(jnp.sqrt(dx * dx + d * d) - d)
        + jnp.sum(jnp.sqrt(dy * dy + d * d) - d)
        + jnp.sum(jnp.sqrt(dz * dz + d * d) - d)
    ) / jnp.asarray(vol.size, dtype=jnp.float32)


def _center_l2_value_and_grad(volume: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jax.value_and_grad(_center_l2)(volume)


def _center_l2(volume: jax.Array) -> jax.Array:
    vol = jnp.maximum(jnp.asarray(volume, dtype=jnp.float32), 0.0)
    mass = jnp.maximum(jnp.sum(vol), jnp.asarray(1.0e-6, dtype=jnp.float32))
    x_axis = _centered_axis(int(vol.shape[CORE_X_AXIS]))
    y_axis = _centered_axis(int(vol.shape[CORE_Y_AXIS]))
    x_com = jnp.sum(vol * x_axis[:, None, None]) / mass
    y_com = jnp.sum(vol * y_axis[None, :, None]) / mass
    return y_com * y_com + x_com * x_com


def _centered_axis(size: int) -> jax.Array:
    center = (float(size) - 1.0) / 2.0
    half_width = max(float(size) / 2.0, 1.0)
    return (jnp.arange(size, dtype=jnp.float32) - center) / half_width
