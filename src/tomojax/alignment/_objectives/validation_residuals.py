"""Validation residual and normal-equation accumulation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from tomojax.alignment._geometry.geometry_applier import (
    BaseGeometryArrays,
    apply_alignment_state,
    subset_base_geometry,
)
from tomojax.core.projector import forward_project_view_T

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomojax.alignment._model.dof_specs import ActiveParameterView
    from tomojax.alignment._model.state import AlignmentState
    from tomojax.core.geometry import Detector, Grid

    from .loss_adapters import LossAdapter


@dataclass(frozen=True, slots=True)
class ValidationNormalResult:
    """Validation loss, gradient, Hessian approximation, and diagnostics."""

    loss: jnp.ndarray
    grad: jnp.ndarray
    hess: jnp.ndarray
    residual_count: jnp.ndarray
    diagnostics: dict[str, object]


@dataclass(frozen=True, slots=True)
class FoldValidation:
    """What one fold's validation residuals need: its fixed volume, views and operator."""

    frozen_state: AlignmentState
    active_view: ActiveParameterView
    base: BaseGeometryArrays
    grid: Grid
    detector: Detector
    projections: jnp.ndarray
    loss_adapter: LossAdapter
    fold_volume: jnp.ndarray
    val_idx: jnp.ndarray
    val_mask: jnp.ndarray
    views_per_batch: int
    projector_unroll: int
    checkpoint_projector: bool
    gather_dtype: str
    ray_integrator: str = "sampled"


@dataclass(frozen=True, slots=True)
class _Residuals:
    """One fold's weighted residuals, a chunk of views at a time, as a function of ``z``."""

    chunk: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    chunks: int


def accumulate_validation_normals(fold: FoldValidation, z: jnp.ndarray) -> ValidationNormalResult:
    """Validation GN normal equations of one fixed fold volume at the active parameters ``z``."""
    z0 = jnp.asarray(z, dtype=jnp.float32).reshape(-1)
    d = int(z0.size)
    eye = jnp.eye(d, dtype=jnp.float32)
    zeros = (jnp.asarray(0.0, jnp.float32), jnp.zeros((d,), jnp.float32),
             jnp.zeros((d, d), jnp.float32), jnp.asarray(0, jnp.int32))  # fmt: skip
    residuals = _validation_residuals(fold)
    if residuals is None:
        return ValidationNormalResult(*zeros, diagnostics={"active_dim": d})
    chunk = residuals.chunk

    def body(
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], i: jnp.ndarray
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], None]:
        loss_acc, grad_acc, hess_acc, count_acc = carry
        r, lin = jax.linearize(lambda zz: chunk(zz, i), z0)
        cols = jax.vmap(lin)(eye)
        return (
            loss_acc + jnp.float32(0.5) * jnp.vdot(r, r).real,
            grad_acc + cols @ r,
            hess_acc + cols @ cols.T,
            count_acc + jnp.sum(jnp.isfinite(r).astype(jnp.int32)),
        ), None

    steps = jnp.arange(residuals.chunks, dtype=jnp.int32)
    (loss, grad, hess, count), _ = jax.lax.scan(body, zeros, steps)
    diagnostics = {"validation_chunks": int(residuals.chunks), "active_dim": d}
    return ValidationNormalResult(loss, grad, hess, count, diagnostics)


def validation_loss(fold: FoldValidation, z: jnp.ndarray) -> jnp.ndarray:
    """One fixed fold volume's validation loss at ``z``: one projection, no linearisation."""
    z0 = jnp.asarray(z, dtype=jnp.float32).reshape(-1)
    residuals = _validation_residuals(fold)
    if residuals is None:
        return jnp.asarray(0.0, jnp.float32)
    chunk = residuals.chunk

    def body(total: jnp.ndarray, i: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        r = chunk(z0, i)
        return total + jnp.float32(0.5) * jnp.vdot(r, r).real, None

    steps = jnp.arange(residuals.chunks, dtype=jnp.int32)
    total, _ = jax.lax.scan(body, jnp.asarray(0.0, jnp.float32), steps)
    return total


def _validation_residuals(fold: FoldValidation) -> _Residuals | None:
    """The residual function of one fold, or None for a fold without validation views."""
    loss_adapter, detector, val_idx = fold.loss_adapter, fold.detector, fold.val_idx
    if not bool(loss_adapter.supports_setup_validation_lm):
        raise ValueError(
            f"Loss {loss_adapter.name!r} does not support setup validation-LM residuals"
        )
    n_views = int(val_idx.shape[0])
    if n_views == 0:
        return None
    b = _chunk_size(n_views, fold.views_per_batch)
    val_idx = jnp.asarray(val_idx, dtype=jnp.int32).reshape((n_views,))
    val_mask = jnp.asarray(fold.val_mask, dtype=jnp.float32).reshape((n_views,))
    val_base = subset_base_geometry(fold.base, val_idx)
    targets = jnp.asarray(fold.projections, dtype=jnp.float32)[val_idx]

    def residual_chunk(z_candidate: jnp.ndarray, i: jnp.ndarray) -> jnp.ndarray:
        state = fold.active_view.unpack(fold.frozen_state, z_candidate)
        val_state = state.replace(
            pose=state.pose.replace(pose_params=state.pose.pose_params[val_idx])
        )
        effective = apply_alignment_state(val_base, val_state)
        start_shifted, valid_mask, _view_idx = _chunk_schedule(i, n_views=n_views, chunk_size=b)
        T_chunk = jax.lax.dynamic_slice(effective.pose_stack, (start_shifted, 0, 0), (b, 4, 4))
        y_chunk = jax.lax.dynamic_slice(
            targets, (start_shifted, 0, 0), (b, detector.nv, detector.nu)
        )
        global_idx = jax.lax.dynamic_slice(val_idx, (start_shifted,), (b,))
        view_weight = jax.lax.dynamic_slice(val_mask, (start_shifted,), (b,))

        def project_one(T: jnp.ndarray) -> jnp.ndarray:
            return forward_project_view_T(
                T,
                fold.grid,
                detector,
                fold.fold_volume,
                use_checkpoint=fold.checkpoint_projector,
                unroll=int(fold.projector_unroll),
                gather_dtype=fold.gather_dtype,
                det_grid=effective.det_grid,
                ray_integrator=fold.ray_integrator,
            )

        pred = jax.vmap(project_one)(T_chunk)
        mask_state = getattr(loss_adapter.state, "mask", None)
        image_mask = None if mask_state is None else mask_state[global_idx]
        weights = loss_adapter.gauss_newton_weights(y_chunk, image_mask)
        resid = (pred - y_chunk).astype(jnp.float32) * weights
        resid = resid * valid_mask[:, None, None] * view_weight[:, None, None]
        return resid.reshape(-1)

    return _Residuals(residual_chunk, (n_views + b - 1) // b)


def _chunk_size(n_views: int, views_per_batch: int | None) -> int:
    b = (
        int(views_per_batch)
        if views_per_batch is not None and int(views_per_batch) > 0
        else int(n_views)
    )
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
