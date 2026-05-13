from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from tomojax.core.geometry import Detector, Grid
from tomojax.core.projector import forward_project_view_T

from ..geometry.geometry_applier import (
    BaseGeometryArrays,
    apply_alignment_state,
    subset_base_geometry,
)
from ..model.dof_specs import ActiveParameterView
from ..model.state import AlignmentState
from .loss_adapters import LossAdapter


@dataclass(frozen=True, slots=True)
class ValidationNormalResult:
    loss: jnp.ndarray
    grad: jnp.ndarray
    hess: jnp.ndarray
    residual_count: jnp.ndarray
    diagnostics: dict[str, object]


def accumulate_validation_normals(
    *,
    frozen_state: AlignmentState,
    active_view: ActiveParameterView,
    z: jnp.ndarray,
    base: BaseGeometryArrays,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    loss_adapter: LossAdapter,
    fold_volume: jnp.ndarray,
    val_idx: jnp.ndarray,
    val_mask: jnp.ndarray,
    views_per_batch: int,
    projector_unroll: int,
    checkpoint_projector: bool,
    gather_dtype: str,
) -> ValidationNormalResult:
    """Accumulate validation GN normal equations for one fixed fold volume."""
    if not bool(loss_adapter.supports_setup_validation_lm):
        raise ValueError(
            f"Loss {loss_adapter.name!r} does not support setup validation-LM residuals"
        )

    z0 = jnp.asarray(z, dtype=jnp.float32).reshape(-1)
    d = int(z0.size)
    n_views = int(val_idx.shape[0])
    if n_views == 0:
        zeros_h = jnp.zeros((d, d), dtype=jnp.float32)
        zeros_g = jnp.zeros((d,), dtype=jnp.float32)
        return ValidationNormalResult(
            loss=jnp.asarray(0.0, dtype=jnp.float32),
            grad=zeros_g,
            hess=zeros_h,
            residual_count=jnp.asarray(0, dtype=jnp.int32),
            diagnostics={"validation_projection_chunked": True, "active_dim": d},
        )

    b = _chunk_size(n_views, views_per_batch)
    num_chunks = (n_views + b - 1) // b
    val_idx = jnp.asarray(val_idx, dtype=jnp.int32).reshape((n_views,))
    val_mask = jnp.asarray(val_mask, dtype=jnp.float32).reshape((n_views,))
    val_base = subset_base_geometry(base, val_idx)
    targets = jnp.asarray(projections, dtype=jnp.float32)[val_idx]
    local_indices = jnp.arange(n_views, dtype=jnp.int32)
    eye = jnp.eye(d, dtype=jnp.float32)

    def residual_chunk(z_candidate: jnp.ndarray, i: jnp.ndarray) -> jnp.ndarray:
        state = active_view.unpack(frozen_state, z_candidate)
        val_state = state.replace(pose=state.pose.replace(params5=state.pose.params5[val_idx]))
        effective = apply_alignment_state(val_base, val_state)
        start_shifted, valid_mask, _view_idx = _chunk_schedule(
            i,
            n_views=n_views,
            chunk_size=b,
        )
        T_chunk = jax.lax.dynamic_slice(effective.pose_stack, (start_shifted, 0, 0), (b, 4, 4))
        y_chunk = jax.lax.dynamic_slice(
            targets, (start_shifted, 0, 0), (b, detector.nv, detector.nu)
        )
        global_idx = jax.lax.dynamic_slice(val_idx, (start_shifted,), (b,))
        local_idx = jax.lax.dynamic_slice(local_indices, (start_shifted,), (b,))
        view_weight = jax.lax.dynamic_slice(val_mask, (start_shifted,), (b,))

        def project_one(T):
            return forward_project_view_T(
                T,
                grid,
                detector,
                fold_volume,
                use_checkpoint=checkpoint_projector,
                unroll=int(projector_unroll),
                gather_dtype=gather_dtype,
                det_grid=effective.det_grid,
            )

        pred = jax.vmap(project_one)(T_chunk)
        mask_state = getattr(loss_adapter.state, "mask", None)
        image_mask = None if mask_state is None else mask_state[global_idx]
        weights = loss_adapter.gauss_newton_weights(y_chunk, image_mask)
        resid = (pred - y_chunk).astype(jnp.float32) * weights
        resid = resid * valid_mask[:, None, None] * view_weight[:, None, None]
        # Keep view_indices dependency explicit for losses whose masks are global.
        del local_idx
        return resid.reshape(-1)

    def body(carry, i):
        loss_acc, grad_acc, hess_acc, count_acc = carry
        r, lin = jax.linearize(lambda zz: residual_chunk(zz, i), z0)
        cols = jax.vmap(lin)(eye)
        loss_i = jnp.float32(0.5) * jnp.vdot(r, r).real
        grad_i = cols @ r
        hess_i = cols @ cols.T
        count_i = jnp.sum(jnp.isfinite(r).astype(jnp.int32))
        return (
            loss_acc + loss_i,
            grad_acc + grad_i,
            hess_acc + hess_i,
            count_acc + count_i,
        ), None

    init = (
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.zeros((d,), dtype=jnp.float32),
        jnp.zeros((d, d), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    (loss, grad, hess, count), _ = jax.lax.scan(
        body,
        init,
        jnp.arange(num_chunks, dtype=jnp.int32),
    )
    return ValidationNormalResult(
        loss=loss,
        grad=grad,
        hess=hess,
        residual_count=count,
        diagnostics={
            "validation_projection_chunked": True,
            "validation_chunks": int(num_chunks),
            "views_per_batch": int(b),
            "active_dim": int(d),
        },
    )


def score_validation_fixed_volume(
    *,
    frozen_state: AlignmentState,
    active_view: ActiveParameterView,
    z: jnp.ndarray,
    base: BaseGeometryArrays,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    loss_adapter: LossAdapter,
    fold_volume: jnp.ndarray,
    val_idx: jnp.ndarray,
    val_mask: jnp.ndarray,
    views_per_batch: int,
    projector_unroll: int,
    checkpoint_projector: bool,
    gather_dtype: str,
) -> jnp.ndarray:
    normals = accumulate_validation_normals(
        frozen_state=frozen_state,
        active_view=active_view,
        z=z,
        base=base,
        grid=grid,
        detector=detector,
        projections=projections,
        loss_adapter=loss_adapter,
        fold_volume=fold_volume,
        val_idx=val_idx,
        val_mask=val_mask,
        views_per_batch=views_per_batch,
        projector_unroll=projector_unroll,
        checkpoint_projector=checkpoint_projector,
        gather_dtype=gather_dtype,
    )
    return normals.loss


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
