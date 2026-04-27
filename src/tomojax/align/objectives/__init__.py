from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from tomojax.core.geometry import Detector, Grid
from tomojax.core.projector import forward_project_view_T

from ..geometry.geometry_applier import BaseGeometryArrays, apply_alignment_state
from .loss_adapters import LossAdapter, build_loss_adapter
from .loss_specs import AlignmentLossSpec
from ..model.state import AlignmentState


ObjectiveKind = Literal["fixed_volume", "bilevel_cv", "all_data_bilevel"]
DifferentiationMode = Literal["none", "unrolled", "implicit"]
InnerInitPolicy = Literal["zeros", "current_level_volume", "previous_fold_volume", "previous_stage_volume"]


@dataclass(frozen=True, slots=True)
class ObjectiveProvenance:
    outer_loss_source: str
    outer_loss_kind: str
    inner_data_term: str
    inner_regulariser: str
    validation_split: str
    differentiation_mode: DifferentiationMode
    initialization_policy: InnerInitPolicy

    def to_dict(self) -> dict[str, str]:
        return {
            "outer_loss_source": self.outer_loss_source,
            "outer_loss_kind": self.outer_loss_kind,
            "inner_data_term": self.inner_data_term,
            "inner_regulariser": self.inner_regulariser,
            "validation_split": self.validation_split,
            "differentiation_mode": self.differentiation_mode,
            "initialization_policy": self.initialization_policy,
        }


@dataclass(frozen=True, slots=True)
class ObjectiveResult:
    value: jnp.ndarray
    aux: dict[str, object]


class AlignmentObjective:
    kind: ObjectiveKind

    def evaluate(self, state: AlignmentState) -> ObjectiveResult:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FixedVolumeProjectionObjective(AlignmentObjective):
    base: BaseGeometryArrays
    grid: Grid
    detector: Detector
    projections: jnp.ndarray
    volume: jnp.ndarray
    loss_adapter: LossAdapter
    provenance: ObjectiveProvenance
    views_per_batch: int = 1
    projector_unroll: int = 1
    checkpoint_projector: bool = True
    gather_dtype: str = "fp32"
    kind: ObjectiveKind = "fixed_volume"

    @classmethod
    def from_loss_spec(
        cls,
        *,
        base: BaseGeometryArrays,
        grid: Grid,
        detector: Detector,
        projections: jnp.ndarray,
        volume: jnp.ndarray,
        loss_spec: AlignmentLossSpec,
        **kwargs,
    ) -> "FixedVolumeProjectionObjective":
        adapter = build_loss_adapter(loss_spec, projections)
        return cls(
            base=base,
            grid=grid,
            detector=detector,
            projections=jnp.asarray(projections, dtype=jnp.float32),
            volume=jnp.asarray(volume, dtype=jnp.float32),
            loss_adapter=adapter,
            provenance=ObjectiveProvenance(
                outer_loss_source="AlignmentLossSpec",
                outer_loss_kind=adapter.name,
                inner_data_term="none",
                inner_regulariser="none",
                validation_split="none",
                differentiation_mode="none",
                initialization_policy="current_level_volume",
            ),
            **kwargs,
        )

    def evaluate(self, state: AlignmentState) -> ObjectiveResult:
        effective = apply_alignment_state(self.base, state)
        value = project_and_score_stack(
            pose_stack=effective.pose_stack,
            grid=self.grid,
            detector=self.detector,
            volume=self.volume,
            det_grid=effective.det_grid,
            targets=self.projections,
            loss_adapter=self.loss_adapter,
            views_per_batch=self.views_per_batch,
            projector_unroll=self.projector_unroll,
            checkpoint_projector=self.checkpoint_projector,
            gather_dtype=self.gather_dtype,
            view_indices=jnp.arange(int(self.projections.shape[0]), dtype=jnp.int32),
        )
        return ObjectiveResult(
            value=value,
            aux={
                "objective_kind": self.kind,
                "loss_kind": self.loss_adapter.name,
                "objective_provenance": self.provenance.to_dict(),
            },
        )


def project_stack(
    *,
    pose_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    views_per_batch: int = 1,
    projector_unroll: int = 1,
    checkpoint_projector: bool = True,
    gather_dtype: str = "fp32",
) -> jnp.ndarray:
    n_views = int(pose_stack.shape[0])
    if n_views == 0:
        return jnp.zeros((0, detector.nv, detector.nu), dtype=jnp.float32)
    b = _chunk_size(n_views, views_per_batch)
    num_chunks = (n_views + b - 1) // b
    vm_project = jax.vmap(
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
    )

    def body(out, i):
        start_shifted, _vmask, view_idx = _chunk_schedule(i, n_views=n_views, chunk_size=b)
        T_chunk = jax.lax.dynamic_slice(pose_stack, (start_shifted, 0, 0), (b, 4, 4))
        pred = vm_project(T_chunk)
        return out.at[view_idx].set(pred), None

    init = jnp.zeros((n_views, detector.nv, detector.nu), dtype=jnp.float32)
    out, _ = jax.lax.scan(body, init, jnp.arange(num_chunks, dtype=jnp.int32))
    return out


def project_and_score_stack(
    *,
    pose_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    targets: jnp.ndarray,
    loss_adapter: LossAdapter,
    views_per_batch: int = 1,
    projector_unroll: int = 1,
    checkpoint_projector: bool = True,
    gather_dtype: str = "fp32",
    view_mask: jnp.ndarray | None = None,
    view_indices: jnp.ndarray | None = None,
) -> jnp.ndarray:
    n_views = int(pose_stack.shape[0])
    if n_views == 0:
        return jnp.asarray(0.0, dtype=jnp.float32)
    b = _chunk_size(n_views, views_per_batch)
    num_chunks = (n_views + b - 1) // b
    nv = int(targets.shape[1])
    nu = int(targets.shape[2])
    loss_mask = getattr(loss_adapter.state, "mask", None)
    vm_project = jax.vmap(
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
    )
    local_indices = (
        jnp.arange(n_views, dtype=jnp.int32)
        if view_indices is None
        else jnp.asarray(view_indices, dtype=jnp.int32)
    )
    per_view_weight = (
        jnp.ones((n_views,), dtype=jnp.float32)
        if view_mask is None
        else jnp.asarray(view_mask, dtype=jnp.float32).reshape((n_views,))
    )

    def body(loss_acc, i):
        start_shifted, valid_mask, _view_idx = _chunk_schedule(
            i, n_views=n_views, chunk_size=b
        )
        T_chunk = jax.lax.dynamic_slice(pose_stack, (start_shifted, 0, 0), (b, 4, 4))
        y_chunk = jax.lax.dynamic_slice(targets, (start_shifted, 0, 0), (b, nv, nu))
        idx_chunk = jax.lax.dynamic_slice(local_indices, (start_shifted,), (b,))
        weight_chunk = jax.lax.dynamic_slice(per_view_weight, (start_shifted,), (b,))
        pred = vm_project(T_chunk)
        if loss_mask is None:
            image_mask = None
        else:
            image_mask = jax.lax.dynamic_slice(loss_mask, (start_shifted, 0, 0), (b, nv, nu))
        losses = loss_adapter.per_view_loss(
            pred,
            y_chunk,
            image_mask,
            view_indices=idx_chunk,
        )
        return loss_acc + jnp.sum(losses * valid_mask * weight_chunk), None

    loss, _ = jax.lax.scan(
        body,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.arange(num_chunks, dtype=jnp.int32),
    )
    return loss


def _chunk_size(n_views: int, views_per_batch: int | None) -> int:
    b = int(views_per_batch) if views_per_batch is not None and int(views_per_batch) > 0 else int(n_views)
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


def score_projection_stack(
    adapter: LossAdapter,
    pred: jnp.ndarray,
    targets: jnp.ndarray,
    *,
    mask: jnp.ndarray | None = None,
    view_indices: jnp.ndarray | None = None,
) -> jnp.ndarray:
    losses = adapter.per_view_loss(
        pred,
        targets,
        getattr(adapter.state, "mask", None),
        view_indices=view_indices,
    )
    if mask is not None:
        losses = losses * jnp.asarray(mask, dtype=jnp.float32)
    return jnp.sum(losses)
