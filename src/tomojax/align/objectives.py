from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry import Detector, Grid
from tomojax.core.projector import forward_project_view_T

from .dof_specs import ActiveParameterView
from .geometry_applier import BaseGeometryArrays, apply_alignment_state, subset_base_geometry
from .losses import AlignmentLossSpec, LossAdapter, build_loss_adapter
from .state import AlignmentState


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


@dataclass(frozen=True, slots=True)
class FoldArrays:
    train_idx: jnp.ndarray
    train_mask: jnp.ndarray
    val_idx: jnp.ndarray
    val_mask: jnp.ndarray

    @property
    def n_folds(self) -> int:
        return int(self.train_idx.shape[0])

    def to_metadata(self) -> dict[str, object]:
        return {
            "n_folds": int(self.train_idx.shape[0]),
            "max_train": int(self.train_idx.shape[1]),
            "max_val": int(self.val_idx.shape[1]),
            "train_counts": [int(v) for v in np.asarray(jnp.sum(self.train_mask, axis=1))],
            "val_counts": [int(v) for v in np.asarray(jnp.sum(self.val_mask, axis=1))],
        }


@dataclass(frozen=True, slots=True)
class FoldSpec:
    n_folds: int = 4
    mode: Literal["interleaved"] = "interleaved"

    def build(self, n_views: int) -> FoldArrays:
        n = int(n_views)
        k = int(self.n_folds)
        if k < 2:
            raise ValueError("bilevel CV requires at least two folds")
        if n < k:
            raise ValueError(
                f"bilevel CV requires at least {k} views for {k} folds; got {n}"
            )
        indices = np.arange(n, dtype=np.int32)
        val_parts = [indices[(indices % k) == fold] for fold in range(k)]
        if any(part.size == 0 for part in val_parts):
            raise ValueError("bilevel CV split produced an empty validation fold")
        train_parts = [np.setdiff1d(indices, val, assume_unique=True) for val in val_parts]
        max_train = max(int(part.size) for part in train_parts)
        max_val = max(int(part.size) for part in val_parts)

        def pad(parts: list[np.ndarray], width: int) -> tuple[jnp.ndarray, jnp.ndarray]:
            idx = np.zeros((k, width), dtype=np.int32)
            mask = np.zeros((k, width), dtype=np.float32)
            for row, part in enumerate(parts):
                idx[row, : part.size] = part
                mask[row, : part.size] = 1.0
            return jnp.asarray(idx, dtype=jnp.int32), jnp.asarray(mask, dtype=jnp.float32)

        train_idx, train_mask = pad(train_parts, max_train)
        val_idx, val_mask = pad(val_parts, max_val)
        return FoldArrays(
            train_idx=train_idx,
            train_mask=train_mask,
            val_idx=val_idx,
            val_mask=val_mask,
        )


def objective_value_and_grad(
    objective: "AlignmentObjective",
    view: ActiveParameterView,
    frozen_state: AlignmentState,
) -> Callable[[jnp.ndarray], tuple[tuple[jnp.ndarray, dict[str, object]], jnp.ndarray]]:
    def value_fn(active_values: jnp.ndarray) -> tuple[jnp.ndarray, dict[str, object]]:
        state = view.unpack(frozen_state, active_values)
        result = objective.evaluate(state)
        return result.value, result.aux

    return jax.value_and_grad(value_fn, has_aux=True)


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


ReconFoldFn = Callable[[AlignmentState, jnp.ndarray, jnp.ndarray, BaseGeometryArrays], jnp.ndarray]


@dataclass(frozen=True, slots=True)
class BilevelCVProjectionObjective(AlignmentObjective):
    base: BaseGeometryArrays
    grid: Grid
    detector: Detector
    projections: jnp.ndarray
    loss_spec: AlignmentLossSpec
    folds: FoldArrays
    fold_metadata: dict[str, object]
    val_loss_adapters: tuple[LossAdapter, ...]
    reconstruct_fold: ReconFoldFn
    provenance: ObjectiveProvenance
    views_per_batch: int = 1
    projector_unroll: int = 1
    checkpoint_projector: bool = True
    gather_dtype: str = "fp32"
    kind: ObjectiveKind = "bilevel_cv"

    @classmethod
    def from_loss_spec(
        cls,
        *,
        base: BaseGeometryArrays,
        grid: Grid,
        detector: Detector,
        projections: jnp.ndarray,
        loss_spec: AlignmentLossSpec,
        folds: FoldArrays,
        reconstruct_fold: ReconFoldFn,
        inner_regulariser: str = "huber_tv",
        differentiation_mode: DifferentiationMode = "unrolled",
        initialization_policy: InnerInitPolicy = "zeros",
        **kwargs,
    ) -> "BilevelCVProjectionObjective":
        projections_arr = jnp.asarray(projections, dtype=jnp.float32)
        adapter = build_loss_adapter(loss_spec, projections_arr)
        val_loss_adapters = tuple(
            build_loss_adapter(loss_spec, projections_arr[folds.val_idx[fold]])
            for fold in range(folds.n_folds)
        )
        return cls(
            base=base,
            grid=grid,
            detector=detector,
            projections=projections_arr,
            loss_spec=loss_spec,
            folds=folds,
            fold_metadata=folds.to_metadata(),
            val_loss_adapters=val_loss_adapters,
            reconstruct_fold=reconstruct_fold,
            provenance=ObjectiveProvenance(
                outer_loss_source="AlignmentLossSpec",
                outer_loss_kind=adapter.name,
                inner_data_term="l2_projection",
                inner_regulariser=inner_regulariser,
                validation_split="interleaved_kfold",
                differentiation_mode=differentiation_mode,
                initialization_policy=initialization_policy,
            ),
            **kwargs,
        )

    def evaluate_fold(self, state: AlignmentState, fold: int) -> jnp.ndarray:
        fold_i = int(fold)
        train_idx = self.folds.train_idx[fold_i]
        train_mask = self.folds.train_mask[fold_i]
        val_idx = self.folds.val_idx[fold_i]
        val_mask = self.folds.val_mask[fold_i]
        train_base = subset_base_geometry(self.base, train_idx)
        volume = self.reconstruct_fold(state, train_idx, train_mask, train_base)
        val_base = subset_base_geometry(self.base, val_idx)
        val_state = state.replace(
            pose=state.pose.replace(params5=state.pose.params5[val_idx])
        )
        effective = apply_alignment_state(val_base, val_state)
        targets = self.projections[val_idx]
        val_adapter = self.val_loss_adapters[fold_i]
        return project_and_score_stack(
            pose_stack=effective.pose_stack,
            grid=self.grid,
            detector=self.detector,
            volume=volume,
            det_grid=effective.det_grid,
            targets=targets,
            loss_adapter=val_adapter,
            views_per_batch=self.views_per_batch,
            projector_unroll=self.projector_unroll,
            checkpoint_projector=self.checkpoint_projector,
            gather_dtype=self.gather_dtype,
            view_mask=val_mask,
            view_indices=jnp.arange(int(targets.shape[0]), dtype=jnp.int32),
        )

    def evaluate(self, state: AlignmentState) -> ObjectiveResult:
        fold_values: list[jnp.ndarray] = []
        for fold in range(self.folds.n_folds):
            fold_values.append(self.evaluate_fold(state, fold))
        value = jnp.sum(jnp.stack(fold_values))
        return ObjectiveResult(
            value=value,
            aux={
                "objective_kind": self.kind,
                "loss_kind": self.provenance.outer_loss_kind,
                "objective_provenance": self.provenance.to_dict(),
                "folds": self.fold_metadata,
                "fold_eval_mode": "python_loop",
                "views_per_batch": int(self.views_per_batch),
            },
        )

    def value_for_active_z(
        self,
        *,
        frozen_state: AlignmentState,
        view: ActiveParameterView,
        z: jnp.ndarray,
    ) -> jnp.ndarray:
        state = view.unpack(frozen_state, z)
        total = jnp.asarray(0.0, dtype=jnp.float32)
        for fold in range(self.folds.n_folds):
            total = total + self.evaluate_fold(state, fold)
        return total

    def value_and_grad_for_active_z(
        self,
        *,
        frozen_state: AlignmentState,
        view: ActiveParameterView,
        z: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        total_value = jnp.asarray(0.0, dtype=jnp.float32)
        total_grad = jnp.zeros_like(z)
        for fold in range(self.folds.n_folds):

            def fold_objective(z_candidate: jnp.ndarray) -> jnp.ndarray:
                return self.evaluate_fold(view.unpack(frozen_state, z_candidate), fold)

            value, grad = jax.value_and_grad(fold_objective)(z)
            total_value = total_value + value
            total_grad = total_grad + grad
        return total_value, total_grad

    def finite_difference_value_and_grad_for_active_z(
        self,
        *,
        frozen_state: AlignmentState,
        view: ActiveParameterView,
        z: jnp.ndarray,
        eps: float = 1e-2,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        value = self.value_for_active_z(frozen_state=frozen_state, view=view, z=z)
        z_flat = jnp.asarray(z, dtype=jnp.float32).reshape(-1)
        eps_arr = jnp.asarray(float(eps), dtype=jnp.float32)
        grad_values: list[jnp.ndarray] = []
        for idx in range(int(z_flat.size)):
            basis = jnp.zeros_like(z_flat).at[idx].set(eps_arr).reshape(z.shape)
            plus = self.value_for_active_z(
                frozen_state=frozen_state,
                view=view,
                z=z + basis,
            )
            minus = self.value_for_active_z(
                frozen_state=frozen_state,
                view=view,
                z=z - basis,
            )
            grad_values.append((plus - minus) / (jnp.float32(2.0) * eps_arr))
        grad = jnp.stack(grad_values).reshape(z.shape) if grad_values else jnp.zeros_like(z)
        return value, grad


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
