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
        pred = project_stack(
            pose_stack=effective.pose_stack,
            grid=self.grid,
            detector=self.detector,
            volume=self.volume,
            det_grid=effective.det_grid,
            views_per_batch=self.views_per_batch,
            projector_unroll=self.projector_unroll,
            checkpoint_projector=self.checkpoint_projector,
            gather_dtype=self.gather_dtype,
        )
        value = score_projection_stack(
            self.loss_adapter,
            pred,
            self.projections,
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

    def evaluate(self, state: AlignmentState) -> ObjectiveResult:
        fold_values: list[jnp.ndarray] = []
        for fold in range(self.folds.n_folds):
            train_idx = self.folds.train_idx[fold]
            train_mask = self.folds.train_mask[fold]
            val_idx = self.folds.val_idx[fold]
            val_mask = self.folds.val_mask[fold]
            train_base = subset_base_geometry(self.base, train_idx)
            volume = self.reconstruct_fold(state, train_idx, train_mask, train_base)
            val_base = subset_base_geometry(self.base, val_idx)
            val_state = state.replace(
                pose=state.pose.replace(params5=state.pose.params5[val_idx])
            )
            effective = apply_alignment_state(val_base, val_state)
            pred = project_stack(
                pose_stack=effective.pose_stack,
                grid=self.grid,
                detector=self.detector,
                volume=volume,
                det_grid=effective.det_grid,
                views_per_batch=self.views_per_batch,
                projector_unroll=self.projector_unroll,
                checkpoint_projector=self.checkpoint_projector,
                gather_dtype=self.gather_dtype,
            )
            targets = self.projections[val_idx]
            val_adapter = self.val_loss_adapters[fold]
            local_indices = jnp.arange(int(targets.shape[0]), dtype=jnp.int32)
            loss = score_projection_stack(
                val_adapter,
                pred,
                targets,
                mask=val_mask,
                view_indices=local_indices,
            )
            fold_values.append(loss)
        value = jnp.sum(jnp.stack(fold_values))
        return ObjectiveResult(
            value=value,
            aux={
                "objective_kind": self.kind,
                "loss_kind": self.provenance.outer_loss_kind,
                "objective_provenance": self.provenance.to_dict(),
                "folds": self.fold_metadata,
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
    del views_per_batch
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
    )(pose_stack)


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
