from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

from tomojax.align.model.dof_specs import ActiveParameterView
from tomojax.align.objectives.folds import FoldSpec
from tomojax.align.geometry.geometry_applier import BaseGeometryArrays, apply_alignment_state
from tomojax.align.objectives.fixed_volume import (
    FixedVolumeProjectionObjective,
    project_stack,
)
from tomojax.align.model.state import AlignmentState, PoseState, SetupGeometryState
from tomojax.align.objectives.losses import L2OtsuLossSpec
from tomojax.align.objectives.losses import build_loss_adapter
from tomojax.align.objectives.validation_residuals import (
    accumulate_validation_normals,
    score_validation_fixed_volume,
)
from tomojax.core.geometry import Detector, Grid, ParallelGeometry
from tomojax.core.projector import forward_project_view


def _case(size: int = 8, n_views: int = 8, hidden_det_u_px: float = 2.0):
    grid = Grid(nx=size, ny=size, nz=size, vx=1.0, vy=1.0, vz=1.0)
    det_nom = Detector(nu=size, nv=size, du=1.0, dv=1.0)
    det_true = Detector(nu=size, nv=size, du=1.0, dv=1.0, det_center=(hidden_det_u_px, 0.0))
    thetas = np.linspace(0.0, 180.0, n_views, endpoint=False, dtype=np.float32)
    geom_nom = ParallelGeometry(grid=grid, detector=det_nom, thetas_deg=thetas)
    geom_true = ParallelGeometry(grid=grid, detector=det_true, thetas_deg=thetas)
    volume_np = np.zeros((size, size, size), dtype=np.float32)
    volume_np[1:5, 2:6, 2:5] = 1.0
    volume_np[5:7, 1:4, 5:7] = 0.7
    volume = jnp.asarray(volume_np)
    projections = jnp.stack(
        [
            forward_project_view(
                geom_true,
                grid,
                det_true,
                volume,
                i,
                gather_dtype="fp32",
            )
            for i in range(n_views)
        ],
        axis=0,
    )
    return grid, det_nom, geom_nom, volume, projections


def test_fold_spec_builds_static_padded_interleaved_arrays():
    folds = FoldSpec(n_folds=4).build(10)

    assert folds.train_idx.shape == (4, 8)
    assert folds.val_idx.shape == (4, 3)
    assert [int(v) for v in jnp.sum(folds.val_mask, axis=1)] == [3, 3, 2, 2]
    assert tuple(np.asarray(folds.val_idx[0, :3])) == (0, 4, 8)
    assert folds.to_metadata()["n_folds"] == 4


def test_fold_spec_rejects_too_few_views():
    with pytest.raises(ValueError, match="requires at least"):
        FoldSpec(n_folds=4).build(3)


def test_fixed_volume_objective_scores_with_existing_l2_otsu_loss_adapter():
    grid, detector, geometry, volume, projections = _case(hidden_det_u_px=0.0)
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    state = AlignmentState(setup=SetupGeometryState(), pose=PoseState.zeros(projections.shape[0]))
    objective = FixedVolumeProjectionObjective.from_loss_spec(
        base=base,
        grid=grid,
        detector=detector,
        projections=projections,
        volume=volume,
        loss_spec=L2OtsuLossSpec(),
        checkpoint_projector=False,
    )

    result = objective.evaluate(state)

    assert float(result.value) == pytest.approx(0.0, abs=1e-5)
    assert result.aux["objective_kind"] == "fixed_volume"
    assert result.aux["loss_kind"] == "l2_otsu"


def test_project_stack_honors_views_per_batch_without_changing_values():
    grid, detector, geometry, volume, _projections = _case(n_views=7, hidden_det_u_px=0.0)
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    state = AlignmentState(setup=SetupGeometryState(), pose=PoseState.zeros(7))
    effective = apply_alignment_state(base, state)

    streamed = project_stack(
        pose_stack=effective.pose_stack,
        grid=grid,
        detector=detector,
        volume=volume,
        det_grid=effective.det_grid,
        views_per_batch=1,
        checkpoint_projector=False,
    )
    batched = project_stack(
        pose_stack=effective.pose_stack,
        grid=grid,
        detector=detector,
        volume=volume,
        det_grid=effective.det_grid,
        views_per_batch=7,
        checkpoint_projector=False,
    )

    assert streamed.shape == batched.shape == (7, detector.nv, detector.nu)
    np.testing.assert_allclose(np.asarray(streamed), np.asarray(batched), atol=1e-5, rtol=1e-5)


def test_stopped_validation_score_prefers_hidden_offset_without_candidate_enumeration():
    grid, detector, geometry, true_volume, projections = _case(hidden_det_u_px=2.0)
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    folds = FoldSpec(n_folds=4).build(projections.shape[0])
    adapter = build_loss_adapter(L2OtsuLossSpec(), projections)
    nominal = AlignmentState(setup=SetupGeometryState(), pose=PoseState.zeros(projections.shape[0]))
    corrected = nominal.replace(setup=nominal.setup.replace(det_u_px=jnp.asarray(2.0)))
    view = ActiveParameterView.from_dofs(("det_u_px",))

    nominal_value = score_validation_fixed_volume(
        frozen_state=nominal,
        active_view=view,
        z=view.pack(nominal),
        base=base,
        grid=grid,
        detector=detector,
        projections=projections,
        loss_adapter=adapter,
        fold_volume=true_volume,
        val_idx=folds.val_idx[0],
        val_mask=folds.val_mask[0],
        views_per_batch=1,
        projector_unroll=1,
        checkpoint_projector=False,
        gather_dtype="fp32",
    )
    corrected_value = score_validation_fixed_volume(
        frozen_state=nominal,
        active_view=view,
        z=view.pack(corrected),
        base=base,
        grid=grid,
        detector=detector,
        projections=projections,
        loss_adapter=adapter,
        fold_volume=true_volume,
        val_idx=folds.val_idx[0],
        val_mask=folds.val_mask[0],
        views_per_batch=1,
        projector_unroll=1,
        checkpoint_projector=False,
        gather_dtype="fp32",
    )

    assert float(corrected_value) < float(nominal_value)


def test_validation_normals_match_fixed_volume_finite_difference():
    grid, detector, geometry, volume, projections = _case(n_views=6, hidden_det_u_px=1.0)
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    folds = FoldSpec(n_folds=2).build(projections.shape[0])
    state = AlignmentState(setup=SetupGeometryState(), pose=PoseState.zeros(projections.shape[0]))
    view = ActiveParameterView.from_dofs(("det_u_px",))
    adapter = build_loss_adapter(L2OtsuLossSpec(), projections)
    # Avoid testing exactly at the nominal detector-grid origin: the ray sampler
    # has legitimate piecewise-linear kinks there, so finite differences measure
    # a symmetric secant rather than the implementation's local JVP.
    z = view.pack(state) + jnp.asarray([0.23], dtype=jnp.float32)
    val_idx = folds.val_idx[0]
    val_mask = folds.val_mask[0]

    normals = accumulate_validation_normals(
        frozen_state=state,
        active_view=view,
        z=z,
        base=base,
        grid=grid,
        detector=detector,
        projections=projections,
        loss_adapter=adapter,
        fold_volume=volume,
        val_idx=val_idx,
        val_mask=val_mask,
        views_per_batch=1,
        projector_unroll=1,
        checkpoint_projector=False,
        gather_dtype="fp32",
    )

    eps = jnp.asarray([1e-2], dtype=jnp.float32)
    plus = score_validation_fixed_volume(
        frozen_state=state,
        active_view=view,
        z=z + eps,
        base=base,
        grid=grid,
        detector=detector,
        projections=projections,
        loss_adapter=adapter,
        fold_volume=volume,
        val_idx=val_idx,
        val_mask=val_mask,
        views_per_batch=1,
        projector_unroll=1,
        checkpoint_projector=False,
        gather_dtype="fp32",
    )
    minus = score_validation_fixed_volume(
        frozen_state=state,
        active_view=view,
        z=z - eps,
        base=base,
        grid=grid,
        detector=detector,
        projections=projections,
        loss_adapter=adapter,
        fold_volume=volume,
        val_idx=val_idx,
        val_mask=val_mask,
        views_per_batch=1,
        projector_unroll=1,
        checkpoint_projector=False,
        gather_dtype="fp32",
    )
    fd_grad = (plus - minus) / (2.0 * eps[0])

    assert jnp.isfinite(normals.loss)
    assert normals.grad.shape == (1,)
    assert normals.hess.shape == (1, 1)
    np.testing.assert_allclose(
        np.asarray(normals.grad[0]),
        np.asarray(fd_grad),
        rtol=2e-2,
        atol=2e-2,
    )
