from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from tomojax.align.dof_specs import ActiveParameterView
from tomojax.align.geometry_applier import BaseGeometryArrays, apply_alignment_state
from tomojax.align.objectives import (
    BilevelCVProjectionObjective,
    FixedVolumeProjectionObjective,
    FoldSpec,
    objective_value_and_grad,
    project_stack,
)
from tomojax.align.state import AlignmentState, PoseState, SetupGeometryState
from tomojax.align.losses import L2OtsuLossSpec
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


def test_bilevel_cv_objective_uses_validation_loss_without_candidate_enumeration():
    grid, detector, geometry, true_volume, projections = _case(hidden_det_u_px=2.0)
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    folds = FoldSpec(n_folds=4).build(projections.shape[0])

    def true_reconstruction(state, train_idx, train_mask, train_base):
        del state, train_idx, train_mask, train_base
        return true_volume

    objective = BilevelCVProjectionObjective.from_loss_spec(
        base=base,
        grid=grid,
        detector=detector,
        projections=projections,
        loss_spec=L2OtsuLossSpec(),
        folds=folds,
        reconstruct_fold=true_reconstruction,
        checkpoint_projector=False,
    )
    nominal = AlignmentState(setup=SetupGeometryState(), pose=PoseState.zeros(projections.shape[0]))
    corrected = nominal.replace(setup=nominal.setup.replace(det_u_px=jnp.asarray(2.0)))

    nominal_value = objective.evaluate(nominal).value
    corrected_value = objective.evaluate(corrected).value

    assert float(corrected_value) < float(nominal_value)
    aux = objective.evaluate(corrected).aux
    assert aux["objective_kind"] == "bilevel_cv"
    assert aux["objective_provenance"]["outer_loss_kind"] == "l2_otsu"
    assert aux["objective_provenance"]["validation_split"] == "interleaved_kfold"
    assert aux["fold_eval_mode"] == "python_loop"
    assert aux["views_per_batch"] == 1


def test_bilevel_fold_safe_value_and_grad_matches_all_fold_reference():
    grid, detector, geometry, true_volume, projections = _case(n_views=6, hidden_det_u_px=2.0)
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    folds = FoldSpec(n_folds=3).build(projections.shape[0])

    def true_reconstruction(state, train_idx, train_mask, train_base):
        del state, train_idx, train_mask, train_base
        return true_volume

    objective = BilevelCVProjectionObjective.from_loss_spec(
        base=base,
        grid=grid,
        detector=detector,
        projections=projections,
        loss_spec=L2OtsuLossSpec(),
        folds=folds,
        reconstruct_fold=true_reconstruction,
        checkpoint_projector=False,
        views_per_batch=1,
    )
    state = AlignmentState(setup=SetupGeometryState(), pose=PoseState.zeros(projections.shape[0]))
    view = ActiveParameterView.from_dofs(("det_u_px",))
    z = view.pack(state)

    seq_value, seq_grad = objective.value_and_grad_for_active_z(
        frozen_state=state,
        view=view,
        z=z,
    )
    ref_value, ref_grad = jax.value_and_grad(
        lambda zz: objective.evaluate(view.unpack(state, zz)).value
    )(z)

    np.testing.assert_allclose(np.asarray(seq_value), np.asarray(ref_value), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(seq_grad), np.asarray(ref_grad), atol=1e-5, rtol=1e-5)


def test_bilevel_finite_difference_value_and_grad_is_finite():
    grid, detector, geometry, true_volume, projections = _case(n_views=6, hidden_det_u_px=2.0)
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    folds = FoldSpec(n_folds=3).build(projections.shape[0])

    def true_reconstruction(state, train_idx, train_mask, train_base):
        del state, train_idx, train_mask, train_base
        return true_volume

    objective = BilevelCVProjectionObjective.from_loss_spec(
        base=base,
        grid=grid,
        detector=detector,
        projections=projections,
        loss_spec=L2OtsuLossSpec(),
        folds=folds,
        reconstruct_fold=true_reconstruction,
        checkpoint_projector=False,
        views_per_batch=1,
    )
    state = AlignmentState(setup=SetupGeometryState(), pose=PoseState.zeros(projections.shape[0]))
    view = ActiveParameterView.from_dofs(("det_u_px",))

    value, grad = objective.finite_difference_value_and_grad_for_active_z(
        frozen_state=state,
        view=view,
        z=view.pack(state),
        eps=1e-2,
    )

    assert jnp.isfinite(value)
    assert grad.shape == (1,)
    assert jnp.isfinite(grad[0])


def test_objective_value_and_grad_operates_on_active_whitened_vector():
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
    view = ActiveParameterView.from_dofs(("det_u_px",))
    value_and_grad = objective_value_and_grad(objective, view, state)

    (value, aux), grad = value_and_grad(view.pack(state))

    assert jnp.isfinite(value)
    assert grad.shape == (1,)
    assert aux["objective_kind"] == "fixed_volume"
