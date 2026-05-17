from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.align.model.dof_specs import ActiveParameterView
from tomojax.align.model.motion_models import (
    PoseMotionModel,
    expand_motion_coefficients,
    fit_motion_coefficients,
)
from tomojax.align.model.state import AlignmentState, PoseState, SetupGeometryState
from tomojax.align.optimizers import (
    ActiveLbfgsConfig,
    BoundTransform,
    PoseLbfgsConfig,
    PoseOptimizationContext,
    ValidationLmConfig,
    run_active_lbfgs,
    run_active_validation_lm,
    run_pose_lbfgs,
)


def _lbfgs_config(**kwargs) -> PoseLbfgsConfig:
    values = {
        "maxiter": 1,
        "ftol": 0.0,
        "gtol": 0.0,
        "maxls": 5,
        "memory_size": 3,
    }
    values.update(kwargs)
    return PoseLbfgsConfig(**values)


def _pose_context(
    *,
    frozen_params5: jnp.ndarray,
    active_cols: np.ndarray,
    bounds_lower: jnp.ndarray,
    bounds_upper: jnp.ndarray,
    apply_param_constraints,
    motion_model: PoseMotionModel | None = None,
) -> PoseOptimizationContext:
    return PoseOptimizationContext(
        active_cols=active_cols,
        frozen_params5=frozen_params5,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        apply_param_constraints=apply_param_constraints,
        motion_model=motion_model,
    )


def test_bound_transform_round_trips_mixed_bounds():
    lower = jnp.asarray([-jnp.inf, 0.0, -jnp.inf, -2.0, 3.0], dtype=jnp.float32)
    upper = jnp.asarray([jnp.inf, jnp.inf, 1.0, 2.0, 3.0], dtype=jnp.float32)
    values = jnp.asarray([1.25, 0.7, 0.3, 0.5, 3.0], dtype=jnp.float32)
    transform = BoundTransform.from_bounds(lower, upper, value_shape=(5,))

    z = transform.to_unconstrained(values)
    recovered = transform.from_unconstrained(z)

    assert bool(jnp.all(jnp.isfinite(z)))
    assert bool(jnp.all(jnp.isfinite(recovered)))
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(values), rtol=1e-5, atol=2e-5)


def test_bound_transform_outputs_stay_finite_and_bounded():
    lower = jnp.asarray([-jnp.inf, -1.0, 0.0, -2.0, 4.0], dtype=jnp.float32)
    upper = jnp.asarray([jnp.inf, 1.0, jnp.inf, 2.0, 4.0], dtype=jnp.float32)
    transform = BoundTransform.from_bounds(lower, upper, value_shape=(5,))
    values = transform.from_unconstrained(
        jnp.asarray([-100.0, -50.0, 0.0, 50.0, 100.0], dtype=jnp.float32)
    )

    assert bool(jnp.all(jnp.isfinite(values)))
    assert values[1] >= -1.0
    assert values[1] <= 1.0
    assert values[2] >= 0.0
    assert values[3] >= -2.0
    assert values[3] <= 2.0
    assert values[4] == pytest.approx(4.0)


def test_bound_transform_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="bounds size"):
        BoundTransform.from_bounds(
            jnp.zeros((2,), dtype=jnp.float32),
            jnp.zeros((2,), dtype=jnp.float32),
            value_shape=(3,),
        )


def test_pose_lbfgs_active_packing_excludes_frozen_dofs():
    frozen = jnp.asarray(
        [
            [0.11, -0.22, 0.33, 0.0, 0.0],
            [0.44, -0.55, 0.66, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    active_cols = np.asarray([3, 4], dtype=np.int32)
    bounds_lower = jnp.asarray([-jnp.inf, -jnp.inf, -jnp.inf, -1.0, -1.0], dtype=jnp.float32)
    bounds_upper = jnp.asarray([jnp.inf, jnp.inf, jnp.inf, 1.0, 1.0], dtype=jnp.float32)

    def objective(params5):
        target = jnp.asarray([[0.2, -0.1], [0.1, 0.15]], dtype=jnp.float32)
        return jnp.sum((params5[:, 3:] - target) ** 2)

    result = run_pose_lbfgs(
        params5_in=frozen,
        motion_coeffs_in=None,
        loss_before_value=float(objective(frozen)),
        objective_fn=objective,
        eval_loss_fn=lambda params, _label: float(objective(params)),
        is_expected_failure=lambda _exc: False,
        cfg=_lbfgs_config(maxiter=3),
        context=_pose_context(
            frozen_params5=frozen,
            active_cols=active_cols,
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
            apply_param_constraints=lambda params: jnp.clip(params, bounds_lower, bounds_upper),
        ),
    )

    np.testing.assert_array_equal(np.asarray(result.params5[:, :3]), np.asarray(frozen[:, :3]))
    assert result.stats["lbfgs_backend"] == "optax"
    assert result.stats["optimizer"] == "lbfgs"
    assert result.stats["optimizer_backend"] == "optax"


def test_pose_lbfgs_smooth_unbounded_refines_constraints_like_active_path():
    params0 = jnp.zeros((2, 5), dtype=jnp.float32)
    basis = jnp.asarray([[1.0], [0.0]], dtype=jnp.float32)
    model = PoseMotionModel(
        name="polynomial",
        basis=basis,
        basis_pinv=jnp.linalg.pinv(basis),
        active_indices=(3,),
        active_names=("dx",),
        frozen_params5=params0,
    )
    coeffs0 = jnp.asarray([[4.0]], dtype=jnp.float32)
    bounds_lower = jnp.full((5,), -jnp.inf, dtype=jnp.float32)
    bounds_upper = jnp.full((5,), jnp.inf, dtype=jnp.float32)

    def zero_mean_dx(params5):
        dx = params5[:, 3]
        return params5.at[:, 3].set(dx - jnp.mean(dx))

    expected_params = zero_mean_dx(expand_motion_coefficients(model, coeffs0))
    expected_coeffs = fit_motion_coefficients(model, expected_params)
    expected_params = zero_mean_dx(expand_motion_coefficients(model, expected_coeffs))
    expected_coeffs = fit_motion_coefficients(model, expected_params)
    expected_params = zero_mean_dx(expand_motion_coefficients(model, expected_coeffs))

    result = run_pose_lbfgs(
        params5_in=params0,
        motion_coeffs_in=coeffs0,
        loss_before_value=10.0,
        objective_fn=lambda params: jnp.sum(params[:, 3] ** 2),
        eval_loss_fn=lambda _params, _label: 0.0,
        is_expected_failure=lambda _exc: False,
        cfg=_lbfgs_config(maxiter=0),
        context=_pose_context(
            frozen_params5=params0,
            active_cols=np.asarray([3], dtype=np.int32),
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
            apply_param_constraints=zero_mean_dx,
            motion_model=model,
        ),
    )

    assert result.accepted is True
    np.testing.assert_allclose(
        np.asarray(result.params5),
        np.asarray(expected_params),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(result.motion_coeffs),
        np.asarray(expected_coeffs),
        rtol=1e-6,
        atol=1e-6,
    )


def test_pose_lbfgs_selects_best_candidate_when_last_is_worse():
    params0 = jnp.zeros((1, 5), dtype=jnp.float32)
    bounds_lower = jnp.full((5,), -jnp.inf, dtype=jnp.float32)
    bounds_upper = jnp.full((5,), jnp.inf, dtype=jnp.float32)

    result = run_pose_lbfgs(
        params5_in=params0,
        motion_coeffs_in=None,
        loss_before_value=5.0,
        objective_fn=lambda params: jnp.sum(params[:, 3] ** 2),
        eval_loss_fn=lambda _params, label: 10.0 if label == "last" else 1.0,
        is_expected_failure=lambda _exc: False,
        cfg=_lbfgs_config(),
        context=_pose_context(
            frozen_params5=params0,
            active_cols=np.asarray([3], dtype=np.int32),
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
            apply_param_constraints=lambda params: params,
        ),
    )

    assert result.accepted is True
    assert result.stats["lbfgs_selected_candidate"] == "best"
    assert result.stats["lbfgs_accepted"] is True
    assert result.stats["optimizer_accepted"] is True
    assert result.loss == pytest.approx(1.0)


def test_pose_lbfgs_rejects_non_improving_candidate():
    params0 = jnp.zeros((1, 5), dtype=jnp.float32)
    bounds_lower = jnp.full((5,), -jnp.inf, dtype=jnp.float32)
    bounds_upper = jnp.full((5,), jnp.inf, dtype=jnp.float32)

    result = run_pose_lbfgs(
        params5_in=params0,
        motion_coeffs_in=None,
        loss_before_value=0.5,
        objective_fn=lambda params: jnp.sum(params[:, 3] ** 2),
        eval_loss_fn=lambda _params, _label: 1.0,
        is_expected_failure=lambda _exc: False,
        cfg=_lbfgs_config(),
        context=_pose_context(
            frozen_params5=params0,
            active_cols=np.asarray([3], dtype=np.int32),
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
            apply_param_constraints=lambda params: params,
        ),
    )

    assert result.accepted is False
    assert result.stats["lbfgs_selected_candidate"] == "rejected"
    assert result.stats["lbfgs_accepted"] is False
    assert result.stats["optimizer_accepted"] is False
    assert result.loss == pytest.approx(0.5)
    np.testing.assert_array_equal(np.asarray(result.params5), np.asarray(params0))


def test_active_lbfgs_updates_setup_state_without_pose_special_case():
    state = AlignmentState(
        setup=SetupGeometryState(det_u_px=jnp.asarray(0.0)), pose=PoseState.zeros(1)
    )
    view = ActiveParameterView.from_dofs(("det_u_px",))

    def objective(candidate):
        return (candidate.setup.det_u_px - 3.0) ** 2

    result = run_active_lbfgs(
        state=state,
        view=view,
        objective_fn=objective,
        cfg=ActiveLbfgsConfig(maxiter=6, ftol=0.0, gtol=1e-6, maxls=10),
    )

    assert result.accepted is True
    assert abs(float(result.state.setup.det_u_px) - 3.0) < 1e-3
    assert result.stats["step_norm_whitened"] > 0.0


def test_active_validation_lm_updates_setup_state_from_normal_equation():
    state = AlignmentState(
        setup=SetupGeometryState(det_u_px=jnp.asarray(0.0)), pose=PoseState.zeros(1)
    )
    view = ActiveParameterView.from_dofs(("det_u_px",))
    grad = jnp.asarray([-4.0], dtype=jnp.float32)
    hess = jnp.asarray([[2.0]], dtype=jnp.float32)

    def score(z):
        return float((z[0] - 2.0) ** 2)

    result = run_active_validation_lm(
        state=state,
        view=view,
        loss=4.0,
        grad=grad,
        hess=hess,
        score_fn=score,
        cfg=ValidationLmConfig(damping=1e-6),
    )

    assert result.accepted is True
    assert result.stats["optimizer"] == "validation_lm"
    assert result.stats["optimizer_backend"] == "streamed_normals"
    assert abs(float(result.state.setup.det_u_px) - 2.0) < 1e-4
    assert "det_u_px" in result.stats["step_by_dof_native_units"]


def test_active_lbfgs_updates_pose_state_without_setup_special_case():
    params = jnp.zeros((2, 5), dtype=jnp.float32)
    state = AlignmentState(setup=SetupGeometryState(), pose=PoseState(params))
    view = ActiveParameterView.from_dofs(("dx",))

    def objective(candidate):
        return jnp.sum((candidate.pose.params5[:, 3] - 0.25) ** 2)

    result = run_active_lbfgs(
        state=state,
        view=view,
        objective_fn=objective,
        cfg=ActiveLbfgsConfig(maxiter=6, ftol=0.0, gtol=1e-6, maxls=10),
    )

    assert result.accepted is True
    np.testing.assert_allclose(
        np.asarray(result.state.pose.params5[:, 3]),
        np.asarray([0.25, 0.25], dtype=np.float32),
        atol=1e-3,
    )
