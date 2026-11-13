"""The innovation objective and its positive quadratic majorizer agree."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.alignment._joint._pose import MotionPrior


@pytest.mark.parametrize("dofs", [2, 5, 6])
def test_innovation_gradient_and_center_curvature_match_the_coupled_normal(dofs):
    views, delta, strength = 17, 0.02, 0.4
    rng = np.random.default_rng(71)
    gauge = rng.normal(size=(views * dofs, 6))
    with jax.enable_x64(True):
        prior = MotionPrior(views, gauge, knot_spacing=(5,) * dofs, delta=delta)
        pose = jnp.asarray(rng.normal(scale=0.1, size=views * dofs))
        gradient = jax.grad(lambda p: prior.value(p, strength))
        expected = prior.apply(pose, prior.weights(pose, strength))
        np.testing.assert_allclose(gradient(pose), expected, rtol=2e-12, atol=2e-12)

        zero = jnp.zeros_like(pose)
        assert float(prior.value(zero, strength)) == 0
        np.testing.assert_array_equal(gradient(zero), zero)
        curvature = jax.jvp(gradient, (zero,), (pose,))[1]
        expected = prior.apply(pose, jnp.full((views - 1, dofs), strength / delta))
        np.testing.assert_allclose(curvature, expected, rtol=2e-12, atol=2e-12)


def test_irls_quadratic_is_a_tangent_upper_bound_for_arbitrary_pose_changes():
    views, dofs, delta, strength = 21, 5, 0.02, 0.6
    rng = np.random.default_rng(72)
    gauge = rng.normal(size=(views * dofs, 6))
    with jax.enable_x64(True):
        prior = MotionPrior(views, gauge, knot_spacing=(6,) * dofs, delta=delta)
        u = np.asarray(prior.U)
        q = np.eye(views * dofs) - u @ u.T
        difference = np.kron(np.diff(np.eye(views), axis=0), np.eye(dofs)) @ q
        anchor = rng.normal(scale=0.05, size=views * dofs)
        squared_anchor = np.sum((difference @ anchor).reshape(-1, dofs) ** 2, axis=1)
        # Without parameters every column is a translation: one robust weight per link.
        weights = np.asarray(prior.weights(jnp.asarray(anchor), strength))[:, 0]
        assert np.all(weights > 0)
        anchor_value = float(prior.value(jnp.asarray(anchor), strength))
        for scale in (0.0, 0.001, 0.05, 0.5, 5.0):
            trial = rng.normal(scale=scale, size=views * dofs)
            squared_trial = np.sum((difference @ trial).reshape(-1, dofs) ** 2, axis=1)
            upper_bound = anchor_value + 0.5 * np.dot(weights, squared_trial - squared_anchor)
            assert float(prior.value(jnp.asarray(trial), strength)) <= upper_bound + 1e-11
        np.testing.assert_allclose(
            anchor_value,
            0.5 * strength * delta * np.log1p(squared_anchor / delta**2).sum(),
            rtol=2e-12,
        )


def test_resolved_jumps_have_decreasing_force_and_coordinate_invariant_cost():
    views, dofs = 23, 5
    rng = np.random.default_rng(73)
    gauge = rng.normal(size=(views * dofs, 6))
    with jax.enable_x64(True):
        prior = MotionPrior(views, gauge, knot_spacing=(6,) * dofs, delta=0.02)
        pose = jnp.asarray(rng.normal(size=(views, dofs)))
        gradient = jax.grad(lambda p: prior.value(p, 0.5))
        near_force = np.linalg.norm(gradient(pose.ravel()))
        far_force = np.linalg.norm(gradient(10 * pose.ravel()))
        assert far_force < 0.11 * near_force

        rotation, _ = np.linalg.qr(rng.normal(size=(dofs, dofs)))
        changed = (pose @ rotation).ravel()
        np.testing.assert_allclose(
            prior.value(pose.ravel(), 0.5), prior.value(changed, 0.5), rtol=2e-12
        )
        np.testing.assert_allclose(
            prior.weights(pose.ravel(), 0.5), prior.weights(changed, 0.5), rtol=2e-12
        )


def test_a_cubic_drift_is_free_and_a_step_is_not():
    # A cubic in the view index, defined without the prior's spline basis, lies in
    # its cubic drift space: zero innovation cost and zero force. A step does not.
    views, dofs, delta, strength = 31, 5, 0.02, 0.6
    t = np.linspace(-1, 1, views)
    rng = np.random.default_rng(74)
    coefficients = rng.normal(scale=0.5, size=(4, dofs))
    drift = np.stack([t**k for k in range(4)], axis=1) @ coefficients
    with jax.enable_x64(True):
        prior = MotionPrior(
            views, np.zeros((views * dofs, 0)), knot_spacing=(6,) * dofs, delta=delta
        )
        smooth = jnp.asarray(drift.reshape(-1))
        assert abs(float(prior.value(smooth, strength))) < 1e-18
        np.testing.assert_allclose(
            jax.grad(lambda p: prior.value(p, strength))(smooth), 0, atol=1e-12
        )
        step = drift.copy()
        step[views // 2 :, 0] += 0.3
        assert float(prior.value(jnp.asarray(step.reshape(-1)), strength)) > 1e-3
