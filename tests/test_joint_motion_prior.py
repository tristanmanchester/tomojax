"""The innovation objective and its positive quadratic majorizer agree."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.alignment._joint._pose import MotionPrior, ViewOrder


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
        # The tail is set so that its largest force is the weight: curvature 2 w / delta at zero.
        expected = prior.apply(pose, jnp.full((views - 1, dofs), 2 * strength / delta))
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
            strength * delta * np.log1p(squared_anchor / delta**2).sum(),
            rtol=2e-12,
        )


def test_the_largest_restoring_force_on_a_jump_is_the_weight():
    # The weight is derived as a threshold: the force the prior can exert against a view's
    # own error. The force on a link of size J is its IRLS weight times J; scaling a pose
    # scales every J, so one link can be put at any size, and the force on it peaks at
    # J = delta, where it equals the weight.
    views, dofs, delta, strength = 25, 2, 0.02, 0.7
    rng = np.random.default_rng(75)
    with jax.enable_x64(True):
        prior = MotionPrior(
            views, np.zeros((views * dofs, 0)), knot_spacing=(6,) * dofs, delta=delta
        )
        pose = jnp.asarray(rng.normal(size=views * dofs))
        jumps = np.sqrt(np.asarray(prior._squared_jumps(prior.matrices, pose)))
        link = int(np.argmax(jumps))
        forces = []
        for size in np.geomspace(0.01, 100, 81) * delta:
            scaled = pose * (size / jumps[link])
            weight = float(np.asarray(prior.weights(scaled, strength))[link, 0])
            forces.append(weight * size)
    assert max(forces) == pytest.approx(strength, rel=1e-12)
    assert np.argmax(forces) == 40  # J = delta, the middle of the geometric sweep
    assert forces[0] < 0.03 * strength and forces[-1] < 0.03 * strength


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


@pytest.mark.parametrize("views", [80, 257])
def test_projection_and_solve_use_only_drift_observed_at_the_views(views):
    from scipy.interpolate import make_interp_spline
    from scipy.linalg import block_diag, orth

    # At 80 views the gap leaves three zero spline columns; reduced QR still
    # returns one direction for each. The longer case also restarts at a repeat.
    positions = np.arange(views, dtype=float)
    positions[40:] += 60
    if views > 80:
        positions[180:] -= 1
        positions[190::3] += 0.25
    order = ViewOrder(views, positions)
    spacing = 10 if views == 80 else 3
    design = []
    for start, stop in order.segments:
        t = np.r_[0, np.cumsum(order.steps[start : stop - 1])]
        sites = np.r_[np.arange(0, t[-1], spacing), t[-1]]
        design.append(make_interp_spline(sites, np.eye(len(sites)), k=min(3, len(sites) - 1))(t))
    basis = orth(block_diag(*design))
    if views == 80:
        assert basis.shape[1] == 12
    q = np.eye(views) - basis @ basis.T
    rng = np.random.default_rng(42)
    rhs = rng.normal(size=views)
    blocks = rng.uniform(0.5, 1.5, views)
    w = order.edge_scale
    d = np.diff(np.eye(views), axis=0)
    lap = d.T @ (w[:, None] * d)
    with jax.enable_x64(True):
        prior = MotionPrior(
            views, np.zeros((views, 0)), knot_spacing=(spacing,), delta=0.2, order=order
        )
        expected = np.linalg.solve(np.diag(blocks) + q @ lap @ q, rhs)
        f = prior.factor(jnp.asarray(blocks[:, None, None]), jnp.asarray(w[:, None]))
        np.testing.assert_allclose(
            prior.solve(f, jnp.asarray(rhs)), expected, atol=2e-12, rtol=2e-12
        )
        jumps = (d @ q @ rhs) * order.jump_scale
        np.testing.assert_allclose(
            prior.value(rhs, 0.4), 0.08 * np.log1p((jumps / 0.2) ** 2).sum(), rtol=2e-12
        )
        system = np.block(
            [[np.diag(blocks) + lap, basis], [basis.T, np.zeros((basis.shape[1],) * 2)]]
        )
        expected = np.linalg.solve(system, np.r_[rhs, np.zeros(basis.shape[1])])[:views]
        np.testing.assert_allclose(
            prior.innovation_solve(blocks[:, None, None], w[:, None], rhs),
            expected,
            atol=2e-12,
            rtol=2e-12,
        )


@pytest.mark.parametrize("views", [1024, 1025])
def test_long_chain_with_only_an_affine_drift_recovers_the_prescribed_solution(views):
    rng = np.random.default_rng(51)
    x = rng.normal(size=(views, 2))
    b = np.column_stack([np.ones(views), np.linspace(-1, 1, views)])

    def residual(values):
        return values - b @ np.linalg.lstsq(b, values, rcond=None)[0]

    f = rng.normal(size=(views, 2, 2))
    f = f @ np.swapaxes(f, 1, 2) + np.eye(2)
    w = rng.uniform(0.3, 4, size=(views - 1, 2))
    differences = w * np.diff(residual(x), axis=0)
    lap = np.pad(differences, ((1, 0), (0, 0))) - np.pad(differences, ((0, 1), (0, 0)))
    rhs = (f @ x[..., None])[..., 0] + residual(lap)
    with jax.enable_x64(True):
        prior = MotionPrior(views, np.zeros((views * 2, 0)), knot_spacing=(views, views), delta=0.1)
        got = prior.solve(prior.factor(f, w), rhs.ravel()).reshape(x.shape)
        np.testing.assert_allclose(got, x, rtol=2e-12, atol=2e-12)
