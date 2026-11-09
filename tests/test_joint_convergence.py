"""Geometry convergence measures raw proposals, independently of line-search clipping."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.alignment._joint import _solver  # check-public-imports: allow-private


@pytest.mark.parametrize(("a", "beta"), [(1.0, 0.0), (0.25, 0.5)])
def test_step_records_the_raw_pose_proposal_before_line_search_scaling(monkeypatch, a, beta):
    n = 60
    image, state = jnp.ones(1), jnp.zeros(2 * n)
    level = object.__new__(_solver._Level)
    level.lam = 0
    level.ops = SimpleNamespace(
        prepare=lambda value: value,
        pad_poses=lambda value: value,
        weighted_forward=lambda *_: (jnp.zeros(1), jnp.zeros(2 * n)),
    )
    level.prior = SimpleNamespace(
        weights=lambda *_: None,
        apply=lambda value, _: jnp.zeros_like(value),
        factor=lambda *_: None,
        solve=lambda _, value: value,
        expand=lambda value: value.reshape(n, 2),
    )
    level.rhs = lambda *_: jnp.ones(1)
    level.inverse = lambda *_: jnp.ones(1)
    level.schur = lambda *_: SimpleNamespace(
        solution=jnp.ones(1),
        iterations=1,
        relative_residual=0.0,
        residual_is_explicit=True,
        matvecs=1,
    )
    level.norm = jnp.vdot
    level.energy = lambda value: jnp.vdot(value, value)
    level.trial = lambda x, c, dx, dc, _, step, fraction: (
        99.0,
        x + step * dx,
        c + step * fraction * dc,
    )
    monkeypatch.setattr(_solver, "_line_search", lambda trial, *_: (a, beta, trial(a, beta)))
    # Identity pose normal: back-substitution proposes (-3, -4) at every view.
    linear = (
        -jnp.ones(1),
        jnp.ones(1),
        None,
        None,
        jnp.tile(jnp.array([3.0, 4.0]), n),
        jnp.broadcast_to(jnp.eye(2), (n, 2, 2)),
    )
    result = level.step(image, state, linear, 0.001, 0.0, 100.0)
    assert result is not None
    assert result[4] == pytest.approx(a * beta * 5.0)
    assert result[-1]["proposed_pose_step_rms"] == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("proposals", "expected_steps", "status", "forced_damping"),
    [
        ([1.0] * 8, 8, "budget_exhausted", None),
        ([0.003] * 8, 8, "budget_exhausted", None),
        ([0.001] * 8, 4, "converged", None),
        ([1.0, 1.0, 1.0, 0.001, 0.001, 1.0], 5, "converged", None),
        ([0.001, 0.001, 1.0, 0.001, 1.0, 0.001, 0.001, 1.0], 7, "converged", None),
        ([0.001] * 2, 2, "budget_exhausted", None),
        pytest.param(
            [0.001] * 8,
            8,
            "budget_exhausted",
            1.0,
            id="heavy_damping_does_not_converge",
        ),
    ],
)
def test_stopping_requires_two_small_raw_proposals_after_the_minimum_iterations(
    monkeypatch, proposals, expected_steps, status, forced_damping
):
    remaining = iter(proposals)

    class Level:
        def __init__(self, ops, *_args, **_kwargs):
            self.ops, self.y, self.gain = ops, np.zeros((60, 1, 1)), np.ones(60)

        def linearize(self, _state, _image):
            return self.ops.linearize()

        def refit(self, state, image, _sums):
            return False, self.linearize(state, image)

        def image_penalty(self, _image):
            return 0.0

        def step(self, image, state, _linear, lm, _weight, objective):
            # Image improvement stays large, while the line search holds all poses.
            # Neither quantity determines whether the geometry has converged.
            return (
                1.0,
                0.0,
                (objective / 2, image, state),
                objective / 2,
                0.0,
                lm if forced_damping is None else forced_damping,
                1,
                0.0,
                {"proposed_pose_step_rms": next(remaining)},
            )

    ops = SimpleNamespace(
        n=60,
        pad_images=lambda value: value,
        pad_precision=lambda value: value,
        pad_poses=lambda value: value,
        sensitivity=lambda _: (1.0, np.ones(1, bool)),
        linearize=lambda *_: (
            jnp.zeros(1),
            jnp.ones(1),
            None,
            None,
            jnp.zeros(60),
            jnp.ones((60, 1, 1)),
            100.0,
            (60, 0),
            # Normal equations whose solution is gain one and no background.
            np.concatenate(
                [np.tile(np.eye(3), (60, 1, 1)), np.tile([[1.0], [0], [0]], (60, 1, 1))], -1
            ),
        ),
    )
    prior = SimpleNamespace(size=60, value=lambda *_: 0.0)
    monkeypatch.setattr(_solver, "_Level", Level)
    with jax.enable_x64(True):
        _, fitted, _, record = _solver.solve(
            np.zeros((60, 1, 1)),
            None,
            prior,
            0.0,
            0.0,
            ops=ops,
            budget=_solver.Budget(outer=len(proposals), inner=1),
            x0=np.ones(1),
            p0=None,
            pose_sigma=0.01,
            delta=1,
            scales=(1, 1, 1),
        )
    assert record["status"] == status
    assert len(record["steps"]) == expected_steps
    assert record["final_objective"] == pytest.approx(100.0 / 2**expected_steps)
    np.testing.assert_array_equal(fitted, 0)
    assert all(step["pose_step_rms"] == 0 for step in record["steps"])
    if forced_damping is not None:
        assert all(step["lm"] == forced_damping for step in record["steps"])
    np.testing.assert_array_equal(
        [step["proposed_pose_step_rms"] for step in record["steps"]],
        proposals[:expected_steps],
    )
