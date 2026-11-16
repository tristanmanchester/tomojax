"""Geometry convergence measures raw proposals, independently of line-search clipping."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.alignment._joint import _common, _solver  # check-public-imports: allow-private


@pytest.mark.parametrize(("a", "beta"), [(1.0, 0.0), (0.25, 0.5)])
def test_step_records_the_raw_pose_proposal_before_line_search_scaling(monkeypatch, a, beta):
    n = 60
    image, state = jnp.ones(1), jnp.zeros(2 * n)
    level = object.__new__(_solver._Level)
    level.lam, level.precision, level.stacks = 0, None, {"r": None}
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
    level.trial = lambda x, c, dx, dc, _, _held, step, fraction: (
        99.0,
        x + step * dx,
        c + step * fraction * dc,
    )
    monkeypatch.setattr(_solver, "_line_search", lambda trial, *_: (a, beta, trial(a, beta)))
    # Identity pose normal: back-substitution proposes (-3, -4) at every view.
    # The views' shared offset and ramp contribute nothing here.
    shared = _common.Common(
        jnp.eye(2), jnp.eye(2), jnp.zeros((2, 1)), jnp.zeros((2, 2 * n)), jnp.zeros(2)
    )
    linear = (
        -jnp.ones(1),
        jnp.ones(1),
        None,
        None,
        jnp.tile(jnp.array([3.0, 4.0]), n),
        jnp.broadcast_to(jnp.eye(2), (n, 2, 2)),
        None,
        shared,
    )
    result = level.step(image, state, linear, 0.001, 0.0, 100.0)
    assert result is not None
    assert result[4] == pytest.approx(a * beta * 5.0)
    assert result[-1]["proposed_pose_step_rms"] == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("proposals", "expected_steps", "status", "forced_damping", "undamped"),
    [
        ([1.0] * 8, 8, "step_budget_reached", None, None),
        ([0.01] * 8, 8, "step_budget_reached", None, None),
        ([0.001] * 8, 4, "pose_steps_below_tolerance", None, None),
        ([1.0, 1.0, 1.0, 0.001, 0.001, 1.0], 5, "pose_steps_below_tolerance", None, None),
        (
            [0.001, 0.001, 1.0, 0.001, 1.0, 0.001, 0.001, 1.0],
            7,
            "pose_steps_below_tolerance",
            None,
            None,
        ),
        ([0.001] * 2, 2, "step_budget_reached", None, None),
        # Damped harder than the initial damping, a small proposal is checked again at the
        # initial damping: damping can hide a step the fit still wants to take...
        pytest.param(
            [0.001] * 8, 8, "step_budget_reached", 1.0, 0.05, id="damping_hides_a_large_step"
        ),
        # ...and cannot stop a fit whose poses have settled from being called converged.
        pytest.param(
            [0.001] * 8, 4, "pose_steps_below_tolerance", 1.0, 0.004, id="settled_under_damping"
        ),
    ],
)
def test_stopping_requires_two_small_proposals_at_the_initial_damping(
    monkeypatch, proposals, expected_steps, status, forced_damping, undamped
):
    remaining = iter(proposals)
    rechecked = []

    class Level:
        def __init__(self, ops, prior, *_args, **_kwargs):
            self.ops, self.prior, self.y = ops, prior, np.zeros((60, 1, 1))

        def linearize(self, _state, _image):
            return self.ops.linearize()

        def image_penalty(self, _image):
            return 0.0

        def proposal(self, _image, _state, _linear, lm, _weight):
            rechecked.append(lm)
            return undamped

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
        parameters=SimpleNamespace(local=()),  # no per-view columns: the photon scale stands
        linearize=lambda *_: (
            jnp.zeros(1),
            jnp.ones(1),
            None,
            None,
            jnp.zeros(60),
            jnp.ones((60, 1, 1)),
            100.0,
            (60, 0),
            None,
            None,
        ),
    )
    prior = SimpleNamespace(size=60, value=lambda *_: 0.0)
    monkeypatch.setattr(_solver, "_Level", Level)
    placed = SimpleNamespace(close=lambda: None, record=dict)
    monkeypatch.setattr(_solver, "level_stacks", lambda *_args, **_kwargs: placed)
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
        # The recheck is made at the initial damping, once per small damped proposal.
        assert rechecked == [_solver.Budget(1, 1).lm_initial] * expected_steps
    else:
        assert rechecked == []  # damping at its initial value: the step's own proposal decides
    np.testing.assert_array_equal(
        [step["proposed_pose_step_rms"] for step in record["steps"]],
        proposals[:expected_steps],
    )
