"""The nonlinear joint step must decrease the true objective with bounded work."""

import numpy as np

from tomojax.alignment._joint._solver import _line_search  # check-public-imports: allow-private


def test_joint_quadratic_step_checks_pose_fractions():
    calls = []

    def trial(a, beta):
        calls.append((a, beta))
        x, p = 1 - a, 1 - a * beta
        return 0.5 * (x * x + p * p), x, p

    a, beta, state = _line_search(trial, 1.0, -1.0, -1.0)
    assert (a, beta) == (1.0, 1.0)
    assert state[0] == 0
    assert calls == [(1.0, 1.0), (1.0, 0.0), (1.0, 0.5)]


def test_pose_fraction_checks_true_nonlinear_objective():
    def trial(a, beta):
        x, p = 1 - a, 1 - a * beta
        return 0.5 * (x * x + p * p) + 100 * (a * beta) ** 4, x, p

    a, beta, state = _line_search(trial, 1.0, -1.0, -1.0)
    assert beta < 1
    assert state[0] <= 1 - 1e-4 * a * (1 + beta)
    assert np.isfinite(state[0])
    a, beta, state = _line_search(lambda a, b: (10 - 9 * a + a * b * (2 * a - 1), a, b), 10, -9, -1)
    assert (a, beta, state[0]) == (1.0, 0.0, 1.0)


def test_nonfinite_trials_cannot_be_accepted():
    def trial(a, beta):
        return float("inf"), a, beta

    assert _line_search(trial, 1.0, -1.0, -1.0) is None


def test_pose_move_must_improve_on_the_same_image_only_step():
    # The image supplies all improvement. Ties must keep the held pose.
    a, beta, state = _line_search(lambda a, b: (2 - a, a, b), 2.0, -1.0, 0.0)
    assert (a, beta, state[0]) == (1.0, 0.0, 1.0)
