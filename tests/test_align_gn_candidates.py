import jax.numpy as jnp
import numpy as np

from tomojax.align.pipeline import _select_gn_candidate


def _scalar_key(x):
    return float(np.asarray(x)[0, 0])


def test_select_gn_candidate_accepts_raw_step_without_extra_evals():
    params_prev = jnp.zeros((1, 1), dtype=jnp.float32)
    dp_all = jnp.ones_like(params_prev)
    eval_calls = []

    def eval_loss(candidate):
        eval_calls.append(_scalar_key(candidate))
        return 4.0

    best_params, best_loss = _select_gn_candidate(
        params_prev,
        dp_all,
        loss_before=5.0,
        eval_loss=eval_loss,
        gn_accept_tol=0.0,
    )

    np.testing.assert_allclose(best_params, params_prev + dp_all)
    assert best_loss == 4.0
    assert eval_calls == [1.0]


def test_select_gn_candidate_accepts_half_step_after_rejecting_raw():
    params_prev = jnp.zeros((1, 1), dtype=jnp.float32)
    dp_all = jnp.ones_like(params_prev)
    eval_calls = []

    def eval_loss(candidate):
        key = _scalar_key(candidate)
        eval_calls.append(key)
        if key == 1.0:
            return 6.0
        if key == 0.5:
            return 4.0
        raise AssertionError(f"unexpected candidate key {key}")

    best_params, best_loss = _select_gn_candidate(
        params_prev,
        dp_all,
        loss_before=5.0,
        eval_loss=eval_loss,
        gn_accept_tol=0.0,
    )

    np.testing.assert_allclose(best_params, params_prev + 0.5 * dp_all)
    assert best_loss == 4.0
    assert eval_calls == [1.0, 0.5]


def test_select_gn_candidate_only_smooths_the_better_base_candidate():
    params_prev = jnp.zeros((1, 1), dtype=jnp.float32)
    dp_all = jnp.ones_like(params_prev)
    eval_calls = []
    smooth_bases = []

    def eval_loss(candidate):
        key = _scalar_key(candidate)
        eval_calls.append(key)
        losses = {
            1.0: 8.0,
            0.5: 9.0,
            10.0: 4.0,
            20.0: 3.5,
            30.0: 3.0,
            40.0: 2.5,
        }
        if key not in losses:
            raise AssertionError(f"unexpected candidate key {key}")
        return losses[key]

    def smooth_candidate(base, weights):
        base_key = _scalar_key(base)
        smooth_bases.append(base_key)
        return jnp.asarray(weights, dtype=jnp.float32).reshape((1, 1))

    best_params, best_loss = _select_gn_candidate(
        params_prev,
        dp_all,
        loss_before=5.0,
        eval_loss=eval_loss,
        gn_accept_tol=0.0,
        smooth_candidate=smooth_candidate,
        light_smoothness_weights_sq=jnp.array([10.0], dtype=jnp.float32),
        medium_smoothness_weights_sq=jnp.array([20.0], dtype=jnp.float32),
        smoothness_weights_sq=jnp.array([30.0], dtype=jnp.float32),
        trans_only_smoothness_weights_sq=jnp.array([40.0], dtype=jnp.float32),
    )

    np.testing.assert_allclose(best_params, jnp.array([[40.0]], dtype=jnp.float32))
    assert best_loss == 2.5
    assert eval_calls == [1.0, 0.5, 10.0, 20.0, 30.0, 40.0]
    assert smooth_bases == [1.0, 1.0, 1.0, 1.0]


def test_select_gn_candidate_reverts_when_no_candidate_is_acceptable():
    params_prev = jnp.zeros((1, 1), dtype=jnp.float32)
    dp_all = jnp.ones_like(params_prev)

    def eval_loss(candidate):
        key = _scalar_key(candidate)
        losses = {
            1.0: 8.0,
            0.5: 7.0,
            10.0: 6.5,
            20.0: 6.0,
        }
        if key not in losses:
            raise AssertionError(f"unexpected candidate key {key}")
        return losses[key]

    def smooth_candidate(base, weights):
        del base
        return jnp.asarray(weights, dtype=jnp.float32).reshape((1, 1))

    best_params, best_loss = _select_gn_candidate(
        params_prev,
        dp_all,
        loss_before=5.0,
        eval_loss=eval_loss,
        gn_accept_tol=0.0,
        smooth_candidate=smooth_candidate,
        light_smoothness_weights_sq=jnp.array([10.0], dtype=jnp.float32),
        medium_smoothness_weights_sq=jnp.array([20.0], dtype=jnp.float32),
    )

    np.testing.assert_allclose(best_params, params_prev)
    assert best_loss == 5.0
