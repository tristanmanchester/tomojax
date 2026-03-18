import jax.numpy as jnp
import pytest

from tomojax.align.losses import (
    LossState,
    _loss_cauchy,
    _loss_welsch,
    build_loss,
    loss_is_within_relative_tolerance,
)
from tomojax.align.pipeline import _should_prefer_gn_candidate


def test_loss_is_within_relative_tolerance_allows_small_increase():
    assert loss_is_within_relative_tolerance(100.0, 100.5, 0.01)
    assert loss_is_within_relative_tolerance(100.0, 99.0, 0.01)
    assert not loss_is_within_relative_tolerance(100.0, 101.0, 0.01)


def test_cauchy_matches_log1p_form():
    pred = jnp.array([[0.0, 2.0, 5.0]], dtype=jnp.float32)
    tar = jnp.array([[0.0, 1.0, -1.0]], dtype=jnp.float32)
    st = LossState(kind="cauchy", params={"c": 2.0})

    r = (pred - tar).astype(jnp.float32)
    expected = jnp.sum(0.5 * (2.0 ** 2) * jnp.log1p((r / 2.0) ** 2))

    assert float(_loss_cauchy(pred, tar, st)) == pytest.approx(float(expected), rel=1e-6)


def test_welsch_preserves_exponential_form():
    pred = jnp.array([[0.0, 2.0, 5.0]], dtype=jnp.float32)
    tar = jnp.array([[0.0, 1.0, -1.0]], dtype=jnp.float32)
    st = LossState(kind="welsch", params={"c": 2.0})

    r = (pred - tar).astype(jnp.float32)
    expected = jnp.sum(0.5 * (2.0 ** 2) * (1.0 - jnp.exp(-((r / 2.0) ** 2))))

    assert float(_loss_welsch(pred, tar, st)) == pytest.approx(float(expected), rel=1e-6)


def test_build_loss_distinguishes_cauchy_and_welsch():
    targets = jnp.zeros((1, 1, 1), dtype=jnp.float32)
    pred = jnp.array([[[10.0]]], dtype=jnp.float32)
    tar = jnp.zeros_like(pred)

    cauchy_fn, _ = build_loss("cauchy", {"c": 1.0}, targets)
    welsch_fn, _ = build_loss("welsch", {"c": 1.0}, targets)
    leclerc_fn, _ = build_loss("leclerc", {"c": 1.0}, targets)

    cauchy_val = float(cauchy_fn(pred, tar, None)[0])
    welsch_val = float(welsch_fn(pred, tar, None)[0])
    leclerc_val = float(leclerc_fn(pred, tar, None)[0])

    assert cauchy_val > welsch_val
    assert cauchy_val == pytest.approx(0.5 * float(jnp.log1p(100.0)), rel=1e-6)
    assert welsch_val == pytest.approx(0.5 * (1.0 - float(jnp.exp(-100.0))), rel=1e-6)
    assert leclerc_val == pytest.approx(welsch_val, rel=1e-6)


def test_build_loss_accepts_lorentzian_alias_for_cauchy():
    targets = jnp.zeros((1, 1, 1), dtype=jnp.float32)
    pred = jnp.array([[[3.0]]], dtype=jnp.float32)
    tar = jnp.zeros_like(pred)

    cauchy_fn, _ = build_loss("cauchy", {"c": 1.0}, targets)
    lorentzian_fn, _ = build_loss("lorentzian", {"c": 1.0}, targets)

    cauchy_val = float(cauchy_fn(pred, tar, None)[0])
    lorentzian_val = float(lorentzian_fn(pred, tar, None)[0])

    assert lorentzian_val == pytest.approx(cauchy_val, rel=1e-6)


def test_gn_candidate_must_improve_current_best_loss():
    assert not _should_prefer_gn_candidate(100.0, 100.5, 100.8, 0.01)
    assert _should_prefer_gn_candidate(100.0, 100.5, 100.4, 0.01)
