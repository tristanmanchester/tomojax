import jax.numpy as jnp
import pytest

from tomojax.align.losses import (
    LossState,
    _loss_cauchy,
    _loss_mi_kde,
    _loss_renyi_mi,
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


def _reference_kde_probabilities(
    pred: jnp.ndarray,
    tar: jnp.ndarray,
    bx: jnp.ndarray,
    by: jnp.ndarray,
    bwx: float,
    bwy: float,
):
    pr = pred.ravel()[:, None]
    tr = tar.ravel()[:, None]
    Wx = jnp.exp(-0.5 * ((pr - bx[None, :]) / bwx) ** 2)
    Wy = jnp.exp(-0.5 * ((tr - by[None, :]) / bwy) ** 2)
    Wx = Wx / (jnp.sum(Wx, axis=1, keepdims=True) + 1e-12)
    Wy = Wy / (jnp.sum(Wy, axis=1, keepdims=True) + 1e-12)
    Px = jnp.clip(Wx.mean(axis=0), 1e-12, 1.0)
    Py = jnp.clip(Wy.mean(axis=0), 1e-12, 1.0)
    Pxy = jnp.clip((Wx[:, :, None] * Wy[:, None, :]).mean(axis=0), 1e-12, 1.0)
    return Px, Py, Pxy


def _reference_mi_kde_loss(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> float:
    assert st.bins_x is not None
    assert st.bins_y is not None
    assert st.bw_x is not None
    assert st.bw_y is not None
    Px, Py, Pxy = _reference_kde_probabilities(pred, tar, st.bins_x, st.bins_y, st.bw_x, st.bw_y)
    Hx = -jnp.sum(Px * jnp.log(Px))
    Hy = -jnp.sum(Py * jnp.log(Py))
    Hxy = -jnp.sum(Pxy * jnp.log(Pxy))
    mi = Hx + Hy - Hxy
    if int(st.params.get("nmi", 0)) == 1:
        mi = mi / (jnp.sqrt(Hx * Hy) + 1e-12)
        return float((1.0 - mi) * jnp.float32(pred.size))
    return float((-mi) * jnp.float32(pred.size))


def _reference_renyi_mi_loss(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> float:
    assert st.bins_x is not None
    assert st.bins_y is not None
    assert st.bw_x is not None
    assert st.bw_y is not None
    a = float(st.params["alpha"])
    Px, Py, Pxy = _reference_kde_probabilities(pred, tar, st.bins_x, st.bins_y, st.bw_x, st.bw_y)
    Hx = (1.0 / (1.0 - a)) * jnp.log(jnp.sum(Px ** a))
    Hy = (1.0 / (1.0 - a)) * jnp.log(jnp.sum(Py ** a))
    Hxy = (1.0 / (1.0 - a)) * jnp.log(jnp.sum(Pxy ** a))
    return float((-(Hx + Hy - Hxy)) * jnp.float32(pred.size))


def test_mi_kde_matches_broadcast_reference():
    pred = jnp.array([[0.1, 0.7], [-0.3, 1.2]], dtype=jnp.float32)
    tar = jnp.array([[0.2, 0.6], [-0.1, 1.0]], dtype=jnp.float32)
    st = LossState(
        kind="mi_kde",
        params={"bins": 4},
        bins_x=jnp.array([-0.5, 0.0, 0.5, 1.0], dtype=jnp.float32),
        bins_y=jnp.array([-0.4, 0.1, 0.6, 1.1], dtype=jnp.float32),
        bw_x=0.25,
        bw_y=0.3,
    )

    actual = float(_loss_mi_kde(pred, tar, st))
    expected = _reference_mi_kde_loss(pred, tar, st)

    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_nmi_kde_matches_broadcast_reference():
    pred = jnp.array([[0.0, 0.8], [-0.2, 1.1]], dtype=jnp.float32)
    tar = jnp.array([[0.1, 0.9], [-0.3, 0.95]], dtype=jnp.float32)
    st = LossState(
        kind="nmi_kde",
        params={"bins": 4, "nmi": 1},
        bins_x=jnp.array([-0.5, 0.0, 0.5, 1.0], dtype=jnp.float32),
        bins_y=jnp.array([-0.4, 0.1, 0.6, 1.1], dtype=jnp.float32),
        bw_x=0.2,
        bw_y=0.25,
    )

    actual = float(_loss_mi_kde(pred, tar, st))
    expected = _reference_mi_kde_loss(pred, tar, st)

    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_renyi_mi_matches_broadcast_reference():
    pred = jnp.array([[0.05, 0.75], [-0.25, 1.15]], dtype=jnp.float32)
    tar = jnp.array([[0.15, 0.85], [-0.05, 1.05]], dtype=jnp.float32)
    st = LossState(
        kind="renyi_mi",
        params={"bins": 4, "alpha": 1.5},
        bins_x=jnp.array([-0.5, 0.0, 0.5, 1.0], dtype=jnp.float32),
        bins_y=jnp.array([-0.4, 0.1, 0.6, 1.1], dtype=jnp.float32),
        bw_x=0.22,
        bw_y=0.28,
    )

    actual = float(_loss_renyi_mi(pred, tar, st))
    expected = _reference_renyi_mi_loss(pred, tar, st)

    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("kind", ["renyi_mi", "tsallis_mi"])
def test_build_loss_rejects_alpha_equal_one_for_renyi_family(kind):
    targets = jnp.zeros((1, 8, 8), dtype=jnp.float32)

    with pytest.raises(ValueError, match="alpha=1"):
        build_loss(kind, {"alpha": 1.0}, targets)


def test_gn_candidate_must_improve_current_best_loss():
    assert not _should_prefer_gn_candidate(100.0, 100.5, 100.8, 0.01)
    assert _should_prefer_gn_candidate(100.0, 100.5, 100.4, 0.01)



def test_per_view_loss_uses_global_view_indices_for_precomputes():
    pytest.importorskip("scipy")

    targets = jnp.zeros((4, 8, 8), dtype=jnp.float32)
    targets = targets.at[0, :, 1].set(1.0)
    targets = targets.at[1, :, 2].set(1.0)
    targets = targets.at[2, :, 5].set(1.0)
    targets = targets.at[3, :, 6].set(1.0)

    loss_fn, _ = build_loss("chamfer_edge", {}, targets)
    pred = targets[2:]

    correct = loss_fn(
        pred,
        pred,
        None,
        view_indices=jnp.array([2, 3], dtype=jnp.int32),
    )
    wrong = loss_fn(pred, pred, None)

    assert float(jnp.mean(correct)) < 0.5 * float(jnp.mean(wrong))
