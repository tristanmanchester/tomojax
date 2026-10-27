"""Splits exclude observations, and scores neither fit nor rescale predictions."""

import json

import numpy as np
import pytest

from tomojax.evaluation import Score, Split


def splits():
    shape = (24, 2, 31)
    return [
        Split.interleaved(shape, stride=4, offset=1),
        Split.angular(shape, np.arange(24) * 15, start=330, width=60, guard=15),
        Split.detector(shape, start=2, width=3, period=12, guard=2, offsets=np.arange(24) % 4),
    ]


@pytest.mark.parametrize("split", splits())
def test_frozen_masks_and_streamed_scores_match_independent_sums(split):
    rng = np.random.default_rng(42)
    target = rng.normal(size=split.shape).astype(np.float32)
    prediction = target * 1.13 + 0.3
    weights = rng.uniform(size=(split.shape[0], 1, split.shape[2]))
    test = np.broadcast_to(split.test, split.shape)
    train = np.broadcast_to(split.train, split.shape)
    assert not np.any(test & train)
    fit = target[train].copy()
    altered = target.copy()
    altered[~train] = 1234
    np.testing.assert_array_equal(altered[train], fit)
    score = split.score(prediction, target, weights=weights)
    w = np.broadcast_to(weights, split.shape)[test]
    residual = prediction[test].astype(float) - target[test].astype(float)
    assert score.squared_error == pytest.approx(np.sum(w * residual**2), rel=1e-14)
    assert score.target_energy == pytest.approx(np.sum(w * target[test].astype(float) ** 2))
    assert score.count == test.sum()
    total = Score(0, 0, 0, 0)
    for start in range(0, split.shape[0], 5):
        v = slice(start, start + 5)
        total += split.score(prediction[v], target[v], weights=weights[v], views=v)
    assert total.mse == pytest.approx(score.mse, rel=1e-14)
    assert total.relative_l2 == pytest.approx(score.relative_l2, rel=1e-14)
    assert total.count == score.count
    fingerprint = split.fingerprint
    assert json.loads(split.specification)["kind"] in {"interleaved", "angular", "detector"}
    for mask in (split.train, split.test):
        with pytest.raises(ValueError):
            mask.setflags(write=True)
    assert split.fingerprint == fingerprint


def test_split_boundaries_and_guard_are_in_acquisition_coordinates():
    split = Split.angular(
        (6, 1, 1), np.array([30, 350, 5, 345, 90, 10]), start=350, width=20, guard=5
    )
    np.testing.assert_array_equal(split.test.ravel(), [False, True, True, False, False, False])
    np.testing.assert_array_equal(split.train.ravel(), [True, False, False, False, True, False])
    split = Split.detector((2, 1, 16), start=4, width=3, period=16, guard=2, offsets=[0, 1])
    np.testing.assert_array_equal(np.flatnonzero(split.test[0]), [4, 5, 6])
    np.testing.assert_array_equal(np.flatnonzero(~split.train[1]), np.arange(3, 10))
    assert split.fingerprint != Split.detector((2, 1, 16), start=4, width=3, period=16).fingerprint


def test_scoring_excludes_only_unscored_nonfinite_values_and_never_fits_scale():
    split = Split.interleaved((4, 1, 2), stride=2)
    target = np.ones(split.shape)
    prediction = target * 2
    target[1::2] = prediction[1::2] = np.nan
    score = split.score(prediction, target)
    assert score.mse == score.relative_l2 == 1
    assert score.count == 4
    prediction[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        split.score(prediction, target)
    weights = np.ones(split.shape)
    weights[0] = 0
    assert split.score(prediction, target, weights=weights).count == 2
    with pytest.raises(ValueError, match="nonnegative"):
        split.score(prediction, target, weights=-1)


def test_invalid_splits_do_not_silently_use_training_observations_for_scoring():
    with pytest.raises(ValueError, match="nonempty"):
        Split.interleaved((1, 1, 2), stride=2)
    with pytest.raises(ValueError, match="every view"):
        Split.detector((2, 1, 5), start=9, width=2, period=16)
    with pytest.raises(ValueError, match="width"):
        Split.angular((4, 1, 1), np.arange(4), start=0, width=360)
    with pytest.raises(ValueError, match="disjoint"):
        Split(
            shape=(2, 1, 2),
            train=np.ones((2, 1, 1), bool),
            test=np.ones((2, 1, 1), bool),
            specification="{}",
        )
