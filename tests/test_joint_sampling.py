"""Coarsened observations average finite pixels and weight each block by its observed fraction."""

import numpy as np
import pytest

from tomojax.alignment._joint._fit import _coarsened  # check-public-imports: allow-private

pytestmark = pytest.mark.numerical


@pytest.mark.parametrize("missing", [False, True])
def test_coarsening_memory_map_matches_dense_fp32(tmp_path, missing):
    source = np.lib.format.open_memmap(tmp_path / "views.npy", "w+", dtype="f8", shape=(3, 6, 8))
    source[:] = np.random.default_rng(81).normal(size=source.shape)
    if missing:
        source[1, 2, 3] = np.nan
    actual = _coarsened(source, 2, batch_views=1)
    expected = _coarsened(source.astype(np.float32), 2, batch_views=3)
    assert (actual[1] is None) is not missing
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)


def test_partial_blocks_average_their_finite_pixels():
    measured = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    measured[0, 0, 0] = np.nan
    measured[1, 2:, 2:] = np.nan
    y, precision = _coarsened(measured, 2)
    np.testing.assert_allclose(y[0, 0, 0], np.mean([1, 4, 5]))
    np.testing.assert_allclose(precision[0, 0, 0], 0.75)
    assert y[1, 1, 1] == 0 and precision[1, 1, 1] == 0
    np.testing.assert_array_equal(precision[1, :, 0], 1)
