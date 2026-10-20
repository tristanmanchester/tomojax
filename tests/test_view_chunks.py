"""Fixed-size view chunks: full chunks, a shifted last chunk and the mask of its repeats."""

from __future__ import annotations

import numpy as np
import pytest

from tomojax.core.view_chunks import chunk_schedule, chunk_size


@pytest.mark.parametrize(("n_views", "size"), [(10, 4), (12, 4), (7, 7), (5, 8), (1, 3)])
def test_chunks_cover_every_view_once_through_the_mask(n_views: int, size: int) -> None:
    b = chunk_size(n_views, size, default=1)
    assert b == min(size, n_views)
    counted = np.zeros(n_views, int)
    for i in range((n_views + b - 1) // b):
        start, mask, idx = (np.asarray(x) for x in chunk_schedule(i, n_views=n_views, chunk_size=b))
        assert idx[0] == start and idx[-1] <= n_views - 1 and mask.dtype == np.float32
        assert set(np.unique(mask)) <= {0.0, 1.0}
        counted[idx[mask == 1.0]] += 1
    np.testing.assert_array_equal(counted, 1)


def test_last_chunk_is_shifted_back_and_masks_the_views_it_repeats() -> None:
    start, mask, idx = (np.asarray(x) for x in chunk_schedule(2, n_views=10, chunk_size=4))
    assert start == 6
    np.testing.assert_array_equal(idx, [6, 7, 8, 9])
    np.testing.assert_array_equal(mask, [0.0, 0.0, 1.0, 1.0])


def test_chunk_size_default_applies_when_no_batch_is_requested() -> None:
    assert chunk_size(30, None, default=1) == 1
    assert chunk_size(30, 0, default=30) == 30
    assert chunk_size(30, 50, default=1) == 30
