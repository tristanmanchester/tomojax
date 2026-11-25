"""The levels' observations come from one front-to-back pass over the frames.

Coarsened observations average finite pixels and weight each block by its observed fraction.
"""

import numpy as np
import pytest

import tomojax as tj

# check-public-imports: allow-private
from tomojax.alignment._joint import _fit, _noise
from tomojax.geometry import stack_view_poses

pytestmark = pytest.mark.numerical


@pytest.mark.parametrize("missing", [False, True])
def test_coarsening_memory_map_matches_dense_fp32(tmp_path, missing):
    source = np.lib.format.open_memmap(tmp_path / "views.npy", "w+", dtype="f8", shape=(3, 6, 8))
    source[:] = np.random.default_rng(81).normal(size=source.shape)
    if missing:
        source[1, 2, 3] = np.nan
    (actual, *_), *rest = _fit._read(source, [2], views_per_batch=1)
    (expected, *_), *expected_rest = _fit._read(source.astype(np.float32), [2], views_per_batch=3)
    assert (actual[1] is None) is not missing
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    assert rest == expected_rest


def test_partial_blocks_average_their_finite_pixels():
    measured = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    measured[0, 0, 0] = np.nan
    measured[1, 2:, 2:] = np.nan
    [(y, precision)], _, _ = _fit._read(measured, [2])
    np.testing.assert_allclose(y[0, 0, 0], np.mean([1, 4, 5]))
    np.testing.assert_allclose(precision[0, 0, 0], 0.75)
    assert y[1, 1, 1] == 0 and precision[1, 1, 1] == 0
    np.testing.assert_array_equal(precision[1, :, 0], 1)


@pytest.mark.parametrize("missing_columns", [False, True])
def test_one_pass_gives_each_binning_the_scale_and_the_noise_of_separate_reads(missing_columns):
    # More windows than the noise estimate samples, so they are drawn at random and fall in
    # every slab; with every fourth column missing no fourth-order window is whole and the
    # estimate falls back to lower orders.
    values = np.random.default_rng(83).normal(1.0, 0.1, size=(40, 52, 56))
    if missing_columns:
        values[:, :, ::4] = np.nan
    binned, scale, sd = _fit._read(values, [2, 4], views_per_batch=3)
    for b, (y, precision) in zip([2, 4], binned, strict=True):
        [(alone, alone_precision)], _, _ = _fit._read(values, [b], views_per_batch=7)
        np.testing.assert_array_equal(y, alone)
        assert (precision is None) is (alone_precision is None) is not missing_columns
    sample = values[:: len(values) // 16].astype(np.float32)
    peak = np.max(np.abs(sample[np.isfinite(sample)]))
    assert scale == 2.0 ** -np.round(np.log2(peak))
    assert sd == _noise.estimate_noise(values, dtype=np.float32)


def test_coarsening_reads_one_view_batch_at_a_time():
    values = np.random.default_rng(82).normal(size=(5, 6, 8))

    class Source:
        shape = values.shape

        def __init__(self):
            self.reads: list[int] = []

        def __getitem__(self, key):
            block = values[key]
            self.reads.append(len(block))
            return block

        def __array__(self, *args, **kwargs):
            raise AssertionError("the whole stack must not be converted at once")

    source = Source()
    [(actual, _)], _, _ = _fit._read(source, [2], views_per_batch=2)
    [(expected, _)], _, _ = _fit._read(values.astype(np.float32), [2], views_per_batch=5)
    np.testing.assert_array_equal(actual, expected)
    assert source.reads == [2, 2, 1]  # a partial last batch, never more than two views


class _Recorded(np.ndarray):
    """Frames that record the views each indexing asks for (a slice as a range)."""

    def __array_finalize__(self, obj):
        self.reads = getattr(obj, "reads", None)

    def __getitem__(self, key):
        views = key[0] if isinstance(key, tuple) else key
        if not (isinstance(views, slice) and views == slice(None)):  # a crop keeps every view
            self.reads.append(range(len(self))[views] if isinstance(views, slice) else "gather")
        return super().__getitem__(key)


@pytest.mark.slow
def test_the_fit_reads_the_frames_front_to_back_once():
    # A memory map read at random stops reading ahead, and a sweep after that reads a page per
    # fault: every level's observations, the scale and the noise come from one sweep.
    grid, detector, views = tj.Grid(16, 16, 8, 1.0, 1.0, 1.0), tj.Detector(24, 12, 1.0, 1.0), 40
    geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, views, endpoint=False))
    volume = np.zeros((8, 16, 16), np.float32)
    volume[2:6, 4:11, 5:12] = 1.0
    projections = np.array(tj.project(geometry, volume)).view(_Recorded)
    projections.reads = []
    nominal = np.asarray(stack_view_poses(geometry, views))
    _fit.fit_poses(projections, nominal, grid, detector)
    assert projections.reads == [range(s, min(s + 16, views)) for s in range(0, views, 16)]
