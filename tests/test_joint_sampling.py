"""The levels' observations come from one front-to-back pass over the frames.

Coarsened observations average finite pixels and weight each block by its observed fraction.
"""

from dataclasses import replace

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
    measured = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    measured[1, 0, 0] = np.nan
    measured[2, 2:, 2:] = np.nan
    [(y, precision)], _, _ = _fit._read(measured, [2], views_per_batch=1)
    np.testing.assert_allclose(y[1, 0, 0], np.mean([17, 20, 21]))
    np.testing.assert_allclose(precision[1, 0, 0], 0.75)
    assert y[2, 1, 1] == 0 and precision[2, 1, 1] == 0
    np.testing.assert_array_equal(precision[2, :, 0], 1)
    np.testing.assert_array_equal(precision[[0, 3]], 1)


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


def test_column_calibration_respects_missing_rays_at_both_resolutions():
    rng = np.random.default_rng(31)
    data = rng.normal(0, 0.01, (64, 16, 32)).astype(np.float32)
    data[:, 4:10, 9:20] += 0.8
    data += rng.normal(0, 0.06, 32).astype(np.float32)
    data[::5, :, 7] = np.nan
    data[:, 8:, 3::5] = np.nan
    data[:, :, -1] = np.nan
    levels, _, noise = _fit._read(data, [2, 1])
    _fit.calibrate_columns(levels, [2, 1], noise)
    coarse, fine = levels
    weighted = (fine[0] * fine[1]).reshape(64, 8, 2, 16, 2).mean(axis=(2, 4))
    np.testing.assert_allclose(coarse[0] * coarse[1], weighted, rtol=1e-6, atol=2e-7)
    assert np.std(fine[0][:, :3, :-1].mean((0, 1))) < 0.03


@pytest.mark.slow
def test_column_shadow_on_a_stationary_object_keeps_pose_and_image():
    sim = tj.simulate("shepp", size=48, n_views=96, geometry="parallel")
    data = np.asarray(sim.scan.projections)
    profile = data.mean(axis=1)
    shadow = np.zeros_like(profile)
    u = np.arange(data.shape[-1])
    for i, row in enumerate(profile):
        shift = 3 * np.sin(i * 0.8) * np.exp(-(((i - 27) / 7) ** 2))
        shadow[i] = np.interp(u - shift, u, row) - row
    measured = replace(sim.scan, projections=data + shadow[:, None, :])
    result = tj.align(measured, fit="translations", grid=measured.grid)
    assert np.sqrt(np.mean(result.poses[:, 3] ** 2)) < 0.02
    relative_error = np.linalg.norm(result.image - sim.phantom) / np.linalg.norm(sim.phantom)
    assert relative_error < 0.16


def test_masked_column_covariance_has_matched_transpose_and_preserves_flux():
    import jax.numpy as jnp

    column_filter = _fit.column_filter

    rng = np.random.default_rng(711)
    x, z = rng.normal(size=(2, 4, 3, 8, 7)).astype(np.float32)
    w = rng.uniform(0.15, 1, (4, 8, 7)).astype(np.float32)
    w[0] = 0
    w[1, :, 3] = 0
    w[2, 2:5, :4] = 0
    forward = column_filter(x, 0.85, w)
    masked = np.where(w[:, None] > 0, x, np.nan)
    np.testing.assert_array_equal(column_filter(masked, 0.85, w), forward)
    adjoint = column_filter(jnp.asarray(z), 0.85, jnp.asarray(w), transpose=True)
    np.testing.assert_allclose(np.vdot(forward, z), np.vdot(x, adjoint), rtol=2e-6)
    np.testing.assert_allclose(
        (forward * w[:, None]).sum((2, 3)), (x * w[:, None]).sum((2, 3)), atol=2e-6
    )
    columns = np.broadcast_to(x[:, :, :1, :], x.shape)
    filtered = column_filter(columns, 0.85, None)
    contrast = columns - columns.mean(axis=-1, keepdims=True)
    np.testing.assert_allclose(filtered, columns - 0.85 * contrast, atol=3e-7)


def test_column_calibration_requires_evidence_in_distant_rows():
    rng = np.random.default_rng(13)
    n, v, u = 96, 32, 48
    data = rng.normal(0, 0.01, (n, v, u)).astype(np.float32)
    t = np.arange(n)[:, None]
    x = np.linspace(-1, 1, u)
    signal = (0.2 * np.exp(-(((t - 24) / 3) ** 2)) * np.sin(2 * np.pi * x)).astype(np.float32)
    localized = data.copy()
    localized[:, v // 4 : 3 * v // 4] += signal[:, None]
    common = data + signal[:, None]
    residual = []
    for original in (localized, common):
        levels, _, noise = _fit._read(original, [1])
        _fit.calibrate_columns(levels, [1], noise)
        result = levels[0][0]
        # Score the imposed field at its peak, relative to exposures far from it.
        difference = (result[24] - result[60:80].mean(axis=0))[v // 4 : 3 * v // 4].mean(axis=0)
        residual.append(np.vdot(difference, signal[24]) / np.vdot(signal[24], signal[24]))
    assert residual[0] == pytest.approx(1.0, abs=0.05)
    assert residual[1] < 0.1
