"""Transfer-buffer reuse follows array ownership, not submission order."""

import gc
import weakref

import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.recon import _fourier_backend, _pinned


@pytest.mark.parametrize(
    "view", [lambda a: a[:, 1:], lambda a: np.frombuffer(memoryview(a), dtype=np.float32)]
)
def test_pool_waits_for_every_view_and_does_not_escape_with_output(view):
    allocated = []

    def allocate(size):
        allocated.append(size)
        return bytearray(size)

    pool = _pinned.PinnedPool(allocate)
    first = pool.empty((3, 5))
    first.fill(7)
    retained = view(first)
    expected = retained.copy()
    del first
    gc.collect()
    second = pool.empty((3, 5))
    second.fill(9)
    np.testing.assert_array_equal(retained, expected)
    assert allocated == [60, 60]
    del second
    gc.collect()
    reused = pool.empty((3, 5))
    assert allocated == [60, 60]
    owner = weakref.ref(pool)
    del pool
    gc.collect()
    assert owner() is None
    np.testing.assert_array_equal(retained, expected)
    np.testing.assert_array_equal(reused, 9)


def test_pool_allocation_can_collect_a_preceding_buffer():
    pool = _pinned.PinnedPool(lambda size: (gc.collect(), bytearray(size))[1])
    previous = [pool.empty((3, 5))]
    previous.append(previous)
    del previous
    pool.empty((7, 5))
    assert len(pool.free[60]) == 1


@pytest.mark.gpu
def test_async_fourier_owns_upload_and_retained_outputs_without_global_watchers():
    pytest.importorskip("cupy")
    from tomojax.geometry import Detector, Grid

    grid, detector = Grid(11, 13, 2, 1, 1, 1), Detector(17, 2, 1, 1)
    angles = tuple(np.linspace(0, np.pi, 17, endpoint=False))
    plan = _fourier_backend.FourierPlan(
        grid, detector, 17, np.zeros(17, np.int8), angles, 2, "cupy", async_transfers=True
    )
    first = plan.enqueue(np.ones((2, 17, 17), np.float32), count=2)
    expected = first.wait().copy()
    uploaded = weakref.ref(plan._upload_slot)
    second = plan.enqueue(np.full((2, 17, 17), 2, np.float32), count=1)
    result = second.wait()
    np.testing.assert_array_equal(first.output, expected)
    np.testing.assert_array_equal(result, 2 * expected[:1])
    del first, second, plan
    gc.collect()
    assert uploaded() is None
    np.testing.assert_array_equal(result, 2 * expected[:1])


def test_fourier_output_mapping_follows_escaped_buffer_views():
    from tomojax.geometry import Detector, Grid, ParallelGeometry
    from tomojax.recon import FourierConfig, fourier

    grid, detector = Grid(4, 5, 2, 1, 1, 1), Detector(7, 2, 1, 1)
    geometry = ParallelGeometry(grid, detector, [0.0, 90.0])
    output = fourier(
        geometry,
        grid,
        detector,
        np.ones((2, 2, 7), np.float32),
        config=FourierConfig(backend="numpy"),
    )
    retained = np.frombuffer(memoryview(output), dtype=np.float32)
    expected, owner = retained.copy(), weakref.ref(output.base)
    del output
    gc.collect()
    assert owner() is not None
    np.testing.assert_array_equal(retained, expected)
    del retained
    gc.collect()
    assert owner() is None


@pytest.mark.gpu
def test_raw_fourier_releases_ordered_rows_before_allocating_the_fft(monkeypatch):
    import cupy as cp

    import tomojax as tj
    from tomojax.recon import FourierConfig

    grid = tj.Grid(9, 9, 6, 1, 1, 1)
    detector = tj.Detector(11, 6, 1, 1)
    geometry = tj.ParallelGeometry(grid, detector, [60, 0, 120])
    frames = tj.Frames(np.full((3, 6, 11), 400, np.float32), geometry, white_level=1000)
    contiguous, rfft = cp.ascontiguousarray, cp.fft.rfft
    ordered = []
    checks = []

    def remember(array, *args, **kwargs):
        result = contiguous(array, *args, **kwargs)
        if result.shape == (2, 3, 11):
            ordered.append(weakref.ref(result))
        return result

    def transform(array, *args, **kwargs):
        assert ordered and all(ref() is None for ref in ordered)
        checks.append(None)
        return rfft(array, *args, **kwargs)

    monkeypatch.setattr(cp, "ascontiguousarray", remember)
    monkeypatch.setattr(cp.fft, "rfft", transform)
    with tj.reconstruct_slabs(
        frames, "fourier", config=FourierConfig(backend="cupy", slices_per_batch=2)
    ) as slabs:
        assert [section.stop for section, _ in slabs] == [2, 4, 6]
    assert len(checks) == 3


@pytest.mark.parametrize(
    "backend",
    ["numpy", *(pytest.param(name, marks=pytest.mark.gpu) for name in ("cupy", "async", "rows"))],
)
def test_a_slab_with_a_nonfinite_value_is_refused_when_collected(backend):
    from tomojax.geometry import Detector, Grid

    grid, detector = Grid(11, 13, 2, 1, 1, 1), Detector(17, 2, 1, 1)
    angles = tuple(np.linspace(0, np.pi, 17, endpoint=False))
    plan = _fourier_backend.FourierPlan(
        grid,
        detector,
        17,
        np.zeros(17, np.int8),
        angles,
        2,
        "numpy" if backend == "numpy" else "cupy",
        async_transfers=backend in {"async", "rows"},
    )
    finite = np.ones((2, 17, 17), np.float32)
    bad = finite.copy()
    bad[1, 3, 4] = np.inf
    if backend in {"numpy", "cupy"}:
        assert np.isfinite(plan.reconstruct(finite, count=2)).all()
        with pytest.raises(ValueError, match="finite"):
            _ = plan.reconstruct(bad, count=2)
        return
    if backend == "async":
        first, second = plan.enqueue(finite, count=2), plan.enqueue(bad, count=2)
    else:
        order = plan.xp.arange(17)
        rows = np.array([0.0, 1.0])
        first = plan.enqueue_rows(finite.transpose(1, 0, 2), order, rows)
        second = plan.enqueue_rows(bad.transpose(1, 0, 2), order, rows)
    assert np.isfinite(first.wait()).all()
    with pytest.raises(ValueError, match="finite"):
        _ = second.wait()
