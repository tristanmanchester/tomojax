"""Transfer-buffer reuse follows array ownership, not submission order."""

import gc
import weakref

import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.recon._pinned import PinnedPool


@pytest.mark.parametrize(
    "view", [lambda a: a[:, 1:], lambda a: np.frombuffer(memoryview(a), dtype=np.float32)]
)
def test_pool_waits_for_every_view_and_does_not_escape_with_output(view):
    allocated = []

    def allocate(size):
        allocated.append(size)
        return bytearray(size)

    pool = PinnedPool(allocate)
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
    pool = PinnedPool(lambda size: (gc.collect(), bytearray(size))[1])
    previous = [pool.empty((3, 5))]
    previous.append(previous)
    del previous
    pool.empty((7, 5))
    assert len(pool.free[60]) == 1


@pytest.mark.gpu
def test_async_fourier_owns_upload_and_retained_outputs_without_global_watchers():
    pytest.importorskip("cupy")
    from tomojax.geometry import Detector, Grid

    # check-public-imports: allow-private
    from tomojax.recon._fourier_backend import FourierPlan

    grid, detector = Grid(11, 13, 2, 1, 1, 1), Detector(17, 2, 1, 1)
    angles = tuple(np.linspace(0, np.pi, 17, endpoint=False))
    plan = FourierPlan(
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
    from tomojax.recon import FourierConfig, fourier_reconstruct

    grid, detector = Grid(4, 5, 2, 1, 1, 1), Detector(7, 2, 1, 1)
    geometry = ParallelGeometry(grid, detector, [0.0, 90.0])
    output = fourier_reconstruct(
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
