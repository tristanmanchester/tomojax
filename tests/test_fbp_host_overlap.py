"""Bounded CUDA output-copy orchestration, using controlled stand-ins on the CPU."""

from concurrent.futures import ThreadPoolExecutor
import importlib
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from tomojax.geometry import Detector, Grid, ParallelGeometry
from tomojax.recon import FBPHostConfig, fbp_host


def _setup(monkeypatch, volume, *, out=None):
    module = importlib.import_module("tomojax.recon.fbp_host")
    grid, detector = Grid(3, 4, 6, 1, 1, 1), Detector(5, 6, 1, 1)
    geometry = ParallelGeometry(grid, detector, [0, 60, 120])
    data = np.zeros((3, 6, 5), np.float32)

    class Poses:
        shape = (3, 4, 4)

        def devices(self):
            return [
                SimpleNamespace(platform="gpu", client=SimpleNamespace(platform_version="cuda"))
            ]

        def __array__(self, dtype=None, copy=None):
            return np.array([geometry.pose_for_view(i) for i in range(3)], dtype=dtype, copy=copy)

    monkeypatch.setattr(module, "_OUTPUT_OVERLAP_MIN_BYTES", 0)
    monkeypatch.setattr(module, "device_free_memory_bytes", lambda: 2**30)
    monkeypatch.setattr(module, "stack_view_poses", lambda *_: Poses())
    monkeypatch.setattr(module, "_fbp_from_host", volume)
    return lambda: fbp_host(
        geometry, grid, detector, data, config=FBPHostConfig(slices_per_batch=2), out=out
    )


def test_one_pending_copy_overlaps_exactly_one_computing_slab(monkeypatch):
    copying, second_computed, release = Event(), Event(), Event()
    computed, enqueued = [], []

    class Volume:
        def __init__(self, index, shape):
            self.index, self.shape = index, shape

        def copy_to_host_async(self):
            enqueued.append(self.index)

        def __array__(self, dtype=None, copy=None):
            if self.index == 0:
                copying.set()
                assert release.wait(5), "test did not release the first copy"
            return np.full(self.shape, self.index + 1, dtype=dtype or np.float32)

    def compute(*args, **kwargs):
        index = len(computed)
        grid = kwargs["grid"]
        computed.append(index)
        if index == 1:
            second_computed.set()
        return Volume(index, (grid.nx, grid.ny, grid.nz))

    run = _setup(monkeypatch, compute)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(run)
        try:
            assert copying.wait(5)
            assert second_computed.wait(5)
            assert computed == [0, 1]
            assert enqueued == [0]
        finally:
            release.set()
        result = future.result(timeout=5)
    np.testing.assert_array_equal(result, np.broadcast_to([1, 1, 2, 2, 3, 3], result.shape))
    assert enqueued == [0, 1, 2]


def test_output_copy_failure_is_not_reported_as_success(monkeypatch):
    class Volume:
        def copy_to_host_async(self):
            pass

        def __array__(self, dtype=None, copy=None):
            raise RuntimeError("output copy failed")

    run = _setup(monkeypatch, lambda *args, **kwargs: Volume())
    with pytest.raises(RuntimeError, match="output copy failed"):
        run()


@pytest.mark.gpu
@pytest.mark.parametrize("z_center", [0.0, 0.3, 5.0])
@pytest.mark.parametrize("storage", ["C", "F", "axial", "reverse"])
def test_cuda_overlap_matches_serial_slabs_bit_for_bit(monkeypatch, z_center, storage):
    import jax

    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    module = importlib.import_module("tomojax.recon.fbp_host")
    grid, detector = Grid(9, 7, 11, 1, 1, 1, vol_center=(0, 0, z_center)), Detector(13, 11, 1, 1)
    geometry = ParallelGeometry(grid, detector, [0, 29, 73, 118, 163])
    data = np.random.default_rng(142).normal(size=(5, 11, 13)).astype(np.float32)
    config = FBPHostConfig(slices_per_batch=4)
    monkeypatch.setattr(module, "_OUTPUT_OVERLAP_MIN_BYTES", 2**60)
    expected = fbp_host(geometry, grid, detector, data, config=config)
    monkeypatch.setattr(module, "_OUTPUT_OVERLAP_MIN_BYTES", 0)
    if storage == "axial":
        out = np.empty((grid.nz, grid.nx, grid.ny), np.float32).transpose(1, 2, 0)
    elif storage == "reverse":
        out = np.empty(expected.shape, np.float32, order="F")[::-1]
    else:
        out = np.empty(expected.shape, np.float32, order=storage)
    actual = fbp_host(geometry, grid, detector, data, config=config, out=out)
    assert actual is out
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("free", [None, 1])
def test_no_extra_device_slab_when_memory_is_unknown_or_tight(monkeypatch, free):
    def compute(*args, **kwargs):
        grid = kwargs["grid"]
        # A NumPy result deliberately has no async device-copy method.
        return np.ones((grid.nx, grid.ny, grid.nz), np.float32)

    run = _setup(monkeypatch, compute)
    module = importlib.import_module("tomojax.recon.fbp_host")
    monkeypatch.setattr(module, "device_free_memory_bytes", lambda: free)
    np.testing.assert_array_equal(run(), np.ones((3, 4, 6), np.float32))


def test_packed_transfer_reserves_an_additional_device_slab(monkeypatch):
    def compute(*args, **kwargs):
        grid = kwargs["grid"]
        return np.ones((grid.nx, grid.ny, grid.nz), np.float32)

    out = np.empty((3, 4, 6), np.float32, order="F")
    run = _setup(monkeypatch, compute, out=out)
    module = importlib.import_module("tomojax.recon.fbp_host")
    # One 96-byte extra slab fits the 144-byte budget, but packing needs two.
    monkeypatch.setattr(module, "device_free_memory_bytes", lambda: 720)
    assert run() is out
    np.testing.assert_array_equal(out, 1)
