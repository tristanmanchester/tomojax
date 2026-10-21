"""Encoded payload ownership, shared lifetimes and slab-store integration."""

import gc
import weakref

import jax
import numpy as np
import pytest

from tomojax.core.device_storage import DeviceStorage
from tomojax.core.host_planes import HostPlaneOperator
from tomojax.core.host_volume import SlabStore
from tomojax.core.joseph import plane_coefficients
from tomojax.core.joseph_regions import PlaneRegion
from tomojax.geometry import Detector, Grid, LaminographyGeometry, stack_view_poses
from tomojax.recon.fista_host import fista_tv_core_host
from tomojax.recon.fista_tv_core import FistaCoreConfig

pytestmark = pytest.mark.numerical


@pytest.fixture(params=["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def device(request):
    try:
        return jax.devices(request.param)[0]
    except RuntimeError:
        pytest.skip(f"{request.param} device unavailable")


def test_payload_copies_input_and_each_read(device):
    storage = DeviceStorage(device, 32)
    words = np.array([0, 0x80000000, 0x7FC12345, 0x7F800000, 0xFFFFFFFF], np.uint32)
    expected = words.view(np.uint8).copy()
    block = storage.put(words.view(np.uint8))
    words.fill(0)
    first = block.read()
    np.testing.assert_array_equal(first, expected)
    first.fill(0)
    np.testing.assert_array_equal(block.read(), expected)
    assert storage.used == block.nbytes == expected.nbytes
    assert block.resident_nbytes == (block.nbytes if device.platform == "cpu" else 0)


def test_growth_and_payload_lifetime_outlive_owner(device):
    storage = DeviceStorage(device, 0)
    host = storage.put(np.arange(4, dtype=np.uint8))
    assert storage.used == storage.budget == 0 and host.resident_nbytes == 4
    storage.reserve(16)
    first = storage.put(np.arange(16, dtype=np.uint8))
    storage.reserve(8)
    second = storage.put(np.arange(8, dtype=np.uint8))
    assert storage.used == storage.budget == 24
    owner = weakref.ref(storage)
    del storage
    gc.collect()
    assert owner() is None
    np.testing.assert_array_equal(first.read(), np.arange(16, dtype=np.uint8))
    np.testing.assert_array_equal(second.read(), np.arange(8, dtype=np.uint8))


@pytest.mark.parametrize("compressed", [False, True])
def test_shared_slab_snapshots_fallback_spill_and_extent_reuse(device, tmp_path, compressed):
    shape = (9, 7, 5)
    rng = np.random.default_rng(4)
    words = rng.integers(0, 2**32, shape, dtype=np.uint32)
    if compressed:
        words.fill(0x80000000)
    values = words.view(np.float32)
    probe = SlabStore(shape, depth=2)
    probe.write(0, 5, values)
    budget = probe.encoded_nbytes
    assert (budget < values.nbytes) == compressed
    storage = DeviceStorage(device, budget)
    original = SlabStore(shape, depth=2, storage=storage)
    original.write(0, 5, values)
    snapshot = original.copy()
    fallback = SlabStore(shape, depth=2, storage=storage)
    fallback.write(0, 5, values)
    assert storage.used == budget == fallback.resident_nbytes
    original.write(0, 5, np.zeros(shape, np.float32))
    assert storage.used == budget
    np.testing.assert_array_equal(snapshot.read(1, 4).view(np.uint32), words[:, :, 1:4])
    del snapshot
    gc.collect()
    assert storage.used == 0 and fallback.resident_nbytes == budget
    original.write(0, 5, values)
    assert storage.used == budget  # Freed adjacent extents coalesce and are reusable.
    shared = original.copy()
    original.spill(tmp_path)
    fallback.spill(tmp_path)
    assert original.resident_nbytes == fallback.resident_nbytes == 0
    assert not list(tmp_path.iterdir())
    for store in (original, fallback, shared):
        out = store.materialize()
        np.testing.assert_array_equal(out.view(np.uint32), words)
        out.fill(0)
        np.testing.assert_array_equal(store.read(0, 5).view(np.uint32), words)


def test_fista_preserves_shared_storage_and_exact_host_result(device):
    grid, detector = Grid(4, 5, 3, 0.7, 0.8, 1.2), Detector(7, 5, 0.9, 0.8)
    backend = "cuda" if device.platform == "gpu" else "jax"
    rng = np.random.default_rng(61)
    initial = rng.uniform(-0.1, 0.7, (4, 5, 3)).astype(np.float32)
    data = rng.uniform(0, 2, (3, 5, 7)).astype(np.float32)
    storage = DeviceStorage(device, 4096)
    stored = SlabStore(initial.shape, depth=2, storage=storage)
    stored.write(0, 3, initial)
    with jax.default_device(device):
        poses = stack_view_poses(LaminographyGeometry(grid, detector, [13, 97, 208]), 3)
        operator = HostPlaneOperator(
            plane_coefficients(poses, grid, detector),
            region=PlaneRegion(grid=grid, detector=detector, backend=backend),
            slab_depth=2,
            detector_bytes=data.nbytes,
        )
        cfg = FistaCoreConfig(iterations=3, lipschitz=500, nonnegative=True, upper_bound=0.6)
        expected = fista_tv_core_host(operator, initial, projections=data, cfg=cfg)
        actual = fista_tv_core_host(operator, stored, projections=data, cfg=cfg)
    assert actual.x.storage is storage
    assert storage.used == stored.encoded_nbytes + actual.x.encoded_nbytes
    np.testing.assert_array_equal(actual.x.materialize(), expected.x)
    np.testing.assert_array_equal(stored.materialize(), initial)
    for name in ("loss", "data_loss", "regulariser_value"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
