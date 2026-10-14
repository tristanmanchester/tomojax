"""Detector binning reduces device pixels before returning independent host data."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj


def _scan(data):
    views, rows, cols = data.shape
    grid = tj.Grid(8, 7, 3, 1.1, 1.2, 1.3)
    detector = tj.Detector(cols, rows, 0.7, 0.9, (-0.31, 0.23))
    geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, views))
    return tj.Scan(data, geometry, name="binned fixture")


def _capture_downloads(monkeypatch):
    downloads = []

    def track(convert):
        def downloaded(array, *args, **kwargs):
            if isinstance(array, jax.Array):
                downloads.append((array.shape, array.dtype, array.devices()))
            return convert(array, *args, **kwargs)

        return downloaded

    monkeypatch.setattr(np, "array", track(np.array))
    monkeypatch.setattr(np, "asarray", track(np.asarray))
    return downloads


@pytest.mark.parametrize("factor", [2, 3, 4])
@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("layout", ["contiguous", "strided"])
def test_device_binning_matches_host_float32_mean(factor, dtype, layout):
    rng = np.random.default_rng(37)
    values = rng.uniform(0.01, 200, (5, 13, 17)).astype(dtype)
    if layout == "strided":
        values = values[:, ::-1, ::-1]
    with jax.enable_x64():
        original = _scan(jax.device_put(values))
        actual = original.binned(factor)
    expected = _scan(values).binned(factor)
    np.testing.assert_allclose(actual.projections, expected.projections, rtol=3e-7, atol=2e-5)
    assert actual.projections.dtype == np.float32
    assert isinstance(actual.projections, np.ndarray)
    assert actual.projections.flags.writeable
    assert actual.grid == expected.grid
    assert actual.detector == expected.detector
    np.testing.assert_array_equal(actual.angles, expected.angles)
    assert actual.name == original.name
    assert actual.corrections is original.corrections
    assert actual.source is original.source
    actual.projections[:] = 0
    np.testing.assert_array_equal(original.projections, values)


@pytest.mark.parametrize("factor", [2, 3, 4])
def test_device_binning_downloads_only_reduced_pixels(monkeypatch, factor):
    values = np.arange(5 * 13 * 17, dtype=np.float32).reshape(5, 13, 17)
    device_data = jax.device_put(values)
    downloads = _capture_downloads(monkeypatch)
    actual = _scan(device_data).binned(factor)
    assert downloads == [
        ((5, 13 // factor, 17 // factor), np.dtype("float32"), device_data.devices())
    ]
    assert actual.projections.flags.writeable


@pytest.mark.parametrize("storage", ["array", "readonly", "memmap"])
def test_host_binning_stays_exact_and_independent(tmp_path, storage):
    values = np.arange(5 * 13 * 17, dtype=np.float64).reshape(5, 13, 17) / 11
    if storage == "readonly":
        values.flags.writeable = False
    elif storage == "memmap":
        mapped = np.memmap(
            tmp_path / "pixels.dat", mode="w+", dtype=values.dtype, shape=values.shape
        )
        mapped[:] = values
        values = mapped
    expected = values.astype(np.float32)[:, :12, :16].reshape(5, 6, 2, 8, 2).mean(axis=(2, 4))
    actual = _scan(values).binned(2)
    np.testing.assert_array_equal(actual.projections, expected)
    assert actual.projections.flags.writeable
    assert not np.shares_memory(actual.projections, values)


def test_device_binning_casts_before_averaging_and_preserves_nonfinite_blocks():
    values = np.full((2, 6, 8), 1.0, np.float64)
    values[0, :2, :2] = [[np.inf, 1], [1, 1]]
    values[0, 2:4, :2] = [[np.inf, -np.inf], [1, 1]]
    values[0, 4:, :2] = [[np.nan, 1], [1, 1]]
    # These four samples average to a different float32 if cast only after reduction.
    values[1, :2, :2] = [[1 + 2**-24, 1 + 2**-24], [1 + 2**-24, 1 + 3 * 2**-24]]
    with jax.enable_x64(), np.errstate(invalid="ignore"):
        actual = _scan(jax.device_put(values)).binned(2).projections
        expected = _scan(values).binned(2).projections
    np.testing.assert_array_equal(actual, expected)


def test_binning_factor_one_keeps_the_original_device_scan(monkeypatch):
    original = _scan(jax.device_put(np.ones((5, 13, 17), np.float32)))

    def forbidden(*args, **kwargs):
        pytest.fail("factor one must not download pixels")

    monkeypatch.setattr(type(original.projections), "__array__", forbidden)
    assert original.binned(1) is original


@pytest.mark.parametrize("direction", ["forward", "reverse"])
@pytest.mark.parametrize("committed", [False, True])
def test_device_binning_ignores_an_unrelated_default_device(monkeypatch, direction, committed):
    devices = jax.devices()
    if len(devices) < 2:
        devices = [*devices, *jax.devices("cpu")]
    if len(set(devices)) < 2:
        pytest.skip("needs two devices")
    if direction == "reverse":
        devices = devices[::-1]
    values = np.arange(5 * 13 * 17, dtype=np.float32).reshape(5, 13, 17)
    if committed:
        data = jax.device_put(values, devices[-1])
    else:
        with jax.default_device(devices[-1]):
            data = jnp.asarray(values)
    assert data.committed == committed
    downloads = _capture_downloads(monkeypatch)
    with jax.default_device(devices[0]):
        result = _scan(data).binned(2)
    assert downloads == [((5, 6, 8), np.dtype("float32"), {devices[-1]})]
    assert result.projections.flags.writeable


def test_binning_sharded_views_keeps_their_devices(monkeypatch):
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("needs two devices")
    mesh = jax.sharding.Mesh(np.asarray(devices[:2]), ("view",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("view"))
    values = np.arange(6 * 13 * 17, dtype=np.float32).reshape(6, 13, 17)
    data = jax.device_put(values, sharding)
    downloads = _capture_downloads(monkeypatch)
    actual = _scan(data).binned(3)
    assert downloads == [((6, 4, 5), np.dtype("float32"), set(devices[:2]))]
    expected = values[:, :12, :15].reshape(6, 4, 3, 5, 3).mean(axis=(2, 4))
    np.testing.assert_allclose(actual.projections, expected, rtol=2e-7)
