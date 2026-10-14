"""Resident per-view corrections avoid a whole host mirror before the first step."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import BeamHardening, Correction, RejectViews, Stripes, Zingers


def _geometry():
    return tj.ParallelGeometry(
        tj.Grid(11, 11, 9, 1, 1, 1),
        tj.Detector(11, 9, 0.7, 1.1, center=(0.2, -0.3)),
        np.linspace(0, 180, 13),
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int16])
@pytest.mark.parametrize("batch", [3, 20])
def test_device_line_prefix_downloads_only_corrected_float32_batches(dtype, batch, monkeypatch):
    with jax.enable_x64():
        values = np.linspace(0, 5, 13 * 9 * 11).reshape(13, 9, 11).astype(dtype)
        data = jax.device_put(values)
        step = BeamHardening((1, 0.05))
        expected = tj.Scan(values, _geometry()).corrected(step, batch_views=batch)
        original = tj.Scan(data, _geometry(), corrections=(Correction("imported", {}, {}),))
        downloads, scheduled = [], []
        convert, asarray, download = type(data).__array__, np.asarray, type(data).copy_to_host_async

        def tracked(array, *args, **kwargs):
            assert array is not data, "the initial whole input must not be materialized"
            return convert(array, *args, **kwargs)

        def collected(array, *args, **kwargs):
            if isinstance(array, jax.Array):
                downloads.append((tuple(array.shape), array.dtype))
            return asarray(array, *args, **kwargs)

        def asynchronous(array):
            scheduled.append(tuple(array.shape))
            return download(array)

        with monkeypatch.context() as patch:
            patch.setattr(type(data), "__array__", tracked)
            patch.setattr(np, "asarray", collected)
            patch.setattr(type(data), "copy_to_host_async", asynchronous)
            actual = original.corrected(step, batch_views=batch)
        shapes = [(min(batch, 13 - start), 9, 11) for start in range(0, 13, batch)]
        assert downloads == [(shape, np.dtype(np.float32)) for shape in shapes]
        assert scheduled == shapes
        np.testing.assert_array_equal(actual.projections, expected.projections)
        assert actual.projections.dtype == np.float32 and actual.projections.flags.writeable
        assert actual.corrections == (*original.corrections, *expected.corrections)
        assert actual.geometry is original.geometry
        actual.projections[0, 0, 0] = -20
        np.testing.assert_array_equal(np.asarray(data), values)


@pytest.mark.parametrize("tail", ["whole_rows", "selector", "both"])
def test_resident_prefix_preserves_tail_values_views_and_records(tail):
    values = np.random.default_rng(42).uniform(0.5, 0.6, (13, 9, 11)).astype(np.float32)
    values[6] += 2
    steps = [BeamHardening((1, 0.05))]
    if tail in ("whole_rows", "both"):
        steps += [Stripes(width=3), BeamHardening((0.9,))]
    if tail in ("selector", "both"):
        steps += [RejectViews(), BeamHardening((1.1,))]
    expected = tj.Scan(values, _geometry()).corrected(*steps, batch_views=3)
    actual = tj.Scan(jax.device_put(values), _geometry()).corrected(*steps, batch_views=3)
    np.testing.assert_array_equal(actual.projections, expected.projections)
    np.testing.assert_array_equal(actual.angles, expected.angles)
    assert actual.corrections == expected.corrections
    assert actual.detector == expected.detector
    assert actual.grid == expected.grid


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_resident_line_prefix_preserves_existing_input_device(committed, reverse, monkeypatch):
    devices = (jax.devices("cpu")[-1], jax.devices()[0])
    if devices[0] == devices[1]:
        pytest.skip("requires a second CPU device or an accelerator")
    resident, ambient = devices[::-1] if reverse else devices
    values = np.arange(13 * 9 * 11, dtype=np.float32).reshape(13, 9, 11) / 100
    with jax.default_device(resident):
        data = jnp.asarray(values)
    if committed:
        data = jax.device_put(data, resident)
    original = tj.Scan(data, _geometry())
    convert, downloaded = np.asarray, []

    def tracked(array, *args, **kwargs):
        if isinstance(array, jax.Array):
            downloaded.append(array.devices())
        return convert(array, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(np, "asarray", tracked)
        with jax.default_device(ambient):
            result = original.corrected(BeamHardening((1, 0.05)), batch_views=3)
    assert downloaded == [{resident}] * 5
    assert data.devices() == {resident} and data.committed == committed
    expected = values + np.float32(0.05) * values**2
    np.testing.assert_allclose(result.projections, expected, rtol=2e-7)


@pytest.mark.parametrize("kind", ["numpy", "device"])
def test_line_correction_wrong_domain_still_fails_before_download(kind, monkeypatch):
    values = np.full((13, 9, 11), 0.5, np.float32)
    data = jax.device_put(values) if kind == "device" else values
    original = tj.Scan(data, _geometry())
    if kind == "device":

        def forbidden(*args, **kwargs):
            raise AssertionError("invalid domain must be rejected before reading")

        monkeypatch.setattr(type(data), "__array__", forbidden)
    with pytest.raises(ValueError, match="transmission"):
        original.corrected(Zingers())


@pytest.mark.parametrize("batch", [1, 3, 20])
def test_resident_prefix_releases_each_processed_batch_before_the_next(batch, monkeypatch):
    import weakref

    data = jax.device_put(np.ones((13, 9, 11), np.float32))
    scheduled = type(data).copy_to_host_async
    live = []

    def tracked(array):
        live.append(weakref.ref(array))
        assert sum(ref() is not None for ref in live) == 1
        return scheduled(array)

    monkeypatch.setattr(type(data), "copy_to_host_async", tracked)
    result = tj.Scan(data, _geometry()).corrected(BeamHardening((1, 0.05)), batch_views=batch)
    np.testing.assert_allclose(result.projections, 1.05)
    assert not any(ref() is not None for ref in live)


def test_resident_prefix_preserves_named_sharding_and_every_source_buffer(monkeypatch):
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    devices = jax.devices("cpu")
    values = np.linspace(0.1, 2.0, 16 * 9 * 11, dtype=np.float32).reshape(16, 9, 11)
    sharding = NamedSharding(Mesh(np.asarray(devices), ("views",)), PartitionSpec("views"))
    data = jax.device_put(values, sharding)
    pointers = [shard.data.unsafe_buffer_pointer() for shard in data.addressable_shards]
    geometry = tj.ParallelGeometry(_geometry().grid, _geometry().detector, np.arange(16))
    with jax.default_device(devices[0]):
        expected = tj.Scan(values, geometry).corrected(BeamHardening((1, 0.05)), batch_views=3)
    convert = type(data).__array__

    def tracked(array, *args, **kwargs):
        assert array is not data, "the original distributed stack must not be downloaded"
        return convert(array, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(type(data), "__array__", tracked)
        result = tj.Scan(data, geometry).corrected(BeamHardening((1, 0.05)), batch_views=3)
    assert data.sharding == sharding
    assert [shard.data.unsafe_buffer_pointer() for shard in data.addressable_shards] == pointers
    np.testing.assert_array_equal(result.projections, expected.projections)
