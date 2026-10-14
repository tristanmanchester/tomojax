"""Selection commits existing input placement, not a caller's unrelated default."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax._typed_arrays import commit_array


def test_commit_array_leaves_host_and_lazy_inputs_untouched():
    class Lazy:
        def __array__(self):
            raise AssertionError("placement must not read a lazy input")

    for data in (np.arange(17), Lazy(), None):
        assert commit_array(data) is data


@pytest.mark.parametrize("committed", [False, True])
def test_commit_array_preserves_pixels_sharding_and_buffer(committed):
    original = jnp.arange(17, dtype=jnp.float32)
    if committed:
        original = jax.device_put(original, original.sharding)
    placed = commit_array(original)
    assert placed.committed
    assert placed.sharding == original.sharding
    assert placed.unsafe_buffer_pointer() == original.unsafe_buffer_pointer()
    assert original.committed == committed
    np.testing.assert_array_equal(placed, np.arange(17))


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("operation", ["crop", "select"])
@pytest.mark.parametrize("source", ["scan", "frames"])
def test_selection_preserves_input_residency_across_default_contexts(
    source, operation, committed, reverse, monkeypatch
):
    cpu = jax.devices("cpu")
    devices = (cpu[-1], jax.devices()[0])
    if devices[0] == devices[1]:
        pytest.skip("needs a second CPU device or an accelerator")
    resident, ambient = devices[::-1] if reverse else devices
    values = np.arange(13 * 9 * 11, dtype=np.float32).reshape(13, 9, 11)
    with jax.default_device(resident):
        data = jnp.asarray(values)
    if committed:
        data = jax.device_put(data, resident)
    geometry = tj.ParallelGeometry(
        tj.Grid(11, 11, 9, 1.0, 1.0, 1.0),
        tj.Detector(11, 9, 0.7, 1.1),
        [i * 180 / 13 for i in range(13)],
    )
    original = tj.Scan(data, geometry) if source == "scan" else tj.Frames(data, geometry)
    downloads = []
    asarray = np.asarray

    def tracked(value, *args, **kwargs):
        if isinstance(value, jax.Array):
            downloads.append((value.devices(), tuple(value.shape)))
        return asarray(value, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", tracked)
    with jax.default_device(ambient):
        if operation == "crop":
            result = original.cropped(slice(2, 5), slice(3, 8))
            expected = values[:, 2:5, 3:8]
        else:
            result = original.selected([1, 5, 9])
            expected = values[[1, 5, 9]]
    assert downloads == [({resident}, expected.shape)]
    assert data.devices() == {resident} and data.committed == committed
    actual = result.projections if isinstance(result, tj.Scan) else result.counts
    np.testing.assert_array_equal(actual, expected)


def test_commit_array_preserves_named_sharding_and_every_buffer():
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    values = np.arange(16, dtype=np.float32)
    sharding = NamedSharding(
        Mesh(np.asarray(jax.devices("cpu")), ("items",)), PartitionSpec("items")
    )
    original = jax.device_put(values, sharding)
    placed = commit_array(original)
    assert placed.sharding == sharding
    for before, after in zip(original.addressable_shards, placed.addressable_shards, strict=True):
        assert before.data is not None and after.data is not None
        assert before.data.unsafe_buffer_pointer() == after.data.unsafe_buffer_pointer()
    np.testing.assert_array_equal(placed, values)
