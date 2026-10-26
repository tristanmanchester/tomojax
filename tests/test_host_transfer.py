"""Host/device copy ownership and nonuniform storage layouts."""

from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.core.host_transfer import copy_to_host, upload_host, upload_host_views


@pytest.mark.parametrize(
    "dtype", [np.bool_, np.uint8, np.int32, np.float32, np.float64, np.complex64]
)
def test_upload_outputs_are_independent_and_use_jax_dtype(dtype):
    source = (np.arange(48).reshape(4, 4, 3) / 7).astype(dtype)
    expected = source.astype(jax.dtypes.canonicalize_dtype(source.dtype), copy=True)
    value = upload_host(source, jax.devices()[0])
    other = upload_host(np.zeros_like(source), jax.devices()[0])
    assert value.unsafe_buffer_pointer() != other.unsafe_buffer_pointer()
    source.fill(0)
    assert value.dtype == expected.dtype
    np.testing.assert_array_equal(value, expected)
    np.testing.assert_array_equal(other, np.zeros_like(expected))


def test_upload_preserves_float32_bits():
    bits = np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC1ABCD], np.uint32)
    value = upload_host(bits.view(np.float32), jax.devices()[0])
    np.testing.assert_array_equal(np.asarray(value).view(np.uint32), bits)


@pytest.mark.parametrize("source_order", ["C", "F", "strided"])
@pytest.mark.parametrize("target_order", ["C", "F", "strided"])
def test_host_transfers_preserve_owned_views(source_order, target_order):
    device = jax.devices()[0]
    base = np.arange(11 * 13 * 9, dtype=np.float32).reshape(11, 13, 9) / 100
    source = base[::2, 1::2, ::2] if source_order == "strided" else base.copy(order=source_order)
    source.setflags(write=False)
    storage = np.full(tuple(2 * n for n in source.shape), -17, np.float32)
    target = (
        storage[::2, ::2, ::2]
        if target_order == "strided"
        else np.empty_like(source, order=target_order)
    )
    scope = nullcontext()
    if device.platform == "gpu":
        import cupy as cp

        scope = cp.cuda.Stream(non_blocking=True)
    with scope:
        value = upload_host(source, device)
        result = jax.jit(lambda x: x * jnp.float32(1.000001))(value)
        copy_to_host(result, target)
    np.testing.assert_array_equal(target, source * np.float32(1.000001))
    np.testing.assert_array_equal(np.asarray(value), source)
    np.testing.assert_array_equal(storage[1::2], np.full_like(storage[1::2], -17))
    expected = np.asarray(value).copy()
    source.setflags(write=True)
    source[:] = -5
    np.testing.assert_array_equal(value, expected)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_host_transfers_copy_pitched_windows_without_touching_padding(axis):
    base = np.arange(7 * 9 * 13, dtype=np.float32).reshape(7, 9, 13) / 100
    window = tuple(slice(2, -2) if k == axis else slice(None) for k in range(3))
    target = np.full_like(base, -19)
    value = upload_host(base[window], jax.devices()[0])
    copy_to_host(value * jnp.float32(1.000001), target[window])
    expected = np.full_like(base, -19)
    expected[window] = base[window] * np.float32(1.000001)
    np.testing.assert_array_equal(target, expected)


@pytest.mark.parametrize("layout", ["C", "F", "permuted", "strided"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_indexed_host_upload_preserves_view_order_and_dtype_conversion(layout, dtype):
    source = np.arange(9 * 11 * 7, dtype=dtype).reshape(9, 11, 7) / 100
    if layout in {"C", "F"}:
        source = source.copy(order=layout)
    else:
        source = source.transpose(1, 0, 2)[1::2] if layout == "permuted" else source[:, :, ::2]
    source.setflags(write=False)
    indices = np.array([len(source) - 1, 0, 2, 0])
    value = upload_host_views(source, indices, jax.devices()[0])
    np.testing.assert_array_equal(np.asarray(value), source[indices].astype(np.float32))


@pytest.mark.parametrize("shape", [(), (0, 3), (7, 5, 3), (2, 3, 4, 5)])
@pytest.mark.parametrize("layout", ["C", "F", "reversed"])
def test_download_casting_and_transposed_device_layout(shape, layout):
    source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    value = jax.jit(lambda x: x.T)(jnp.asarray(source))
    expected = source.T
    target = np.empty(expected.shape, dtype=np.float64, order="F" if layout == "F" else "C")
    if layout == "reversed" and shape:
        target = target[::-1]
    copy_to_host(value, target, casting="safe")
    np.testing.assert_array_equal(target, expected)
    with pytest.raises(TypeError):
        copy_to_host(value, target)
    np.testing.assert_array_equal(np.asarray(value), expected)


@pytest.mark.gpu
def test_download_layout_conversion_crosses_scratch_tile_boundary():
    source = np.broadcast_to(np.arange(257, dtype=np.float32), (257, 257, 257))
    value = jax.jit(lambda x: x.transpose(2, 0, 1))(jnp.asarray(source))
    target = np.empty(source.shape, np.float32, order="F")[::-1]
    copy_to_host(value, target)
    np.testing.assert_array_equal(target, source.transpose(2, 0, 1))


def test_download_broadcasts_without_modifying_host_padding():
    source = jnp.arange(5, dtype=jnp.float32)
    target = np.full((3, 2, 5), -1, dtype=np.float32)
    copy_to_host(source, target[:, 0])
    np.testing.assert_array_equal(target[:, 0], np.broadcast_to(np.arange(5), (3, 5)))
    np.testing.assert_array_equal(target[:, 1], -1)


def test_leading_axis_slab_upload_owns_partial_final_plane_block():
    from tomojax.core.host_transfer import upload_host_slabs

    shape = (7, 3, 5)
    words = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape) + 0x3F800000
    words[3, 1, :3] = [0x7FC1ABCD, 0x80000000, 0]
    source = words.view(np.float32)

    def slabs():
        for first in range(0, shape[0], 3):
            yield first, source[first : first + 3]

    actual = upload_host_slabs(shape, slabs(), jax.devices()[0])
    source.fill(0)
    expected = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape) + 0x3F800000
    expected[3, 1, :3] = [0x7FC1ABCD, 0x80000000, 0]
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint32), expected)
