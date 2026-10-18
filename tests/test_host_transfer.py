"""Host/device copy ownership and nonuniform storage layouts."""

from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.core.host_transfer import copy_to_host, upload_host


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
