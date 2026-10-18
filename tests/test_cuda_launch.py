"""CUDA launch context, JAX buffer ownership and persistent compilation."""

from concurrent.futures import ThreadPoolExecutor

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.core import cuda

pytestmark = pytest.mark.gpu
_SOURCE = """extern "C" __global__ void add(const float* x,float* y,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) y[i]=x[i]+INCREMENT;
}"""


def _operation(increment=1):
    source = _SOURCE.replace("INCREMENT", f"{increment}.f")

    def launch(context, out, value):
        cuda.launch_kernel(
            source,
            "add",
            stream=context.stream,
            grid=(2,),
            block=(128,),
            arguments=(value, out, np.int32(value.shape[0])),
        )

    add = buffer_callback(
        launch, jax.ShapeDtypeStruct((129,), jnp.float32), command_buffer_compatible=True
    )
    return jax.jit(lambda x: jax.lax.fori_loop(0, 20, lambda _, y: add(y), x))


def test_launch_preserves_buffers_and_context_across_threads():
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    operation = _operation()
    for device in jax.devices():

        def run(value, device=device):
            with jax.default_device(device):
                data = jnp.full((129,), value, jnp.float32)
                actual = operation(data)
                np.testing.assert_array_equal(actual, np.full(129, value + 20))
                np.testing.assert_array_equal(data, np.full(129, value))
                assert actual.devices() == {device}

        run(0)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(run, range(1, 5)))


def test_persistent_binary_is_reused_and_source_changes_recompile(tmp_path, monkeypatch):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    monkeypatch.setenv("TOMOJAX_JAX_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TOMOJAX_JAX_CACHE", raising=False)
    compile_source = cuda._compile_source
    calls = []

    def compile_and_record(*args):
        calls.append(args)
        return compile_source(*args)

    monkeypatch.setattr(cuda, "_compile_source", compile_and_record)
    for increment, expected_compiles in [(2, 1), (2, 1), (3, 2)]:
        cuda._loaded.cache_clear()
        cuda._compiled_binary.cache_clear()
        result = _operation(increment)(jnp.zeros(129, jnp.float32))
        np.testing.assert_array_equal(result, np.full(129, 20 * increment))
        assert len(calls) == expected_compiles
