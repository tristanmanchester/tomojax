"""CUDA launch context, JAX buffer ownership and persistent compilation."""

from concurrent.futures import ThreadPoolExecutor

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.core import cuda

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


@pytest.mark.gpu
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


@pytest.mark.gpu
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


@pytest.mark.gpu
def test_initialization_copies_or_zeros_without_mutating_live_inputs():
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    for zero in (False, True):
        for alias in (False, True):

            def launch(context, out, value, zero=zero):
                cuda.initialize_buffer(out, stream=context.stream, source=None if zero else value)

            operation = jax.jit(
                buffer_callback(
                    launch,
                    jax.ShapeDtypeStruct((129,), jnp.float32),
                    input_output_aliases={0: 0} if alias else {},
                    command_buffer_compatible=True,
                )
            )
            for value in (1, 3):
                data = jnp.full(129, value, jnp.float32)
                np.testing.assert_array_equal(operation(data), np.full(129, 0 if zero else value))
                np.testing.assert_array_equal(data, np.full(129, value))


@pytest.mark.parametrize("library", ["libcuda.so.1", "libnvrtc.so.12"])
def test_missing_cuda_dependency_is_actionable(monkeypatch, library):
    def load(path):
        if path == library:
            raise OSError("missing library")
        return object()

    monkeypatch.setattr(cuda.C, "CDLL", load)
    monkeypatch.setattr(cuda, "find_spec", lambda _: None)
    remedy = "NVIDIA driver" if library == "libcuda.so.1" else r"tomojax\[cuda12\]"
    with pytest.raises(RuntimeError, match=remedy) as error:
        cuda._libraries.__wrapped__()
    assert isinstance(error.value.__cause__, OSError)
