"""Shared JAX error classification, independent of reconstruction recovery policy."""

from __future__ import annotations

import pytest

from tomojax.backends.api import is_oom_error


@pytest.mark.parametrize(
    "message",
    [
        "RESOURCE_EXHAUSTED: allocation failed",
        "CUDA_ERROR_OUT_OF_MEMORY: out of memory",
        "INTERNAL: RET_CHECK failure (fft_thunk.cc:201) fft_plan != nullptr "
        "Failed to create cuFFT batched plan with scratch allocator "
        "[executable_name='jit_gpu_metric']",
    ],
)
@pytest.mark.parametrize("upper_case", [False, True])
def test_jax_allocation_errors_can_select_bounded_recovery(message, upper_case):
    assert is_oom_error(RuntimeError(message.upper() if upper_case else message))


@pytest.mark.parametrize(
    "message",
    [
        "INTERNAL: unrelated compiler failure",
        "INTERNAL: fft_plan != nullptr Failed to create cuFFT plan",
        "INTERNAL: Failed to create cuFFT batched plan with scratch allocator",
        "INTERNAL: allocator configuration is invalid",
        "INVALID_ARGUMENT: bad FFT dimensions",
        "CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered",
        "cuFFT internal error",
        "",
    ],
)
def test_unrelated_errors_do_not_select_recovery(message):
    assert not is_oom_error(RuntimeError(message))
