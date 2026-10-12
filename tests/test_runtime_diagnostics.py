from __future__ import annotations

from dataclasses import dataclass
import logging
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.alignment._objectives import fixed_volume

# check-public-imports: allow-private
from tomojax.backends import _memory, estimate_views_per_batch_info
from tomojax.geometry import Detector, Grid


@dataclass(frozen=True)
class _Device:
    platform: str = "gpu"
    local_hardware_id: int = 0
    compute_capability: object = None


def test_device_memory_probe_logs_resolution_failure_as_unknown(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_devices(_kind: str) -> list[object]:
        raise RuntimeError("device probe unavailable")

    monkeypatch.setattr(jax, "devices", raise_devices)
    monkeypatch.setattr(_memory, "_host_available_memory_bytes", lambda: 64 * 1024**3)

    with caplog.at_level(logging.DEBUG, logger="tomojax.backends._memory"):
        free_bytes = _memory.device_free_memory_bytes()

    assert free_bytes is None
    assert "Device memory probe failed" in caplog.text
    assert "device probe unavailable" in caplog.text


@pytest.mark.parametrize("platform", ["gpu", "tpu"])
@pytest.mark.parametrize("api", ["memory_stats", "device_get_memory_info"])
@pytest.mark.parametrize("failure", ["missing", "empty", "unrecognized", "error"])
def test_unknown_accelerator_memory_does_not_substitute_host_ram(
    monkeypatch, platform, api, failure
):
    host_calls = []
    selected = SimpleNamespace(platform=platform)

    def probe(*args):
        del args
        if failure == "error":
            raise RuntimeError("accelerator statistics unavailable")
        return None if failure == "empty" else {"peak_bytes_in_use": 1234}

    def host():
        host_calls.append(True)
        return 64 * 1024**3

    monkeypatch.setattr(jax, "config", SimpleNamespace(jax_default_device=selected))
    monkeypatch.delattr(jax, "device_get_memory_info", raising=False)
    monkeypatch.setattr(_memory, "_host_available_memory_bytes", host)
    if failure != "missing":
        if api == "memory_stats":
            selected.memory_stats = probe
        else:
            monkeypatch.setattr(jax, "device_get_memory_info", probe, raising=False)
    assert _memory.device_free_memory_bytes() is None
    estimate = estimate_views_per_batch_info(
        n_views=32, grid_nxyz=(4, 4, 4), det_nuv=(4, 4), fallback_batch=3
    )
    assert estimate.free_bytes is None and estimate.fallback_used
    assert estimate.views_per_batch == 3
    assert not host_calls


def test_failed_preferred_accelerator_probe_tries_its_remaining_statistics(monkeypatch):
    selected = SimpleNamespace(platform="gpu", memory_stats=lambda: {"free_bytes": 1234})
    monkeypatch.setattr(jax, "config", SimpleNamespace(jax_default_device=selected))

    def failed(device):
        assert device is selected
        raise RuntimeError("preferred API unavailable")

    monkeypatch.setattr(jax, "device_get_memory_info", failed, raising=False)
    assert _memory.device_free_memory_bytes() == 1234


def test_accelerator_zero_free_memory_is_not_replaced_by_another_probe(monkeypatch):
    selected = SimpleNamespace(platform="gpu", memory_stats=lambda: {"free_bytes": 1234})
    monkeypatch.setattr(jax, "config", SimpleNamespace(jax_default_device=selected))
    monkeypatch.setattr(jax, "device_get_memory_info", lambda _: {"free_bytes": 0}, raising=False)
    assert _memory.device_free_memory_bytes() == 0


def test_memory_probe_without_any_discovered_device_uses_host_memory(monkeypatch):
    monkeypatch.setattr(jax, "config", SimpleNamespace(jax_default_device="cpu"))
    monkeypatch.setattr(jax, "devices", lambda _: [])
    monkeypatch.setattr(_memory, "_host_available_memory_bytes", lambda: 5678)
    assert _memory.device_free_memory_bytes() == 5678


@pytest.mark.parametrize("api", ["memory_stats", "device_get_memory_info"])
def test_memory_probe_honors_an_explicit_default_device(monkeypatch, api):
    wrong = SimpleNamespace(platform="gpu", memory_stats=lambda: {"free_bytes": 9000})
    selected = SimpleNamespace(platform="gpu", memory_stats=lambda: {"free_bytes": 1234})
    monkeypatch.setattr(jax, "config", SimpleNamespace(jax_default_device=selected))
    monkeypatch.setattr(jax, "devices", lambda *_: [wrong, selected])
    if api == "device_get_memory_info":

        def probe(device):
            assert device is selected
            return {"free_bytes": 1234}

        monkeypatch.setattr(jax, "device_get_memory_info", probe, raising=False)
    else:
        monkeypatch.delattr(jax, "device_get_memory_info", raising=False)
    assert _memory.device_free_memory_bytes() == 1234


@pytest.mark.parametrize("default", [None, "cpu", "gpu", "tpu"])
def test_memory_probe_uses_the_requested_platform_or_default_backend(monkeypatch, default):
    requested = []

    def devices(platform):
        requested.append(platform)
        return [SimpleNamespace(platform=platform, memory_stats=lambda: {"free_bytes": 1234})]

    monkeypatch.setattr(jax, "config", SimpleNamespace(jax_default_device=default))
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(jax, "devices", devices)
    monkeypatch.setattr(_memory, "_host_available_memory_bytes", lambda: 5678)
    monkeypatch.delattr(jax, "device_get_memory_info", raising=False)
    assert _memory.device_free_memory_bytes() == (5678 if default == "cpu" else 1234)
    assert requested == [default or "tpu"]


def test_memory_probe_does_not_initialize_gpu_inside_cpu_default_device(monkeypatch):
    device = next(iter(jax.devices("cpu")))

    def forbidden(*args):
        raise AssertionError("Explicit CPU placement must not initialize another backend")

    monkeypatch.setattr(jax, "devices", forbidden)
    monkeypatch.setattr(_memory, "_host_available_memory_bytes", lambda: 5678)
    with jax.default_device(device):
        assert _memory.device_free_memory_bytes() == 5678


def test_invalid_views_per_batch_env_logs_selected_default(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOMOJAX_MAX_VIEWS_PER_BATCH", "not-an-int")

    with caplog.at_level(logging.DEBUG, logger="tomojax.backends._memory"):
        estimate = estimate_views_per_batch_info(
            n_views=16,
            grid_nxyz=(4, 4, 4),
            det_nuv=(4, 4),
            free_bytes_override=1_000_000_000,
            fallback_batch=8,
        )

    assert estimate.views_per_batch == 8
    assert estimate.fallback_used is False
    assert "Ignoring invalid TOMOJAX_MAX_VIEWS_PER_BATCH value" in caplog.text


def test_gpu_compute_capability_probe_logs_subprocess_failure(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_probe(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("nvidia-smi missing")

    _memory._gpu_compute_capability.cache_clear()
    monkeypatch.setattr(_memory, "check_output_resolved_command", raise_probe)

    with caplog.at_level(logging.DEBUG, logger="tomojax.backends._memory"):
        capability = _memory._gpu_compute_capability(_Device())

    assert capability is None
    assert "CUDA compute capability probe failed" in caplog.text
    assert "nvidia-smi missing" in caplog.text


@pytest.mark.parametrize("capability", ["7.5", (7, 5)])
def test_gather_capability_uses_the_selected_device_without_subprocess(monkeypatch, capability):
    def forbidden(*args, **kwargs):
        raise AssertionError("Known device capability must not query another GPU")

    monkeypatch.setattr(_memory, "check_output_resolved_command", forbidden)
    assert _memory._gpu_compute_capability(_Device(compute_capability=capability)) == (7, 5)


@pytest.mark.parametrize(
    ("visible", "identifier"), [(None, "1"), ("3,2", "2"), ("GPU-a,GPU-b", "GPU-b")]
)
def test_gather_capability_subprocess_addresses_the_selected_visible_device(
    monkeypatch, visible, identifier
):
    def probe(command, **kwargs):
        assert f"--id={identifier}" in command
        return "7.5\n"

    _memory._gpu_compute_capability.cache_clear()
    monkeypatch.setattr(_memory, "check_output_resolved_command", probe)
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert _memory._gpu_compute_capability(_Device(local_hardware_id=1)) == (7, 5)


@pytest.mark.parametrize(
    ("platform", "capability", "expected"),
    [
        ("cpu", None, "fp32"),
        ("tpu", None, "bf16"),
        ("gpu", (7, 5), "fp16"),
        ("gpu", (8, 0), "bf16"),
    ],
)
def test_gather_default_honors_explicit_device_instead_of_global_gpu0(
    monkeypatch, platform, capability, expected
):
    selected = _Device(platform=platform, local_hardware_id=1, compute_capability=capability)
    monkeypatch.setattr(jax, "config", SimpleNamespace(jax_default_device=selected))

    def forbidden(*args):
        raise AssertionError("Explicit default device must not list a different backend")

    def supported(device, dtype):
        assert device is selected
        return True

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(jax, "devices", forbidden)
    monkeypatch.setattr(_memory, "_device_supports_dtype", supported)
    assert _memory.default_gather_dtype() == expected


@pytest.mark.parametrize("platform", ["cpu", "gpu", "tpu"])
def test_gather_default_honors_platform_default_device(monkeypatch, platform):
    monkeypatch.setattr(jax, "config", SimpleNamespace(jax_default_device=platform))
    selected = _Device(platform=platform, compute_capability=(8, 0))

    def devices(requested):
        assert requested == platform
        return [selected]

    monkeypatch.setattr(jax, "devices", devices)
    monkeypatch.setattr(_memory, "_device_supports_dtype", lambda *_: True)
    assert _memory.default_gather_dtype() == ("fp32" if platform == "cpu" else "bf16")


def test_gather_jit_probe_and_its_cache_are_scoped_to_the_selected_device(monkeypatch):
    from contextlib import contextmanager

    first, second = _Device(local_hardware_id=0), _Device(local_hardware_id=1)
    active, compiled = [], []

    @contextmanager
    def default_device(device):
        active.append(device)
        yield
        active.pop()

    def jit(fn):
        def run(array):
            compiled.append(active[-1])
            if active[-1] is second:
                raise RuntimeError("dtype unsupported on this device")
            return SimpleNamespace(block_until_ready=lambda: None)

        return run

    _memory._device_supports_dtype.cache_clear()
    monkeypatch.setattr(jax, "default_device", default_device)
    monkeypatch.setattr(jax, "jit", jit)
    monkeypatch.setattr(jnp, "ones", lambda *_args, **_kwargs: None)
    for _ in range(2):
        assert _memory._device_supports_dtype(first, "bfloat16")
        assert not _memory._device_supports_dtype(second, "bfloat16")
    assert compiled == [first, second]


def test_gather_default_cpu_context_does_not_initialize_another_backend(monkeypatch):
    device = jax.devices("cpu")[0]

    def forbidden(*args):
        raise AssertionError("CPU default must neither probe nor use reduced GPU gathers")

    monkeypatch.setattr(jax, "devices", forbidden)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(_memory, "_device_supports_dtype", forbidden)
    with jax.default_device(device):
        assert _memory.default_gather_dtype() == "fp32"


def test_simulation_cpu_context_keeps_full_projection_precision_on_a_gpu_host(monkeypatch):
    from tomojax.datasets import SimConfig, simulate

    cpu = jax.devices("cpu")[0]
    config = SimConfig(
        nx=6, ny=7, nz=5, nu=9, nv=8, n_views=5, phantom="blobs", single_rotate=False
    )
    with jax.default_device(cpu):
        with monkeypatch.context() as reference_settings:
            reference_settings.setitem(simulate.__globals__, "default_gather_dtype", lambda: "fp32")
            reference = simulate(config)["projections"]
        monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
        monkeypatch.setattr(_memory, "_gpu_compute_capability", lambda *_: (8, 0))
        monkeypatch.setattr(_memory, "_device_supports_dtype", lambda *_: True)
        data = simulate(config)
    assert data["projections"].devices() == {cpu}
    np.testing.assert_allclose(data["projections"], reference, rtol=1e-6, atol=1e-6)


def test_pallas_support_probe_logs_debug_before_jax_fallback(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_support(*_args: object, **_kwargs: object) -> str | None:
        raise RuntimeError("support probe failed")

    monkeypatch.setattr(
        fixed_volume,
        "resolve_pallas_callable",
        lambda *_args, **_kwargs: (raise_support, None),
    )

    with caplog.at_level(
        logging.DEBUG,
        logger="tomojax.alignment._objectives.fixed_volume",
    ):
        reason = fixed_volume._pallas_sinogram_fallback_reason(
            pose_stack=jnp.eye(4, dtype=jnp.float32)[None, :, :],
            grid=Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0),
            detector=Detector(nu=2, nv=2, du=1.0, dv=1.0),
            volume=jnp.ones((2, 2, 2), dtype=jnp.float32),
            det_grid=(
                jnp.zeros((4,), dtype=jnp.float32),
                jnp.zeros((4,), dtype=jnp.float32),
            ),
            gather_dtype="fp32",
        )

    assert reason == "RuntimeError: support probe failed"
    assert "Pallas sinogram support probe failed" in caplog.text
