"""Producer-device budgets cannot be replaced by a caller or physical memory budget."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import pytest

from tomojax.backends import device_free_memory_bytes


@pytest.mark.parametrize("free", [0, 300])
@pytest.mark.parametrize("mixed", [False, True])
def test_explicit_allocator_limit_wins_over_physical_free(free, mixed, monkeypatch):
    stats = {"bytes_limit": 1000, "bytes_in_use": 1000 - free}
    if mixed:
        stats["free"] = 9000
    device = SimpleNamespace(platform="gpu", memory_stats=lambda: stats)

    def forbidden(*args, **kwargs):
        raise AssertionError("known allocator headroom must not probe a physical budget")

    monkeypatch.setattr(jax, "devices", forbidden)
    monkeypatch.setattr(jax, "device_get_memory_info", forbidden, raising=False)
    assert device_free_memory_bytes(device=device) == free


@pytest.mark.parametrize("failure", [False, True])
def test_explicit_unknown_gpu_budget_does_not_use_host_or_current_device(failure, monkeypatch):
    def unavailable():
        if failure:
            raise RuntimeError("statistics unavailable")

    cpu = jax.devices("cpu")[0]
    device = SimpleNamespace(platform="gpu", memory_stats=unavailable)
    monkeypatch.delattr(jax, "device_get_memory_info", raising=False)
    monkeypatch.setattr(
        jax, "devices", lambda *args: pytest.fail("explicit device must not discover")
    )
    with jax.default_device(cpu):
        assert device_free_memory_bytes(device=device) is None


def test_explicit_driver_fallback_queries_only_the_supplied_device(monkeypatch):
    device = SimpleNamespace(platform="gpu", memory_stats=lambda: None)
    seen = []

    def info(value):
        seen.append(value)
        return (123, 456)

    monkeypatch.setattr(jax, "device_get_memory_info", info, raising=False)
    monkeypatch.setattr(
        jax, "devices", lambda *args: pytest.fail("explicit device must not discover")
    )
    assert device_free_memory_bytes(device=device) == 123
    assert seen == [device]


def test_explicit_cpu_memory_does_not_initialize_or_probe_accelerators(monkeypatch):
    cpu = jax.devices("cpu")[0]
    expected = device_free_memory_bytes(device=cpu)
    monkeypatch.setattr(
        jax, "devices", lambda *args: pytest.fail("explicit device must not discover")
    )
    monkeypatch.setattr(
        jax,
        "device_get_memory_info",
        lambda *args: pytest.fail("CPU uses host memory"),
        raising=False,
    )
    actual = device_free_memory_bytes(device=cpu)
    assert expected is None or (isinstance(actual, int) and actual >= 0)


def test_invalid_allocator_limit_remains_unknown_instead_of_using_physical_memory(monkeypatch):
    device = SimpleNamespace(
        platform="gpu", memory_stats=lambda: {"bytes_limit": None, "bytes_in_use": 1}
    )
    monkeypatch.setattr(jax, "device_get_memory_info", lambda _: (9000, 10000), raising=False)
    assert device_free_memory_bytes(device=device) is None
