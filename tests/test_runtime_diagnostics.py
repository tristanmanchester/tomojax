from __future__ import annotations

from contextlib import nullcontext
import logging
import sys
from types import SimpleNamespace

import jax
import pytest

# check-public-imports: allow-private
from tomojax.backends import _memory
from tomojax.core import host_volume


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({}, 50),
        ({"/proc/self/cgroup": "0::/leaf\n1:name=systemd:/leaf\n"}, 50),
        ({"/cg/leaf/memory.high": None}, 70),
        ({"/proc/meminfo": None}, 50),
        ({"/cg/memory.current": "90", "/cg/memory.max": "100"}, 10),
        ({"/proc/self/mountinfo": "1 0 0:1 /other /cg ro - cgroup2 cg rw"}, 102400),
        ({"/proc/self/mountinfo": "1 0 0:1 /leaf /cg/leaf ro - cgroup2 cg rw"}, 50),
        ({"/proc/self/cgroup": "0::/../leaf"}, 102400),
    ],
)
def test_host_capacity_respects_visible_cgroups(monkeypatch, updates, expected):
    files = {
        "/proc/self/cgroup": "0::/leaf\n",
        "/proc/self/mountinfo": "1 0 0:1 / /cg ro - cgroup2 cg rw",
        "/proc/meminfo": "MemAvailable: 100 kB\n",
        "/cg/leaf/memory.current": "30",
        "/cg/leaf/memory.high": "80",
        "/cg/leaf/memory.max": "100",
    } | updates

    def read(path):
        value = files.get(str(path))
        if value is None:
            raise FileNotFoundError(str(path))
        return value

    monkeypatch.setattr(host_volume.Path, "read_text", read)
    monkeypatch.setattr(host_volume.os, "sysconf", lambda name: 64)
    assert host_volume.host_available_memory_bytes() == expected


def test_device_memory_probe_logs_debug_before_host_fallback(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_devices(_kind: str) -> list[object]:
        raise RuntimeError("device probe unavailable")

    monkeypatch.setattr(jax, "devices", raise_devices)
    monkeypatch.setattr(_memory, "host_available_memory_bytes", lambda: None)

    with caplog.at_level(logging.DEBUG, logger="tomojax.backends._memory"):
        free_bytes = _memory.device_free_memory_bytes()

    assert free_bytes is None
    assert "Device memory probe failed" in caplog.text
    assert "device probe unavailable" in caplog.text


def test_device_memory_is_read_from_the_requested_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(jax, "device_get_memory_info", raising=False)
    gpu = SimpleNamespace(platform="gpu", memory_stats=lambda: {"free_bytes": 1234})
    assert _memory.device_free_memory_bytes(device=gpu) == 1234

    monkeypatch.setattr(_memory, "host_available_memory_bytes", lambda: 5678)
    assert _memory.device_free_memory_bytes(device=SimpleNamespace(platform="cpu")) == 5678


def test_cuda_memory_fallback_uses_requested_local_device(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = []
    cuda = SimpleNamespace(
        Device=lambda ordinal: (selected.append(ordinal), nullcontext())[1],
        runtime=SimpleNamespace(memGetInfo=lambda: (1234, 5678)),
    )
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=cuda))
    monkeypatch.delattr(jax, "device_get_memory_info", raising=False)
    gpu = SimpleNamespace(platform="gpu", id=17, local_hardware_id=2, memory_stats=lambda: None)
    assert _memory.device_free_memory_bytes(device=gpu) == 1234
    assert selected == [2]


@pytest.mark.parametrize("platform", ["gpu", "tpu"])
def test_unknown_accelerator_memory_does_not_use_host(monkeypatch, platform) -> None:
    monkeypatch.delattr(jax, "device_get_memory_info", raising=False)
    monkeypatch.setattr(_memory, "_cuda_free_memory_bytes", lambda device: None)
    monkeypatch.setattr(_memory, "host_available_memory_bytes", lambda: 5678)
    device = SimpleNamespace(platform=platform, memory_stats=lambda: None)
    assert _memory.device_free_memory_bytes(device=device) is None


@pytest.mark.parametrize(("info", "expected"), [("MemAvailable: 512 kB", 524288), ("", 4096)])
def test_host_capacity_includes_reclaimable_page_cache(monkeypatch, info, expected):
    monkeypatch.setattr(host_volume.Path, "read_text", lambda _: info)
    monkeypatch.setattr(host_volume.os, "sysconf", lambda _: 64)
    assert host_volume.host_available_memory_bytes() == expected
