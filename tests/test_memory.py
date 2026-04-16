from __future__ import annotations

from tomojax.utils import memory as memory_utils


def test_memory_helpers_cover_batch_estimation_and_backend_defaults(monkeypatch):
    monkeypatch.setenv("TOMOJAX_MAX_VIEWS_PER_BATCH", "3")

    capped = memory_utils.estimate_views_per_batch(
        n_views=10,
        grid_nxyz=(16, 16, 16),
        det_nuv=(8, 8),
        free_bytes_override=10**9,
    )
    fallback = memory_utils.estimate_views_per_batch(
        n_views=10,
        grid_nxyz=(16, 16, 16),
        det_nuv=(8, 8),
        free_bytes_override=0,
    )

    assert 1 <= capped <= 3
    assert fallback == 8
    assert memory_utils._bytes_per("bf16") == 2
    assert memory_utils._bytes_per("fp32") == 4

    monkeypatch.setattr(memory_utils, "_current_backend", lambda: "cpu")
    assert memory_utils.default_gather_dtype() == "fp32"


def test_safety_fraction_reduces_batch_estimate(monkeypatch):
    monkeypatch.setenv("TOMOJAX_MAX_VIEWS_PER_BATCH", "128")

    full_budget = memory_utils.estimate_views_per_batch(
        n_views=128,
        grid_nxyz=(64, 64, 64),
        det_nuv=(64, 64),
        free_bytes_override=100_000_000,
        safety_frac=1.0,
    )
    half_budget = memory_utils.estimate_views_per_batch(
        n_views=128,
        grid_nxyz=(64, 64, 64),
        det_nuv=(64, 64),
        free_bytes_override=100_000_000,
        safety_frac=0.5,
    )

    assert full_budget > 1
    assert 1 <= half_budget < full_budget


def test_host_available_memory_bytes_uses_sysconf(monkeypatch):
    monkeypatch.setattr(
        memory_utils.os,
        "sysconf_names",
        {"SC_AVPHYS_PAGES": 1, "SC_PAGE_SIZE": 2},
    )

    def fake_sysconf(name: str) -> int:
        if name == "SC_AVPHYS_PAGES":
            return 256
        if name == "SC_PAGE_SIZE":
            return 4096
        raise AssertionError(f"unexpected sysconf key: {name}")

    monkeypatch.setattr(memory_utils.os, "sysconf", fake_sysconf)

    assert memory_utils._host_available_memory_bytes() == 256 * 4096


def test_host_available_memory_bytes_returns_none_without_sysconf_support(monkeypatch):
    monkeypatch.setattr(memory_utils.os, "sysconf_names", {})

    assert memory_utils._host_available_memory_bytes() is None
