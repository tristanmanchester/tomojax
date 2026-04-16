from __future__ import annotations

from contextlib import nullcontext
import sys
import types

from tomojax.cli import _runtime as runtime_helpers


def test_transfer_guard_context_covers_disabled_and_jax_paths(monkeypatch):
    disabled = runtime_helpers.transfer_guard_context("off")
    assert isinstance(disabled, type(nullcontext()))

    calls: list[str] = []
    sentinel = object()
    fake_jax = types.SimpleNamespace(transfer_guard=lambda mode: calls.append(mode) or sentinel)
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    enabled = runtime_helpers.transfer_guard_context("log")

    assert enabled is sentinel
    assert calls == ["log"]


def test_transfer_guard_context_warns_when_requested_mode_is_unavailable(
    monkeypatch, caplog
):
    fake_jax = types.ModuleType("jax")
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    caplog.set_level("WARNING")

    fallback = runtime_helpers.transfer_guard_context("log")

    assert isinstance(fallback, type(nullcontext()))
    assert any(
        "Requested transfer guard mode 'log'" in record.message for record in caplog.records
    )


def test_transfer_guard_context_warns_when_guard_initialization_fails(
    monkeypatch, caplog
):
    def _boom(mode: str):
        raise RuntimeError(f"cannot enable {mode}")

    fake_jax = types.SimpleNamespace(transfer_guard=_boom)
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    caplog.set_level("WARNING")

    fallback = runtime_helpers.transfer_guard_context("log")

    assert isinstance(fallback, type(nullcontext()))
    assert any("cannot enable log" in record.message for record in caplog.records)
