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
