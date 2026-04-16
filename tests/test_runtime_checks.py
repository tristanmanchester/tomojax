from __future__ import annotations

import os

from tomojax.cli import runtime_checks


def test_runtime_checks_entrypoints_delegate_to_runtime_printer(monkeypatch):
    calls: list[str | None] = []

    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.setattr(
        runtime_checks,
        "_print_runtime",
        lambda: calls.append(os.environ.get("JAX_PLATFORM_NAME")),
    )

    runtime_checks.test_gpu_main()
    runtime_checks.test_cpu_main()

    assert calls == [None, "cpu"]
    assert os.environ["JAX_PLATFORM_NAME"] == "cpu"
