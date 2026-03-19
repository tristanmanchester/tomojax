from __future__ import annotations

import importlib


def test_alignment_modules_import_without_optax_installed() -> None:
    # Regression test for an unused optax import in tomojax.align.pipeline.
    # The test environment intentionally does not provide optax, so these imports
    # would fail before the fix even though the modules do not actually use it.
    for module_name in (
        "tomojax.align.pipeline",
        "tomojax.cli.align",
        "tomojax.cli.loss_bench",
    ):
        mod = importlib.import_module(module_name)
        assert mod is not None
