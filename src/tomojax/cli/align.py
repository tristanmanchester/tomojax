# ruff: noqa: E402
"""Run geometry alignment workflows from the public TomoJAX CLI."""

from __future__ import annotations

from tomojax.cli._jax_allocator import configure_jax_allocator_defaults

configure_jax_allocator_defaults()

from ._align_main import main

__all__ = [
    "main",
]
