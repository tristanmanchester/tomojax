"""Small runtime-check commands exposed by ``tomojax``."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Sequence


class _JaxRuntime(Protocol):
    def default_backend(self) -> str: ...

    def devices(self) -> Sequence[object]: ...


def _print_runtime() -> None:
    """Print the active JAX backend and devices."""
    import jax

    jax_runtime = cast("_JaxRuntime", jax)
    print(f"JAX backend: {jax_runtime.default_backend()}")
    print(f"Devices: {jax_runtime.devices()}")


def test_gpu_main() -> None:
    """Print the default GPU-capable JAX runtime, if available."""
    _print_runtime()


def test_cpu_main() -> None:
    """Force CPU JAX selection and print the resulting runtime."""
    _ = os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    _print_runtime()
