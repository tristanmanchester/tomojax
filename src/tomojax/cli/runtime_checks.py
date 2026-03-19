from __future__ import annotations

import os


def _print_runtime() -> None:
    import jax

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")


def test_gpu_main() -> None:
    _print_runtime()


def test_cpu_main() -> None:
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    _print_runtime()
