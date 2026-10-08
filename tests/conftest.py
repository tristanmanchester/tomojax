import os
from pathlib import Path
import sys

if os.environ.get("JAX_PLATFORMS") == "cpu":
    # Several CPU devices, so tests can share views among devices (test_devices.py).
    os.environ.setdefault("JAX_NUM_CPU_DEVICES", "4")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_collection_modifyitems(config: object, items: list[object]) -> None:
    """Skip ``gpu``-marked tests on machines without a CUDA device."""
    del config
    import jax
    import pytest

    if jax.default_backend() == "gpu":
        return
    skip = pytest.mark.skip(reason="requires CUDA")
    for item in items:
        if item.get_closest_marker("gpu") is not None:  # pyright: ignore[reportAttributeAccessIssue]
            item.add_marker(skip)  # pyright: ignore[reportAttributeAccessIssue]
