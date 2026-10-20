from collections.abc import Iterator
import os
from pathlib import Path
import sys

import pytest

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


@pytest.fixture(autouse=True, scope="module")
def _release_compiled_programs() -> Iterator[None]:
    """Drop each module's compiled programs once it finishes.

    One process compiling the whole suite's programs ends in XLA's CPU compiler
    crashing late in the run; the persistent cache still makes recompiling cheap.
    """
    yield
    import jax

    jax.clear_caches()
    # Return the freed heap to the system: glibc otherwise keeps several GB of
    # per-module scratch, so one process grows by the sum of every module's peak.
    if sys.platform == "linux":
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
