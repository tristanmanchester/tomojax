from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark the retained product-spine tests as surface checks."""
    for item in items:
        item.add_marker(pytest.mark.surface)
