from __future__ import annotations

import importlib
from pathlib import Path

CANONICAL_TOP_LEVEL_MODULES = (
    "align",
    "backends",
    "cli",
    "core",
    "datasets",
    "forward",
    "geometry",
    "io",
    "motion",
    "nuisance",
    "recon",
    "verify",
)


def test_canonical_v2_modules_have_public_facade_files() -> None:
    package_root = Path(__file__).resolve().parents[1] / "src" / "tomojax"

    missing = [
        f"{module_name}/{filename}"
        for module_name in CANONICAL_TOP_LEVEL_MODULES
        for filename in ("README.md", "__init__.py", "api.py")
        if not (package_root / module_name / filename).is_file()
    ]

    assert missing == []


def test_canonical_v2_modules_import_public_facades() -> None:
    for module_name in CANONICAL_TOP_LEVEL_MODULES:
        module = importlib.import_module(f"tomojax.{module_name}")
        api = importlib.import_module(f"tomojax.{module_name}.api")
        module_all: object = module.__dict__.get("__all__")
        api_all: object = api.__dict__.get("__all__")

        assert isinstance(module_all, list | tuple)
        assert isinstance(api_all, list | tuple)
