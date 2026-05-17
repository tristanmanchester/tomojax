from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


def _load_import_guard() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "tools" / "check_public_imports.py"
    spec = importlib.util.spec_from_file_location("check_public_imports", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


find_violations = _load_import_guard().find_violations


def _write(root: Path, relative: str, source: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_import_guard_rejects_legacy_data_from_cli_surface(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "src/tomojax/cli/example.py",
        "from tomojax.data.io_hdf5 import load_nxtomo\n",
    )

    violations = find_violations([path], tmp_path)

    assert len(violations) == 1
    assert violations[0].reason == (
        "legacy data namespace must be reached through tomojax.io or tomojax.datasets"
    )


def test_import_guard_rejects_nested_alignment_from_bench_surface(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "src/tomojax/bench/example.py",
        "from tomojax.align.objectives.loss_specs import parse_loss_spec\n",
    )

    violations = find_violations([path], tmp_path)

    assert len(violations) == 1
    assert violations[0].reason == "nested alignment namespace must be reached through tomojax.align.api"


def test_import_guard_rejects_core_geometry_from_cli_surface(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "src/tomojax/cli/example.py",
        "from tomojax.core.geometry import Detector, Grid\n",
    )

    violations = find_violations([path], tmp_path)

    assert len(violations) == 1
    assert violations[0].reason == (
        "core geometry namespace must be reached through tomojax.geometry"
    )


def test_import_guard_allows_legacy_data_behind_io_adapter(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "src/tomojax/io/_datasets.py",
        "from tomojax.data.io_hdf5 import load_nxtomo\n",
    )

    assert find_violations([path], tmp_path) == []
