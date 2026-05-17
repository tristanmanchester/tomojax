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


_import_guard = _load_import_guard()
find_violations = _import_guard.find_violations
DEFAULT_SCAN_PATHS = _import_guard.DEFAULT_SCAN_PATHS
ALIGNMENT_FACADE_REASON = _import_guard.ALIGNMENT_FACADE_REASON


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
        "from tomojax.align._objectives.loss_specs import parse_loss_spec\n",
    )

    violations = find_violations([path], tmp_path)

    assert len(violations) == 1
    assert violations[0].reason == ALIGNMENT_FACADE_REASON


def test_import_guard_rejects_nested_alignment_from_top_level_bench(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "bench/example.py",
        "from tomojax.align._objectives.loss_specs import parse_loss_spec\n",
    )

    violations = find_violations([path], tmp_path)

    assert len(violations) == 1
    assert violations[0].reason == ALIGNMENT_FACADE_REASON


def test_import_guard_rejects_nested_alignment_io_from_cli_surface(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "src/tomojax/cli/example.py",
        "from tomojax.align.io.params_export import save_alignment_params_json\n",
    )

    violations = find_violations([path], tmp_path)

    assert len(violations) == 1
    assert violations[0].reason == ALIGNMENT_FACADE_REASON


def test_import_guard_allows_declared_cli_alignment_sidecar_adapters(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "src/tomojax/cli/align.py",
        "from tomojax.align.io.checkpoint import load_alignment_checkpoint\n"
        "from tomojax.align.io.params_export import save_alignment_params_json\n",
    )

    assert find_violations([path], tmp_path) == []


def test_import_guard_rejects_nested_alignment_io_from_top_level_bench(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "bench/example.py",
        "from tomojax.align.io.params_export import save_alignment_params_json\n",
    )

    violations = find_violations([path], tmp_path)

    assert len(violations) == 1
    assert violations[0].reason == ALIGNMENT_FACADE_REASON


def test_import_guard_rejects_nested_alignment_pipeline_from_cli_surface(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "src/tomojax/cli/example.py",
        "from tomojax.align.pipeline import align\n",
    )

    violations = find_violations([path], tmp_path)

    assert len(violations) == 1
    assert violations[0].reason == ALIGNMENT_FACADE_REASON


def test_import_guard_allows_current_alignment_smoke_verification_bridge(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        "src/tomojax/bench/alignment_smoke.py",
        "from tomojax.align.verification import verification_from_manifest\n",
    )

    assert find_violations([path], tmp_path) == []


def test_import_guard_rejects_nested_alignment_verification_from_other_bench(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        "src/tomojax/bench/example.py",
        "from tomojax.align.verification import verification_from_manifest\n",
    )

    violations = find_violations([path], tmp_path)

    assert len(violations) == 1
    assert violations[0].reason == ALIGNMENT_FACADE_REASON


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


def test_import_guard_rejects_core_geometry_from_top_level_bench(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "bench/example.py",
        "from tomojax.core.geometry import Detector, Grid\n",
    )

    violations = find_violations([path], tmp_path)

    assert len(violations) == 1
    assert violations[0].reason == (
        "core geometry namespace must be reached through tomojax.geometry"
    )


def test_import_guard_rejects_legacy_data_from_top_level_bench(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "bench/example.py",
        "from tomojax.data.io_hdf5 import load_nxtomo\n",
    )

    violations = find_violations([path], tmp_path)

    assert len(violations) == 1
    assert violations[0].reason == (
        "legacy data namespace must be reached through tomojax.io or tomojax.datasets"
    )


def test_import_guard_rejects_bench_import_from_cli_module(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "src/tomojax/cli/example.py",
        "from tomojax.bench import benchmark_suite\n",
    )

    violations = find_violations([path], tmp_path)

    assert {violation.reason for violation in violations} == {
        "production CLI modules must not import developer benchmark helpers"
    }


def test_import_guard_allows_bench_import_from_dev_dispatcher(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "src/tomojax/cli/main.py",
        "from tomojax.bench import benchmark_suite\n",
    )

    assert find_violations([path], tmp_path) == []


def test_import_guard_allows_marked_white_box_import_from_top_level_bench(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        "bench/example.py",
        "# check-public-imports: allow-private\n"
        "from tomojax.align._alternating import AlternatingAlignmentSolver\n",
    )

    assert find_violations([path], tmp_path) == []


def test_import_guard_scans_top_level_bench_by_default() -> None:
    assert Path("bench") in DEFAULT_SCAN_PATHS


def test_import_guard_allows_legacy_data_behind_io_adapter(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "src/tomojax/io/_datasets.py",
        "from tomojax.data.io_hdf5 import load_nxtomo\n",
    )

    assert find_violations([path], tmp_path) == []
