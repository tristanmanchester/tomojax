from __future__ import annotations

import importlib
from pathlib import Path
import re
import subprocess
import sys

import pytest


def test_public_facades_import_cleanly() -> None:
    modules = (
        "tomojax",
        "tomojax.align",
        "tomojax.align.api",
        "tomojax.backends",
        "tomojax.cli",
        "tomojax.datasets",
        "tomojax.forward",
        "tomojax.geometry",
        "tomojax.io",
        "tomojax.motion",
        "tomojax.nuisance",
        "tomojax.recon",
    )
    for module_name in modules:
        assert importlib.import_module(module_name) is not None


def test_removed_non_product_namespaces_are_absent() -> None:
    for module_name in ("tomojax.bench", "tomojax.verify", "tomojax.data"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_cli_catalog_is_product_only(capsys: pytest.CaptureFixture[str]) -> None:
    from tomojax.cli import PRODUCT_COMMANDS, product_command_names
    from tomojax.cli.main import main

    assert product_command_names() == (
        "inspect",
        "validate",
        "preprocess",
        "ingest",
        "convert",
        "recon",
        "align",
        "simulate",
    )
    assert all(command.name in product_command_names() for command in PRODUCT_COMMANDS)
    assert main(["--help"]) == 0
    captured = capsys.readouterr()
    assert "tomojax inspect scan.nxs" in captured.out
    assert "dev" not in captured.out.lower()
    assert "benchmark" not in captured.out.lower()

    with pytest.raises(SystemExit) as exc_info:
        main(["dev", "--help"])
    assert exc_info.value.code == 2


def test_product_command_help_has_no_dev_story(capsys: pytest.CaptureFixture[str]) -> None:
    from tomojax.cli.main import main

    for command in (
        "inspect",
        "validate",
        "preprocess",
        "ingest",
        "convert",
        "recon",
        "align",
        "simulate",
    ):
        with pytest.raises(SystemExit) as exc_info:
            main([command, "--help"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        lowered = captured.out.lower()
        assert "diagnostic" not in lowered
        assert "benchmark" not in lowered
        assert "v1" not in lowered
        assert "parity" not in lowered


def test_root_docs_do_not_advertise_data_as_public_surface() -> None:
    root = Path(__file__).resolve().parents[1]
    docs = [root / "README.md", *sorted((root / "docs").glob("*.md"))]
    public_docs = "\n".join(path.read_text(encoding="utf-8") for path in docs)

    assert re.search(r"tomojax\.data(?!sets)\b", public_docs) is None
    assert re.search(r"tomojax\.bench\b", public_docs) is None
    assert re.search(r"tomojax\.verify\b", public_docs) is None


def test_private_import_guard_blocks_tests_from_internal_data_namespace(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    bad_test = tmp_path / "test_bad_data_import.py"
    bad_test.write_text("from tomojax._data.phantoms import make_phantom\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "tools/check_public_imports.py", str(bad_test)],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "private data implementation" in result.stderr


def test_private_import_guard_passes_on_product_tree() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "tools/check_public_imports.py"],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
