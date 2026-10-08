from __future__ import annotations

import importlib
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

pytestmark = pytest.mark.surface


def test_public_facades_import_cleanly() -> None:
    modules = (
        "tomojax",
        "tomojax.alignment",
        "tomojax.alignment.api",
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


def test_cli_imports_do_not_configure_jax_allocator(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_ALLOCATOR"):
        monkeypatch.delenv(name, raising=False)

    for module_name in (
        "tomojax.cli.main",
        "tomojax.cli.recon",
        "tomojax.cli.simulate",
        "tomojax.cli.align",
    ):
        module = importlib.import_module(module_name)
        importlib.reload(module)

    assert "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ
    assert "XLA_PYTHON_CLIENT_ALLOCATOR" not in os.environ


def test_non_product_namespaces_are_absent() -> None:
    for module_name in ("tomojax.bench", "tomojax.verify", "tomojax.data"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_io_root_surface_is_smaller_than_full_api() -> None:
    import tomojax.io as io_root
    import tomojax.io.api as io_api

    assert len(io_root.__all__) < len(io_api.__all__)
    assert "load_dataset" in io_root.__all__
    assert "preprocess_nxtomo" in io_root.__all__
    assert "inspect_dataset" not in io_root.__all__
    assert "flat_dark_to_absorption" not in io_root.__all__


def test_cli_catalog_is_product_only(capsys: pytest.CaptureFixture[str]) -> None:
    from tomojax.cli import PRODUCT_COMMANDS, product_command_names
    from tomojax.cli.main import main

    assert product_command_names() == (
        "inspect",
        "import",
        "preprocess",
        "recon",
        "align",
        "export",
        "simulate",
    )
    assert all(command.name in product_command_names() for command in PRODUCT_COMMANDS)
    assert main(["--help"]) == 0
    captured = capsys.readouterr()
    assert "tomojax inspect scan.nxs" in captured.out
    assert "tomojax recon aligned.nxs -o recon.nxs" in captured.out
    assert "dev" not in captured.out.lower()
    assert "benchmark" not in captured.out.lower()

    with pytest.raises(SystemExit) as exc_info:
        main(["dev", "--help"])
    assert exc_info.value.code == 2


def test_product_command_help_has_no_dev_story(capsys: pytest.CaptureFixture[str]) -> None:
    from tomojax.cli import product_command_names
    from tomojax.cli.main import main

    for command in product_command_names():
        with pytest.raises(SystemExit) as exc_info:
            main([command, "--help"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        lowered = captured.out.lower()
        assert "diagnostic" not in lowered
        assert "benchmark" not in lowered
        assert "v" + "1" not in lowered
        assert "par" + "ity" not in lowered
        # One shape for every command: INPUT, then -o OUTPUT guarded by --force.
        usage = captured.out.split("\n\n", 1)[0]
        if command != "simulate":
            assert "INPUT" in usage
        if command != "inspect":
            assert "-o OUTPUT" in usage
        assert "--force" in usage


def test_cli_prints_its_version(capsys: pytest.CaptureFixture[str]) -> None:
    from tomojax import __version__
    from tomojax.cli.main import main

    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"tomojax {__version__}"


def test_cli_lists_expert_settings_as_config_keys(capsys: pytest.CaptureFixture[str]) -> None:
    from tomojax.cli.main import main

    with pytest.raises(SystemExit) as exc_info:
        main(["align", "--help"])
    assert exc_info.value.code == 0
    assert "--outer-iterations" not in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc_info:
        main(["align", "--config-keys"])
    assert exc_info.value.code == 0
    keys = capsys.readouterr().out
    assert "outer_iterations = " in keys
    assert "mode = " in keys


def test_root_docs_do_not_advertise_removed_package_surfaces() -> None:
    root = Path(__file__).resolve().parents[1]
    docs = [root / "README.md", *sorted((root / "docs").glob("*.md"))]
    public_docs = "\n".join(path.read_text(encoding="utf-8") for path in docs)
    assert re.search(r"tomojax\.data(?!sets)\b", public_docs) is None
    assert re.search(r"tomojax\.bench\b", public_docs) is None
    assert re.search(r"tomojax\.verify\b", public_docs) is None
    # Guard actual removed entrypoints. Ordinary words such as compatibility
    # and parity also belong in dependency and numerical comparison guidance.
    assert re.search(r"tomojax\s+(?:bench|verify)\b", public_docs) is None


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


def test_private_import_guard_expands_root_package_from_imports(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    bad_test = tmp_path / "test_bad_root_import.py"
    bad_test.write_text("from tomojax import _data\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "tools/check_public_imports.py", str(bad_test)],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "private data implementation" in result.stderr
    assert "imports tomojax._data" in result.stderr


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


def test_config_files_set_on_off_flags_and_expert_settings(tmp_path: Path) -> None:
    from tomojax.cli.align import build_parser
    from tomojax.cli.config import parse_args_with_config

    config = tmp_path / "align.toml"
    _ = config.write_text("poses = false\nseed_translations = false\n", encoding="utf-8")
    args, metadata = parse_args_with_config(
        build_parser(), ["scan.nxs", "-o", "out.nxs", "--config", str(config)]
    )
    assert args.poses is False
    assert metadata["settings"] == {"seed_translations": False}


def test_config_files_name_the_valid_keys_for_an_unknown_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tomojax.cli.main import main

    config = tmp_path / "align.toml"
    _ = config.write_text("outer_iteration = 3\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _ = main(["align", "scan.nxs", "-o", str(tmp_path / "out.nxs"), "--config", str(config)])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "unknown config key(s)" in err
    assert "outer_iterations" in err


def test_config_files_name_the_new_key_for_a_retired_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tomojax.cli.main import main

    config = tmp_path / "recon.toml"
    _ = config.write_text("lambda_tv = 0.01\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _ = main(["recon", "scan.nxs", "-o", str(tmp_path / "out.nxs"), "--config", str(config)])
    assert exc.value.code == 2
    assert "config key 'lambda_tv'" in (err := capsys.readouterr().err)
    assert err.rstrip().endswith("was renamed 'tv_weight'")


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("outer_iters", "outer_iterations"),
        ("recon_algo", "reconstruction"),
        ("freeze_dofs", "freeze"),
    ],
)
def test_align_config_files_name_the_new_key_for_a_retired_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], old: str, new: str
) -> None:
    from tomojax.cli.main import main

    config = tmp_path / "align.toml"
    _ = config.write_text(f"{old} = 1\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _ = main(["align", "scan.nxs", "-o", str(tmp_path / "out.nxs"), "--config", str(config)])
    assert exc.value.code == 2
    assert capsys.readouterr().err.rstrip().endswith(f"was renamed '{new}'")
