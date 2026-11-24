"""The launcher preserves project selection and uses the active Python runtime."""

import json
import os
import sys

import pytest

from tomojax.cli import studio
from tomojax.cli.main import main

pytestmark = pytest.mark.surface


@pytest.fixture
def native(tmp_path, monkeypatch):
    monkeypatch.setattr(studio, "__file__", str(tmp_path / "checkout/src/tomojax/cli/studio.py"))
    record = tmp_path / "launch.json"
    binary = tmp_path / "tomojax-studio"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import json, os\n"
        "from pathlib import Path\n"
        "Path(os.environ['STUDIO_LAUNCH_RECORD']).write_text(json.dumps({\n"
        "    key: os.environ.get(key) for key in [\n"
        "        'TOMOJAX_STUDIO_PYTHON', 'TOMOJAX_STUDIO_INPUT', 'TOMOJAX_STUDIO_DEMO'\n"
        "    ]\n"
        "}))\n"
        "raise SystemExit(int(os.environ.get('STUDIO_TEST_EXIT', '0')))\n"
    )
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("STUDIO_LAUNCH_RECORD", str(record))
    return record


def test_selected_project_and_python_reach_native_process(native, tmp_path):
    project = tmp_path / "My tomography project"
    project.mkdir()
    assert main(["studio", str(project), "--demo"]) == 0
    launched = json.loads(native.read_text())
    assert launched == {
        "TOMOJAX_STUDIO_PYTHON": sys.executable,
        "TOMOJAX_STUDIO_INPUT": str(project.resolve()),
        "TOMOJAX_STUDIO_DEMO": "1",
    }


def test_new_workspace_does_not_inherit_an_unrelated_input(native, monkeypatch):
    monkeypatch.setenv("TOMOJAX_STUDIO_INPUT", "/old/project")
    monkeypatch.setenv("TOMOJAX_STUDIO_DEMO", "1")
    assert main(["studio"]) == 0
    launched = json.loads(native.read_text())
    assert launched["TOMOJAX_STUDIO_INPUT"] is None
    assert launched["TOMOJAX_STUDIO_DEMO"] is None


def test_native_failure_is_a_cli_failure(native, monkeypatch):
    monkeypatch.setenv("STUDIO_TEST_EXIT", "7")
    assert main(["studio"]) == 1


def test_missing_input_fails_before_launch(native, tmp_path, capsys):
    assert main(["studio", str(tmp_path / "missing.nxs")]) == 1
    assert "missing.nxs" in capsys.readouterr().err
    assert not native.exists()


def test_launcher_help_and_usage_exit_codes(capsys):
    with pytest.raises(SystemExit) as help_exit:
        main(["studio", "--help"])
    assert help_exit.value.code == 0
    assert "PROJECT_OR_INPUT" in capsys.readouterr().out
    with pytest.raises(SystemExit) as usage_exit:
        main(["studio", "--not-an-option"])
    assert usage_exit.value.code == 2


def test_missing_native_application_explains_the_build_step(native, monkeypatch, capsys):
    (native.parent / "tomojax-studio").unlink()
    monkeypatch.setenv("PATH", str(native.parent))
    assert main(["studio"]) == 1
    assert "tomojax studio --build" in capsys.readouterr().err


def test_build_failure_stops_before_launch_and_uses_the_locked_manifest(native, monkeypatch):
    checkout = native.parent / "checkout/studio"
    checkout.mkdir(parents=True)
    (checkout / "Cargo.toml").write_text("[package]\nname='fixture'\n")
    record = native.parent / "build.json"
    cargo = native.parent / "cargo"
    cargo.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['STUDIO_BUILD_RECORD']).write_text(json.dumps(\n"
        "    {'args': sys.argv[1:], 'cwd': str(Path.cwd())}))\n"
        "raise SystemExit(101)\n"
    )
    cargo.chmod(0o755)
    monkeypatch.setenv("STUDIO_BUILD_RECORD", str(record))
    assert main(["studio", "--build"]) == 1
    build = json.loads(record.read_text())
    assert build["cwd"] == str(checkout)
    command = build["args"]
    assert command == [
        "build",
        "--locked",
        "--release",
        "--manifest-path",
        str(checkout / "Cargo.toml"),
        "--target-dir",
        str(checkout / "target"),
    ]
    assert not native.exists()
