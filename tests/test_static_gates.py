"""Guard lightweight CLI help and imports, and generated wire files."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tomojax.cli import product_command_names

pytestmark = pytest.mark.surface

ROOT = Path(__file__).resolve().parents[1]

CLI = """\
import sys
from tomojax.cli.main import run
sys.argv[0] = "tomojax"
sys.exit(run())
"""


def _python(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )


def _without_accelerator_imports(source: str) -> str:
    # argparse exits with SystemExit on --help, so inspect the child in finally.
    return (
        "import sys\ntry:\n"
        + "\n".join(f"    {line}" for line in source.splitlines())
        + """
finally:
    loaded = {name.partition(".")[0] for name in sys.modules} & {"jax", "cupy"}
    if loaded:
        raise SystemExit(
            f"Imported {', '.join(sorted(loaded))}. Defer accelerator imports until compute."
        )
"""
    )


@pytest.mark.parametrize("command", ["", *product_command_names()], ids=lambda c: c or "root")
def test_cli_help_is_lazy(command: str) -> None:
    args = [command, "--help"] if command else ["--help"]
    result = _python("-c", _without_accelerator_imports(CLI), *args)
    assert result.returncode == 0, f"tomojax {' '.join(args)}\n{result.stderr}"
    assert result.stdout.lower().startswith("usage: tomojax")


@pytest.mark.parametrize("module", ["tomojax.errors", "tomojax.wire"])
def test_lightweight_module_import_is_lazy(module: str) -> None:
    result = _python("-c", _without_accelerator_imports(f"import {module}"))
    assert result.returncode == 0, f"import {module} exited {result.returncode}\n{result.stderr}"


@pytest.mark.parametrize("generator", ["build_wire_options.py", "build_wire_schemas.py"])
def test_wire_files_are_current(generator: str) -> None:
    result = _python(f"tools/{generator}", "--check")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("generator", "files"),
    [
        (
            "build_wire_options.py",
            ["src/tomojax/cli/wire_options.py", "src/tomojax/cli/parameters.json"],
        ),
        ("build_wire_schemas.py", ["docs/protocol/arrays.schema.json"]),
    ],
)
def test_wire_check_reports_stale_files_without_writing(
    tmp_path: Path, generator: str, files: list[str]
) -> None:
    # The scripts resolve output paths relative to their own location.
    script = tmp_path / "tools" / generator
    script.parent.mkdir()
    shutil.copyfile(ROOT / "tools" / generator, script)
    if generator == "build_wire_options.py":
        guide = tmp_path / "docs/export.md"
        guide.parent.mkdir()
        guide.write_text("<!-- formats -->\nstale table\n<!-- /formats -->\n")
    if generator == "build_wire_schemas.py":
        schema_dir = tmp_path / "docs/protocol"
        schema_dir.mkdir(parents=True)
        for schema in (ROOT / "docs/protocol").glob("*.schema.json"):
            shutil.copyfile(schema, schema_dir / schema.name)
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stale\n")

    result = _python(str(script), "--check")

    assert result.returncode == 1
    assert f"uv run python tools/{generator}" in result.stderr
    if generator == "build_wire_options.py":
        assert "docs/export.md" in result.stderr
        assert "stale table" in guide.read_text()
    for name in files:
        assert name in result.stderr
        assert (tmp_path / name).read_bytes() == b"stale\n"
