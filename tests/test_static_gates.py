"""Guard lightweight CLI help, public identifier spelling and generated wire files."""

import importlib
import inspect
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest
from tools.guardrails import EXPERT_MODULES, PUBLIC_PACKAGES

from tomojax.cli import product_command_names

pytestmark = pytest.mark.surface

ROOT = Path(__file__).resolve().parents[1]
SPELLING_RULE = (
    "CONTRIBUTING.md uses British English in prose and US English in API/CLI identifiers."
)
BRITISH_WORD = re.compile(
    r"centres?|(?:normalis|optimis)(?:e[ds]?|ing)|regularis.*|colours?"
    r"|(?:initialis|serialis)(?:e.*|ing|ations?)"
)
IDENTIFIER_WORD = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|[^a-z]|$)")
# Already part of the recorded public API. Remove these exceptions when it is renamed.
EXISTING_SPELLINGS = {
    "tomojax.interactive.centre_sweep",
    "tomojax.interactive.centre_sweep(centres)",
}

CLI_HELP = """\
import argparse
import json
import sys
from unittest.mock import patch

options = set()
add_arguments = argparse.HelpFormatter.add_arguments

def collect_options(formatter, actions):
    options.update(
        option
        for action in actions
        if action.help != argparse.SUPPRESS
        for option in action.option_strings
        if option.startswith("--")
    )
    add_arguments(formatter, actions)

with patch.object(argparse.HelpFormatter, "add_arguments", collect_options):
    from tomojax.cli.main import run
    sys.argv[0] = "tomojax"
    try:
        sys.exit(run())
    finally:
        print(json.dumps(sorted(options)))
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


def _has_british_spelling(identifier: str) -> bool:
    return any(BRITISH_WORD.fullmatch(word.lower()) for word in IDENTIFIER_WORD.findall(identifier))


@pytest.mark.parametrize("command", ["", *product_command_names()], ids=lambda c: c or "root")
def test_cli_help_is_lazy_and_uses_us_english(command: str) -> None:
    args = [command, "--help"] if command else ["--help"]
    invocation = " ".join(["tomojax", *args])
    result = _python("-c", _without_accelerator_imports(CLI_HELP), *args)
    assert result.returncode == 0, f"{invocation} exited {result.returncode}\n{result.stderr}"
    options = json.loads(result.stdout.splitlines()[-1])
    assert "--help" in options, f"Could not collect public options from {invocation}"
    violations = [option for option in options if _has_british_spelling(option)]
    assert not violations, f"{SPELLING_RULE}\n{invocation}: {', '.join(violations)}"


@pytest.mark.parametrize("module", ["tomojax.errors", "tomojax.wire"])
def test_lightweight_module_import_is_lazy(module: str) -> None:
    result = _python("-c", _without_accelerator_imports(f"import {module}"))
    assert result.returncode == 0, f"import {module} exited {result.returncode}\n{result.stderr}"


def test_public_python_identifiers_use_us_english() -> None:
    package_roots = {
        f"tomojax.{path.parent.name}"
        for path in (ROOT / "src/tomojax").glob("*/__init__.py")
        if not path.parent.name.startswith("_")
    }
    violations = set()
    for module_name in sorted(package_roots.union(PUBLIC_PACKAGES, EXPERT_MODULES)):
        module = importlib.import_module(module_name)
        for name in getattr(module, "__all__", ()):
            export = getattr(module, name)
            identifiers = [(name, export)]
            if inspect.isclass(export):
                identifiers.extend(
                    (f"{name}.{member}", getattr(export, member))
                    for member in sorted(vars(export))
                    if not member.startswith("_")
                )
            for identifier, obj in identifiers:
                qualified = f"{module_name}.{identifier}"
                if _has_british_spelling(identifier.rsplit(".", 1)[-1]):
                    violations.add(qualified)
                if not callable(obj):
                    continue
                try:
                    parameters = inspect.signature(obj).parameters
                except (TypeError, ValueError):
                    # Some extension types and callables expose no signature.
                    continue
                violations.update(
                    f"{qualified}({parameter})"
                    for parameter in parameters
                    if _has_british_spelling(parameter)
                )
    violations -= EXISTING_SPELLINGS
    assert not violations, SPELLING_RULE + "\n" + "\n".join(sorted(violations))


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
    for name in files:
        assert name in result.stderr
        assert (tmp_path / name).read_bytes() == b"stale\n"
