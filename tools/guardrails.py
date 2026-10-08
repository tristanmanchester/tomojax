"""Record the public API and the code-health ratchets that tests hold the code to.

``tests/guardrails/api_surface.txt`` lists every public name of the user-facing
packages with its signature, so an API change shows up as a diff in review.
``tests/guardrails/ratchets.json`` records measures of debt (configuration
fields, exported names, CLI flags, long files, lint suppressions, complex
functions...) that may fall but not rise.

After an intended change run ``python tools/guardrails.py --write`` and commit
the updated files with it; ``tests/test_architecture.py`` fails until then.
"""
# Introspecting arbitrary modules and JSON is untyped by nature.
# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownParameterType=false

from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "tomojax"
OUT = ROOT / "tests" / "guardrails"
SURFACE_FILE = OUT / "api_surface.txt"
RATCHET_FILE = OUT / "ratchets.json"

# Packages users import; their __all__ is the public API.
PUBLIC_PACKAGES = (
    "tomojax",
    "tomojax.geometry",
    "tomojax.recon",
    "tomojax.io",
    "tomojax.alignment",
    "tomojax.datasets",
)
# Expert modules: not snapshotted name by name, but their size is ratcheted.
EXPERT_MODULES = (
    "tomojax.geometry.api",
    "tomojax.recon.api",
    "tomojax.io.api",
    "tomojax.alignment.api",
    "tomojax.datasets.api",
)
CLI_COMMANDS = ("inspect", "import", "preprocess", "recon", "align", "export", "simulate")
# Files and functions longer than these are recorded with their length, which
# may only fall: debt inside an already long one counts too.
LONG_FILE_LINES = 800
LONG_FUNCTION_LINES = 100
# Public functions whose leading optional argument is deliberately positional.
POSITIONAL_OPTIONS_ALLOWED = {"tomojax.reconstruct": ("method",)}


# ----------------------------------------------------------------------- surface


def _signature(obj: object) -> str:
    """``obj``'s signature, one parameter per line when it is long."""
    try:
        signature = inspect.signature(obj)  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError):
        return ""
    text = str(signature)
    if len(text) <= 80:
        return text
    lines = [f"        {parameter}," for parameter in signature.parameters.values()]
    for index, parameter in enumerate(signature.parameters.values()):
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            lines.insert(index, "        *,")
            break
    tail = (
        ""
        if signature.return_annotation is inspect.Signature.empty
        else (f" -> {inspect.formatannotation(signature.return_annotation)}")
    )
    return "(\n" + "\n".join(lines) + f"\n    ){tail}"


def _describe(module: str, name: str, obj: object) -> list[str]:
    qualified = f"{module}.{name}"
    if inspect.isclass(obj):
        lines = [f"class {qualified}{_signature(obj)}"]
        own = {k for k in vars(obj) if not k.startswith("_")}
        fields = (
            {f.name for f in dataclasses.fields(obj)} if dataclasses.is_dataclass(obj) else set()
        )
        for member in sorted(own - fields):
            value = inspect.getattr_static(obj, member)
            if isinstance(value, property):
                lines.append(f"    .{member}")
            elif callable(value) or isinstance(value, staticmethod | classmethod):
                function = (
                    value.__func__ if isinstance(value, staticmethod | classmethod) else value
                )
                lines.append(f"    .{member}{_signature(function)}")
        return lines
    if callable(obj):
        return [f"def {qualified}{_signature(obj)}"]
    return [f"{qualified}: {type(obj).__name__}"]


def api_surface() -> str:
    """Every public name of :data:`PUBLIC_PACKAGES`, with signatures."""
    lines: list[str] = []
    for module_name in PUBLIC_PACKAGES:
        module = importlib.import_module(module_name)
        lines.append(f"# {module_name}")
        for name in sorted(module.__all__):
            lines.extend(_describe(module_name, name, getattr(module, name)))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------- ratchets


def _python_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _count(pattern: str) -> int:
    regex = re.compile(pattern)
    return sum(len(regex.findall(p.read_text(encoding="utf-8"))) for p in _python_files())


def _complex_functions() -> list[str]:
    """Functions over ruff's default complexity limits, as ``path::rule::name``."""
    command = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        str(SRC),
        "--isolated",
        "--select",
        "C901,PLR0911,PLR0912,PLR0915",
        "--output-format",
        "json",
        "--exit-zero",
    ]
    report = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    functions: dict[Path, dict[int, str]] = {}
    found = set()
    for item in report:
        path = Path(item["filename"]).resolve()
        if path not in functions:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            functions[path] = {
                node.lineno: node.name
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            }
        name = functions[path].get(item["location"]["row"], "?")
        found.add(f"{_relative(path)}::{item['code']}::{name}")
    return sorted(found)


def _type_errors() -> dict[str, int]:
    """Type errors over all of ``src`` by rule, from basedpyright (CI checks only part)."""
    command = [sys.executable, "-m", "basedpyright", str(SRC), "--outputjson"]
    output = subprocess.run(command, check=False, capture_output=True, text=True).stdout
    counts: dict[str, int] = {}
    # Missing optional packages (CuPy, ASTRA...) depend on the environment.
    environmental = {"reportMissingImports", "reportMissingModuleSource"}
    for item in json.loads(output)["generalDiagnostics"]:
        if item["severity"] == "error" and item.get("rule") not in environmental:
            rule = item.get("rule", "other")
            counts[rule] = counts.get(rule, 0) + 1
    return dict(sorted(counts.items()))


def _file_lines() -> dict[str, int]:
    """Files over :data:`LONG_FILE_LINES`, with their length."""
    lengths = {
        _relative(p): len(p.read_text(encoding="utf-8").splitlines()) for p in _python_files()
    }
    return {path: n for path, n in lengths.items() if n > LONG_FILE_LINES}


def _function_lines() -> dict[str, int]:
    """Functions over :data:`LONG_FUNCTION_LINES`, as ``path::qualified.name`` with their length."""
    found: dict[str, int] = {}

    def visit(node: ast.AST, prefix: str, path: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = f"{prefix}{child.name}"
                if not isinstance(child, ast.ClassDef):
                    length = (child.end_lineno or child.lineno) - child.lineno + 1
                    if length > LONG_FUNCTION_LINES:
                        found[f"{path}::{name}"] = length
                visit(child, f"{name}.", path)

    for p in _python_files():
        visit(ast.parse(p.read_text(encoding="utf-8")), "", _relative(p))
    return dict(sorted(found.items()))


def _test_private_imports() -> int:
    """Test imports of private modules, each marked ``check-public-imports: allow-private``."""
    marker = re.compile(r"check-public-imports:\s*allow-private")
    tests = ROOT / "tests"
    return sum(len(marker.findall(p.read_text(encoding="utf-8"))) for p in tests.rglob("*.py"))


def _options_of(qualified: str, obj: Callable[..., object]) -> list[str]:
    """``obj``'s optional arguments that may be passed positionally."""
    allowed = POSITIONAL_OPTIONS_ALLOWED.get(qualified, ())
    found = []
    for parameter in inspect.signature(obj).parameters.values():
        positional = parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
        has_default = parameter.default is not inspect.Parameter.empty
        if positional and has_default and parameter.name not in allowed:
            found.append(f"{qualified}({parameter.name})")
    return found


def _positional_options() -> list[str]:
    """Public functions, methods and configurations taking an option positionally.

    Configurations are the public dataclasses named ``*Config``: all their
    fields are options.
    """
    found = []
    for module_name in PUBLIC_PACKAGES:
        module = importlib.import_module(module_name)
        for name in module.__all__:
            obj = getattr(module, name)
            qualified = f"{module_name}.{name}"
            if inspect.isfunction(obj):
                found.extend(_options_of(qualified, obj))
            elif inspect.isclass(obj):
                if dataclasses.is_dataclass(obj) and name.endswith("Config"):
                    found.extend(_options_of(qualified, obj))
                for member, function in vars(obj).items():
                    if inspect.isfunction(function) and not member.startswith("_"):
                        found.extend(_options_of(f"{qualified}.{member}", function))
    return sorted(found)


def _cli_flags() -> dict[str, int]:
    from tomojax.cli._options import options

    counts = {}
    for command in CLI_COMMANDS:
        module_name = {
            "import": "import_",
            "recon": "_recon_command",
        }.get(command, command)
        module = importlib.import_module(f"tomojax.cli.{module_name}")
        build = getattr(module, "build_parser", None) or module._build_parser  # noqa: SLF001
        counts[command] = sum(1 for action in options(build()) if action.option_strings)
    return counts


def ratchets() -> dict[str, object]:
    """Current values of every ratcheted measure."""
    from tomojax.alignment import AlignConfig

    return {
        "align_config_fields": len(dataclasses.fields(AlignConfig)),
        "exported_names": {
            module: len(importlib.import_module(module).__all__)
            for module in (*PUBLIC_PACKAGES, *EXPERT_MODULES)
        },
        "cli_flags": _cli_flags(),
        "long_files": _file_lines(),
        "long_functions": _function_lines(),
        "suppressions": {
            "noqa": _count(r"#\s*noqa"),
            "type: ignore": _count(r"#\s*type:\s*ignore"),
            "pyright: ignore": _count(r"#\s*pyright:\s*ignore"),
        },
        "complex_functions": _complex_functions(),
        "type_errors": _type_errors(),
        "positional_options": _positional_options(),
        # The internal five-column pose array; public poses have six columns.
        "params5_mentions": _count(r"\bparams5\b"),
        # Each normalises a user string by hand instead of through one parser.
        "string_normalisers": _count(r"\.replace\(\"[-_]\", \"[-_]\"\)"),
        # Code assuming one device, now that work can be shared among several.
        "first_device_probes": _count(r"jax\.devices\(\)\[0\]"),
        # jaxlib's private modules, imported only for the Device type.
        "jaxlib_private_imports": _count(r"\bjaxlib\._"),
        "test_private_imports": _test_private_imports(),
    }


# -------------------------------------------------------------------------- main


def write() -> None:
    """Rewrite the recorded surface and ratchets from the current code."""
    OUT.mkdir(parents=True, exist_ok=True)
    _ = SURFACE_FILE.write_text(api_surface(), encoding="utf-8")
    _ = RATCHET_FILE.write_text(json.dumps(ratchets(), indent=2) + "\n", encoding="utf-8")


def main() -> int:
    """Run ``python tools/guardrails.py [--write]``."""
    parser = argparse.ArgumentParser(description="Record the public API and code-health ratchets.")
    _ = parser.add_argument("--write", action="store_true", help="Record the current values")
    args = parser.parse_args()
    _ = os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if args.write:
        write()
        print(f"wrote {_relative(SURFACE_FILE)} and {_relative(RATCHET_FILE)}")
        return 0
    print(json.dumps(ratchets(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
