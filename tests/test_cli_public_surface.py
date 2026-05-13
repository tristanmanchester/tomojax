from __future__ import annotations

import ast
from pathlib import Path
import re
import tomllib


def test_project_scripts_keep_diagnostics_off_public_surface() -> None:
    payload = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = payload["project"]["scripts"]

    assert "tomojax" in scripts
    assert scripts["tomojax"] == "tomojax.cli.main:main"
    assert set(scripts) == {"tomojax"}

    forbidden_fragments = (
        "bench",
        "pallas-sanity",
        "current-baseline",
        "synthetic-benchmark",
        "test-gpu",
        "test-cpu",
    )
    leaked = [
        name
        for name in scripts
        if name != "tomojax" and any(fragment in name for fragment in forbidden_fragments)
    ]
    assert leaked == []


def test_current_docs_use_grouped_cli_commands() -> None:
    current_docs = [
        Path("README.md"),
        Path("docs/quickstart.md"),
        Path("docs/synthetic-tomography.md"),
        Path("docs/benchmark_runs/2026-05-13-production-readiness.md"),
        Path("docs/benchmark_runs/2026-05-13-production-hardening.md"),
        Path("docs/benchmark_runs/2026-05-13-synthetic128-production-gates.md"),
        Path("src/tomojax/cli/README.md"),
    ]
    retired_command = re.compile(
        r"tomojax-(inspect|validate|preprocess|convert|recon|align|simulate|"
        r"align-auto|loss-bench|misalign|test-gpu|test-cpu)\b",
    )
    leaks = {
        str(path): retired_command.findall(path.read_text(encoding="utf-8"))
        for path in current_docs
    }

    assert {path: matches for path, matches in leaks.items() if matches} == {}


def test_cli_modules_do_not_import_transitional_data_package() -> None:
    cli_files = sorted(Path("src/tomojax/cli").glob("*.py"))
    leaks: dict[str, list[str]] = {}
    for path in cli_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                module = "." * node.level + node.module
                imports.append(module)
        forbidden = [
            module
            for module in imports
            if module == "tomojax.data" or module.startswith(("tomojax.data.", "..data"))
        ]
        if forbidden:
            leaks[str(path)] = forbidden

    assert leaks == {}
