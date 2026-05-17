from __future__ import annotations

import ast
from pathlib import Path
import re
import sys
import tomllib
from typing import cast

from tomojax.cli.main import main


def test_project_scripts_keep_diagnostics_off_public_surface() -> None:
    payload = cast(
        "dict[str, object]", tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    )
    project = cast("dict[str, object]", payload["project"])
    scripts = cast("dict[str, str]", project["scripts"])

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


def test_justfile_benchmark_recipes_use_grouped_developer_cli() -> None:
    justfile = Path("justfile").read_text(encoding="utf-8")

    assert "tomojax.benchmarks" not in justfile
    assert "python -m tomojax.datasets" not in justfile
    assert "python -m tomojax.verify" not in justfile
    assert "tomojax dev align-auto" in justfile
    assert "synthetic-oracle-128:" in justfile


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


def test_current_docs_keep_diagnostics_under_dev_namespace() -> None:
    current_docs = [
        Path("README.md"),
        Path("docs/quickstart.md"),
        Path("docs/synthetic-tomography.md"),
        Path("src/tomojax/cli/README.md"),
    ]
    top_level_diagnostic = re.compile(
        r"\btomojax\s+(?!dev\s)(align-auto|loss-bench|misalign|test-gpu|test-cpu)\b"
    )
    leaks = {
        str(path): top_level_diagnostic.findall(path.read_text(encoding="utf-8"))
        for path in current_docs
    }

    assert {path: matches for path, matches in leaks.items() if matches} == {}


def test_product_docs_do_not_promote_evidence_runners_as_public_entrypoints() -> None:
    product_docs = [
        Path("docs/quickstart.md"),
        Path("docs/real-laminography.md"),
        Path("docs/synthetic-tomography.md"),
        Path("docs/support-matrix.md"),
        Path("docs/benchmark_runs/2026-05-13-production-readiness.md"),
    ]
    leaky_phrases = re.compile(
        r"production-shaped synthetic entrypoint|supported real-data entrypoint|"
        r"clean public entrypoints|public entrypoint|current evidence runs",
        re.IGNORECASE,
    )
    leaks = {
        str(path): leaky_phrases.findall(path.read_text(encoding="utf-8"))
        for path in product_docs
    }

    assert {path: matches for path, matches in leaks.items() if matches} == {}


def test_public_cli_docs_avoid_development_era_terms() -> None:
    public_cli_paths = [
        Path("README.md"),
        Path("src/tomojax/align/README.md"),
        Path("src/tomojax/align/__init__.py"),
        Path("src/tomojax/cli/README.md"),
        Path("src/tomojax/cli/main.py"),
        Path("src/tomojax/cli/simulate.py"),
        Path("src/tomojax/io/README.md"),
        Path("src/tomojax/io/_datasets.py"),
        Path("src/tomojax/recon/README.md"),
    ]
    development_terms = re.compile(
        r"\b(legacy|transitional|pre-v2|mvp|v1|parity|smoke)\b",
        re.IGNORECASE,
    )
    leaks = {
        str(path): development_terms.findall(path.read_text(encoding="utf-8"))
        for path in public_cli_paths
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


def test_production_cli_uses_alignment_facade_for_schedules_and_losses() -> None:
    cli_files = [
        Path("src/tomojax/cli/align.py"),
        Path("src/tomojax/cli/loss_bench.py"),
    ]
    leaks: dict[str, list[str]] = {}
    for path in cli_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append("." * node.level + node.module)
        forbidden = [
            module
            for module in imports
            if module.startswith(
                ("tomojax.align.geometry", "tomojax.align.model", "tomojax.align.objectives")
            )
        ]
        if forbidden:
            leaks[str(path)] = forbidden

    assert leaks == {}


def test_production_entrypoint_preserves_documented_data_shorthand(monkeypatch) -> None:
    calls: list[list[str]] = []

    def capture_sysargv_main() -> None:
        calls.append(list(sys.argv))

    monkeypatch.setattr("tomojax.cli.recon.main", capture_sysargv_main)
    monkeypatch.setattr("tomojax.cli.align.main", capture_sysargv_main)

    assert main(["recon", "scan.nxs", "--out", "recon.nxs"]) == 0
    assert main(["align", "scan.nxs", "--out", "aligned.nxs"]) == 0

    assert calls == [
        ["tomojax recon", "--data", "scan.nxs", "--out", "recon.nxs"],
        ["tomojax align", "--data", "scan.nxs", "--out", "aligned.nxs"],
    ]


def test_production_modules_do_not_import_lower_level_data_package() -> None:
    production_modules = [
        "align",
        "backends",
        "calibration",
        "cli",
        "core",
        "forward",
        "geometry",
        "motion",
        "nuisance",
        "recon",
        "verify",
    ]
    leaks: dict[str, list[str]] = {}
    for module in production_modules:
        for path in sorted(Path("src/tomojax", module).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    imports.append("." * node.level + node.module)
            forbidden = [
                imported
                for imported in imports
                if imported == "tomojax.data" or imported.startswith(("tomojax.data.", "..data"))
            ]
            if forbidden:
                leaks[str(path)] = forbidden

    assert leaks == {}


def test_production_modules_do_not_import_calibration_package_directly() -> None:
    production_modules = [
        "align",
        "backends",
        "cli",
        "core",
        "data",
        "datasets",
        "forward",
        "io",
        "motion",
        "nuisance",
        "recon",
        "verify",
    ]
    leaks: dict[str, list[str]] = {}
    for module in production_modules:
        for path in sorted(Path("src/tomojax", module).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    imports.append("." * node.level + node.module)
            forbidden = [
                imported
                for imported in imports
                if imported == "tomojax.calibration"
                or imported.startswith(("tomojax.calibration.", "..calibration"))
            ]
            if forbidden:
                leaks[str(path)] = forbidden

    assert leaks == {}
