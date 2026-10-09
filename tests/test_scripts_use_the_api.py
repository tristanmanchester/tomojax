"""Benchmark drivers and examples call TomoJAX with keywords it has.

They are not run by the test suite (they need data, GPUs or other libraries),
so a renamed keyword would otherwise go unnoticed until someone runs them.
Each call is resolved through the script's own imports, so only TomoJAX's
functions and classes are checked.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted([*(ROOT / "bench").glob("*.py"), *(ROOT / "examples").glob("*.py")])


def _imports(tree: ast.Module) -> dict[str, tuple[str, list[str]]]:
    """Local names bound to TomoJAX: the module each comes from, and attributes in it.

    A name some other import also binds (TIGRE's ``fbp``, say, in one function)
    is ambiguous without scope analysis, and left out.
    """
    names: dict[str, tuple[str, list[str]]] = {}
    other: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                if alias.name.split(".")[0] != "tomojax":
                    other.add(local)
                else:
                    names[local] = (alias.name if alias.asname else "tomojax", [])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                local = alias.asname or alias.name
                if (node.module or "").split(".")[0] != "tomojax":
                    other.add(local)
                else:
                    names[local] = (node.module or "", [alias.name])
    return {name: bound for name, bound in names.items() if name not in other}


def _dotted(func: ast.expr) -> list[str] | None:
    parts = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if not isinstance(func, ast.Name):
        return None
    return [func.id, *reversed(parts)]


def _resolve(module: str, attributes: list[str]) -> Any:
    """``module``'s object at ``attributes``, importing submodules as Python would."""
    obj: Any = importlib.import_module(module)
    for i, part in enumerate(attributes):
        if hasattr(obj, part):
            obj = getattr(obj, part)
        else:  # a submodule not yet imported
            obj = importlib.import_module(".".join([module, *attributes[: i + 1]]))
    return obj


def _unknown_keywords(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = _imports(tree)
    unknown = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted(node.func)
        if dotted is None or dotted[0] not in names:
            continue
        module, attributes = names[dotted[0]]
        attributes = [*attributes, *dotted[1:]]
        target = ".".join([module, *attributes])
        try:
            parameters = inspect.signature(_resolve(module, attributes)).parameters
        except (AttributeError, ImportError, TypeError, ValueError) as error:
            unknown.append(f"line {node.lineno}: {target} ({type(error).__name__})")
            continue
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            continue
        unknown += [
            f"line {node.lineno}: {target}({k.arg}=...)"
            for k in node.keywords
            if k.arg is not None and k.arg not in parameters
        ]
    return unknown


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_script_calls_tomojax_with_keywords_it_takes(path: Path) -> None:
    unknown = _unknown_keywords(path)
    assert not unknown, f"{path.name} calls TomoJAX in ways it does not support: {unknown}"
