"""Record the public API and the code-health ratchets that tests hold the code to.

``tests/guardrails/api_surface.txt`` lists every public name of the user-facing
packages with its signature, so an API change shows up as a diff in review.
``tests/guardrails/ratchets.json`` records measures of debt (configuration
fields, exported names, CLI flags, CUDA kernels and their compile-time switches,
Python typing debt, Studio Rust debt, long files, and other measures) that may
fall but not rise. The console report also shows informational lines per package.

After an intended change run ``python tools/guardrails.py --write`` and commit
the updated files with it; ``tests/test_architecture.py`` fails until then.
"""
# Introspecting arbitrary modules and JSON is untyped by nature.
# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownParameterType=false

from __future__ import annotations

import argparse
import ast
from bisect import bisect_left, bisect_right
from collections import Counter
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
    from collections.abc import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "tomojax"
STUDIO_CRATES = ROOT / "studio" / "crates"
OUT = ROOT / "tests" / "guardrails"
SURFACE_FILE = OUT / "api_surface.txt"
RATCHET_FILE = OUT / "ratchets.json"

# Packages users import; their __all__ is the public API.
PUBLIC_PACKAGES = (
    "tomojax",
    "tomojax.geometry",
    "tomojax.evaluation",
    "tomojax.progress",
    "tomojax.interactive",
    "tomojax.corrections",
    "tomojax.forward",
    "tomojax.recon",
    "tomojax.io",
    "tomojax.alignment",
    "tomojax.datasets",
)
# Expert modules: not snapshotted name by name, but their size is ratcheted.
EXPERT_MODULES = (
    "tomojax.geometry.api",
    "tomojax.corrections.api",
    "tomojax.recon.api",
    "tomojax.io.api",
    "tomojax.alignment.api",
)
CLI_COMMANDS = (
    "inspect",
    "correct",
    "reconstruct",
    "align",
    "export",
    "simulate",
    "serve",
    "mcp",
    "studio",
    "skill",
)
# Files and functions longer than these are recorded with their length, which
# may only fall: debt inside an already long one counts too.
LONG_FILE_LINES = 800
LONG_FUNCTION_LINES = 100
# A private function used once and no longer than this has to earn its name.
TINY_HELPER_LINES = 5
# Single-use helpers this many single-use callers deep are one function in pieces.
HELPER_CHAIN_DEPTH = 2
# Public functions whose leading optional argument is deliberately positional.
POSITIONAL_OPTIONS_ALLOWED = {"tomojax.reconstruct": ("method",), "tomojax.simulate": ("phantom",)}
# Preprocessor conditionals in kernel source; Python comments carry a space.
CUDA_SWITCH = re.compile(r"^\s*#if(?:n?def)?\b", re.MULTILINE)

# Keep literals as single opaque tokens. A Rust lifetime such as 'a is distinct
# from a character literal, and raw strings close with the opening hash count.
RUST_TOKEN = re.compile(
    r'//[^\n]*|/\*|(?:br|cr|r)(?P<hashes>\#*)".*?"(?P=hashes)'
    r'|"(?:\\.|[^"\\])*"'
    r"|'(?:\\(?:u\{[\da-fA-F_]+\}|x[\da-fA-F]{2}|.)|[^'\\\n])'"
    r"|(?:r\#)?[^\W\d]\w*|->|\S",
    re.DOTALL,
)
RUST_BLOCK_COMMENT = re.compile(r"/\*|\*/")
RUST_GROUP_END = {"(": ")", "[": "]", "{": "}"}


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
        own = {k for k in vars(obj) if not k.startswith("_") or k == "__call__"}
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


def _optional_api_surface(module: str) -> list[str]:
    """Record an optional module's declared API without importing its dependency."""
    path = ROOT / "src" / (module.replace(".", "/") + ".py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    exports = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )
    lines = [f"# {module}"]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in exports:
            result = "" if node.returns is None else f" -> {ast.unparse(node.returns)}"
            lines.append(f"def {module}.{node.name}({ast.unparse(node.args)}){result}")
            continue
        if not isinstance(node, ast.ClassDef) or node.name not in exports:
            continue
        methods = [item for item in node.body if isinstance(item, ast.FunctionDef)]
        constructor = next(item for item in methods if item.name == "__init__")
        constructor.args.args = constructor.args.args[1:]  # omit self, as inspect.signature does
        lines.append(f"class {module}.{node.name}({ast.unparse(constructor.args)})")
        for method in sorted(methods, key=lambda item: item.name):
            if not method.name.startswith("_"):
                result = "" if method.returns is None else f" -> {ast.unparse(method.returns)}"
                lines.append(f"    .{method.name}({ast.unparse(method.args)}){result}")
    return [*lines, ""]


def api_surface() -> str:
    """Every public name of :data:`PUBLIC_PACKAGES`, with signatures."""
    lines: list[str] = []
    for module_name in PUBLIC_PACKAGES:
        module = importlib.import_module(module_name)
        lines.append(f"# {module_name}")
        for name in sorted(module.__all__):
            lines.extend(_describe(module_name, name, getattr(module, name)))
        lines.append("")
    lines.extend(_optional_api_surface("tomojax.torch"))
    return "\n".join(lines)


# ---------------------------------------------------------------------- ratchets


def _python_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def _source_files() -> list[Path]:
    """Python and CUDA sources; many kernels are embedded in Python strings."""
    suffixes = {".py", ".cu", ".cuh"}
    return sorted(
        p for p in SRC.rglob("*") if p.suffix in suffixes and "__pycache__" not in p.parts
    )


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _count(pattern: str | re.Pattern[str], files: list[Path] | None = None) -> int:
    regex = re.compile(pattern)
    paths = _python_files() if files is None else files
    return sum(len(regex.findall(p.read_text(encoding="utf-8"))) for p in paths)


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


def _code_line_count(source: str, node: ast.AST | None = None) -> int:
    """Count physical lines except docstrings; comments and code still count."""
    # AST column offsets count UTF-8 bytes, not Python string characters.
    lines = source.encode("utf-8").splitlines()
    node = ast.parse(source) if node is None else node
    start = getattr(node, "lineno", 1)
    stop = getattr(node, "end_lineno", len(lines))
    prose: set[int] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            if ast.get_docstring(child) is None:
                continue
            doc = child.body[0]
            end = doc.end_lineno or doc.lineno
            prose.update(range(doc.lineno, end + 1))
            # A one-line definition, trailing statement or comment is still code.
            if lines[doc.lineno - 1][: doc.col_offset].strip():
                prose.discard(doc.lineno)
            if lines[end - 1][doc.end_col_offset :].strip():
                prose.discard(end)
    return stop - start + 1 - len(prose)


def _file_lines() -> dict[str, int]:
    """Files over :data:`LONG_FILE_LINES`, excluding docstring spans."""
    lengths = {
        _relative(p): _code_line_count(p.read_text(encoding="utf-8")) for p in _python_files()
    }
    return {path: n for path, n in lengths.items() if n > LONG_FILE_LINES}


def _code_lines() -> dict[str, int]:
    """Report lines per top-level package, excluding Python docstrings."""
    counts: dict[str, int] = {}
    for path in _source_files():
        parts = path.relative_to(SRC).parts
        package = parts[0] if len(parts) > 1 else "(modules)"
        source = path.read_text(encoding="utf-8")
        lines = _code_line_count(source) if path.suffix == ".py" else len(source.splitlines())
        counts[package] = counts.get(package, 0) + lines
    return dict(sorted(counts.items()))


def _pyright_warnings() -> dict[str, int]:
    """Count warnings per rule in the locked CPU dev environment used by CI."""
    # Optional dependencies such as CuPy change both import and inference warnings.
    # Give pyright the isolated interpreter explicitly so it cannot find .venv instead.
    command = [
        "uv",
        "run",
        "--isolated",
        "--locked",
        "--extra",
        "cpu",
        "--dev",
        "--python",
        "3.12",
        "sh",
        "-c",
        'exec basedpyright --outputjson --pythonpath "$(command -v python)"',
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"basedpyright warning scan failed:\n{result.stderr}{result.stdout}")
    counts: Counter[str] = Counter()
    for diagnostic in json.loads(result.stdout)["generalDiagnostics"]:
        if diagnostic["severity"] != "warning":
            continue
        path = Path(diagnostic["file"])
        if path.is_relative_to(SRC) or path.is_relative_to(ROOT / "tools"):
            counts[diagnostic["rule"]] += 1
    return dict(sorted(counts.items()))


def _typing_reference(node: ast.AST, name: str, aliases: set[str], modules: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return isinstance(node.ctx, ast.Load) and node.id in aliases
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.ctx, ast.Load)
        and isinstance(node.value, ast.Name)
        and node.value.id in modules
        and node.attr == name
    )


def _python_debt(sources: Iterable[str]) -> dict[str, int]:
    """Count explicit typing escape hatches and broad handlers in source ASTs."""
    counts = {"explicit_any": 0, "type_casts": 0, "broad_except": 0}
    for source in sources:
        nodes = list(ast.walk(ast.parse(source)))
        typing_modules = {"typing", "typing_extensions"}
        modules = typing_modules.copy()
        names = {"Any": {"Any"}, "cast": {"cast"}}
        for node in nodes:
            if isinstance(node, ast.Import):
                modules.update(
                    a.asname for a in node.names if a.name in typing_modules and a.asname
                )
            elif isinstance(node, ast.ImportFrom) and node.module in typing_modules:
                for alias in node.names:
                    if alias.name in names:
                        names[alias.name].add(alias.asname or alias.name)

        for node in nodes:
            counts["explicit_any"] += _typing_reference(node, "Any", names["Any"], modules)
            if isinstance(node, ast.Call):
                counts["type_casts"] += _typing_reference(node.func, "cast", names["cast"], modules)
            elif isinstance(node, ast.ExceptHandler):
                caught = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
                counts["broad_except"] += node.type is None or any(
                    isinstance(kind, ast.Name) and kind.id == "Exception" for kind in caught
                )
    return counts


def _function_lines() -> dict[str, int]:
    """Long functions as ``path::qualified.name``, excluding docstring spans."""
    found: dict[str, int] = {}

    def visit(node: ast.AST, prefix: str, path: str, source: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = f"{prefix}{child.name}"
                if not isinstance(child, ast.ClassDef):
                    length = _code_line_count(source, child)
                    if length > LONG_FUNCTION_LINES:
                        found[f"{path}::{name}"] = length
                visit(child, f"{name}.", path, source)

    for p in _python_files():
        source = p.read_text(encoding="utf-8")
        visit(ast.parse(source), "", _relative(p), source)
    return dict(sorted(found.items()))


def _rust_tokens(source: str) -> list[re.Match[str]]:
    """Read Rust tokens and their offsets, dropping line and nested block comments."""
    tokens: list[re.Match[str]] = []
    offset = 0
    while (match := RUST_TOKEN.search(source, offset)) is not None:
        offset = match.end()
        if match[0] == "/*":
            depth = 1
            for marker in RUST_BLOCK_COMMENT.finditer(source, offset):
                depth += 1 if marker[0] == "/*" else -1
                if depth == 0:
                    offset = marker.end()
                    break
        elif not match[0].startswith("//"):
            tokens.append(match)
    return tokens


def _rust_groups(tokens: list[str]) -> dict[int, int]:
    """Map opening delimiters to their matching token indices."""
    groups: dict[int, int] = {}
    stack: list[int] = []
    for index, token in enumerate(tokens):
        if token in RUST_GROUP_END:
            stack.append(index)
        elif stack and token == RUST_GROUP_END[tokens[stack[-1]]]:
            groups[stack.pop()] = index
    return groups


def _rust_item_end(
    tokens: list[str], groups: dict[int, int], start: int, *, statement: bool = False
) -> int | None:
    """Find an item's terminal semicolon or closing body brace."""
    index = start
    angles = 0
    while index < len(tokens):
        token = tokens[index]
        if token == ";":
            return index
        if token in {"}", ")", "]"}:
            break
        if index in groups:
            end = groups[index]
            if token == "{" and angles == 0 and not statement:
                return end
            index = end
        elif not statement:
            # Grouped const expressions in generic arguments are part of the
            # signature. Parentheses and brackets above skip their contents.
            if token == "<":
                angles += 1
            elif token == ">" and angles:
                angles -= 1
        index += 1
    return None


def _rust_test_items(tokens: list[str], groups: dict[int, int]) -> dict[int, int]:
    """Find spans of items guarded by an outer cfg(test) attribute."""
    items: dict[int, int] = {}
    index = 0
    while index < len(tokens):
        start = index
        test_only = False
        while tokens[index : index + 2] == ["#", "["] and index + 1 in groups:
            end = groups[index + 1]
            test_only |= tokens[index + 2 : end] == ["cfg", "(", "test", ")"]
            index = end + 1
        if test_only and index < len(tokens):
            head = index
            if tokens[head] == "pub":
                head = groups.get(head + 1, head) + 1
            keyword = tokens[head]
            statement = keyword in {"use", "type", "static", "let"} or (
                keyword == "const" and tokens[head + 1] not in {"fn", "unsafe", "extern"}
            )
            end = _rust_item_end(tokens, groups, index, statement=statement)
            if end is not None:
                items[start] = end
                index = end + 1
        elif index == start:
            index += 1
    return items


def _rust_production_debt(
    source: str, matches: list[re.Match[str]], tokens: list[str]
) -> tuple[int, dict[str, list[int]]]:
    """Return production lines and long function lengths by name."""
    groups = _rust_groups(tokens)
    test_items = _rust_test_items(tokens, groups)
    newlines = [match.start() for match in re.finditer("\n", source)]
    excluded_lines: set[int] = set()
    for start, end in test_items.items():
        start_offset = matches[start].start()
        end_offset = matches[end].end()
        first = bisect_left(newlines, start_offset)
        last = bisect_left(newlines, end_offset)
        # A line shared with production text still counts toward its length.
        line_start = newlines[first - 1] + 1 if first else 0
        line_end = newlines[last] if last < len(newlines) else len(source)
        first += bool(source[line_start:start_offset].strip())
        last -= bool(source[end_offset:line_end].strip())
        excluded_lines.update(range(first, last + 1))
    excluded = sorted(excluded_lines)

    functions: dict[str, list[int]] = {}
    index = 0
    while index < len(tokens):
        if index in test_items:
            index = test_items[index] + 1
            continue
        if tokens[index] == "fn" and tokens[index + 2 : index + 3] in (["("], ["<"]):
            name = tokens[index + 1]
            end = (
                _rust_item_end(tokens, groups, index)
                if name.removeprefix("r#").isidentifier()
                else None
            )
            if end is not None and tokens[end] == "}":
                first = bisect_left(newlines, matches[index].start())
                last = bisect_left(newlines, matches[end].start())
                length = last - first + 1
                length -= bisect_right(excluded, last) - bisect_left(excluded, first)
                if length > LONG_FUNCTION_LINES:
                    functions.setdefault(name, []).append(length)
        index += 1
    return len(source.splitlines()) - len(excluded), functions


def _studio_debt() -> dict[str, object]:
    """Measure lint opt-outs in every Studio crate and long files and functions in production."""
    allows = 0
    long_files: dict[str, int] = {}
    long_functions: dict[str, int] = {}
    sources = (*STUDIO_CRATES.glob("*/src/**/*.rs"), *STUDIO_CRATES.glob("*/tests/**/*.rs"))
    for path in sorted(sources):
        source = path.read_text(encoding="utf-8")
        matches = _rust_tokens(source)
        tokens = [match[0] for match in matches]
        for i, token in enumerate(tokens, start=1):
            if token == "#":
                start = i + (tokens[i : i + 1] == ["!"])
                allows += tokens[start : start + 3] in (["[", "allow", "("], ["[", "expect", "("])
        if (
            path.name == "tests.rs"
            or path.name.endswith("_tests.rs")
            or path.relative_to(STUDIO_CRATES).parts[1] == "tests"
        ):
            continue
        length, functions = _rust_production_debt(source, matches, tokens)
        relative = _relative(path)
        if length > LONG_FILE_LINES:
            long_files[relative] = length
        for name, lengths in functions.items():
            # A file can implement the same method in several impl blocks.
            for rank, length in enumerate(sorted(lengths, reverse=True), start=1):
                suffix = f"#{rank}" if rank > 1 else ""
                long_functions[f"{relative}::{name}{suffix}"] = length
    return {
        "studio_allows": allows,
        "studio_long_files": long_files,
        "studio_long_functions": dict(sorted(long_functions.items())),
    }


def _config_fields() -> dict[str, int]:
    """Field count of every public ``*Config`` dataclass."""
    counts: dict[str, int] = {}
    for module in (*PUBLIC_PACKAGES, *EXPERT_MODULES):
        imported = importlib.import_module(module)
        for name in imported.__all__:
            obj = getattr(imported, name, None)
            if dataclasses.is_dataclass(obj) and name.endswith("Config"):
                counts[name] = len(dataclasses.fields(obj))
    return dict(sorted(counts.items()))


def _body(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    """The statements of ``function`` after its docstring."""
    return function.body[1:] if ast.get_docstring(function) is not None else function.body


def _is_forwarder(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether ``function`` only calls something else with its arguments as they came."""
    body = _body(function)
    if len(body) != 1 or not isinstance(body[0], ast.Return | ast.Expr):
        return False
    call = body[0].value
    if not isinstance(call, ast.Call):
        return False
    arguments = (*call.args, *(keyword.value for keyword in call.keywords))
    plain = ast.Name | ast.Constant | ast.Attribute | ast.Starred
    return all(isinstance(argument, plain) for argument in arguments)


def _helper_debt(sources: Iterable[str]) -> dict[str, int]:
    """Count single-use private functions: tiny ones, forwarders and those in chains.

    References are matched by name across all sources, so this measures the code's
    shape rather than proving a call graph. A chain is what a length limit tempts one
    to write: each link has one caller, and reading one path still means reading them all.
    """
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    references: Counter[str] = Counter()
    referrers: dict[str, str] = {}

    def visit(node: ast.AST, owner: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            inner = owner
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                functions.append(child)
                inner = child.name
            name = child.id if isinstance(child, ast.Name) else getattr(child, "attr", None)
            if name is not None and isinstance(getattr(child, "ctx", None), ast.Load):
                references[name] += 1
                if owner is not None:
                    referrers[name] = owner
            visit(child, inner)

    for source in sources:
        visit(ast.parse(source), None)

    def private(name: str) -> bool:
        return name.startswith("_") and not name.startswith("__")

    helpers = [f for f in functions if private(f.name) and references[f.name] == 1]
    single_use = {f.name for f in helpers}

    def depth(name: str) -> int:
        seen = {name}
        while (name := referrers.get(name, "")) in single_use and name not in seen:
            seen.add(name)
        return len(seen) - 1

    def lines(function: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
        body = _body(function)
        return (body[-1].end_lineno or 0) - body[0].lineno + 1 if body else 0

    return {
        "tiny": sum(lines(f) <= TINY_HELPER_LINES for f in helpers),
        "forwarders": sum(_is_forwarder(f) for f in helpers),
        "chained": sum(depth(name) >= HELPER_CHAIN_DEPTH for name in single_use),
    }


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
        module = importlib.import_module(f"tomojax.cli.{command}")
        build = getattr(module, "build_parser", None) or module._build_parser  # noqa: SLF001
        counts[command] = sum(1 for action in options(build()) if action.option_strings)
    return counts


def ratchets() -> dict[str, object]:
    """Current values of every ratcheted measure."""
    return {
        "config_fields": _config_fields(),
        "exported_names": {
            module: len(importlib.import_module(module).__all__)
            for module in (*PUBLIC_PACKAGES, *EXPERT_MODULES)
        },
        "cli_flags": _cli_flags(),
        "long_files": _file_lines(),
        "long_functions": _function_lines(),
        "pyright_warnings": _pyright_warnings(),
        **_python_debt(p.read_text(encoding="utf-8") for p in _python_files()),
        **_studio_debt(),
        "cuda_kernels": _count(r"__global__", _source_files()),
        "cuda_switches": _count(CUDA_SWITCH, _source_files()),
        # Each is a setting that bypasses the public configuration.
        "environment_reads": _count(r"\bos\.(?:environ|getenv)\b"),
        "suppressions": {
            "noqa": _count(r"#\s*noqa"),
            "type: ignore": _count(r"#\s*type:\s*ignore"),
            "pyright: ignore": _count(r"#\s*pyright:\s*ignore"),
        },
        "complex_functions": _complex_functions(),
        "single_use_helpers": _helper_debt(p.read_text(encoding="utf-8") for p in _python_files()),
        "positional_options": _positional_options(),
        # The internal five-column pose array; public poses have six columns.
        "params5_mentions": _count(r"\bparams5\b"),
        # Each normalizes a user string by hand instead of through one parser.
        "string_normalizers": _count(r"\.replace\(\"[-_]\", \"[-_]\"\)"),
        # Code assuming one device, now that work can be shared among several.
        "first_device_probes": _count(r"jax\.devices\(\)\[0\]"),
        # jaxlib's private modules: none needed.
        "jaxlib_private_imports": _count(r"\bjaxlib\._"),
        "test_private_imports": _test_private_imports(),
    }


# -------------------------------------------------------------------------- main


def write() -> None:
    """Rewrite the recorded surface and ratchets from the current code."""
    surface = api_surface()
    measures = ratchets()
    OUT.mkdir(parents=True, exist_ok=True)
    _ = SURFACE_FILE.write_text(surface, encoding="utf-8")
    _ = RATCHET_FILE.write_text(json.dumps(measures, indent=2) + "\n", encoding="utf-8")


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
    print(json.dumps({**ratchets(), "code_lines": _code_lines()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
