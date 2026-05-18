"""Check that private TomoJAX modules stay behind their public facades."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

ALLOW_PRIVATE_MARKER = "check-public-imports: allow-private"
DEFAULT_SCAN_PATHS = [Path("src/tomojax"), Path("tests"), Path("examples")]
ALIGNMENT_FACADE_REASON = "nested alignment namespace must be reached through tomojax.align.api"
BLOCKED_ALIGNMENT_NAMESPACES = (
    "tomojax.align._geometry",
    "tomojax.align._model",
    "tomojax.align._objectives",
    "tomojax.align.geometry",
    "tomojax.align.io",
    "tomojax.align.model",
    "tomojax.align.objectives",
    "tomojax.align.pipeline",
)
PRODUCT_BOUNDARY_IMPORT_RULES = (
    (
        "tomojax.data",
        "data namespace must be reached through tomojax.io or tomojax.datasets",
    ),
    (
        "tomojax._data",
        "private data implementation must be reached through tomojax.io or tomojax.datasets",
    ),
    (
        "tomojax.core.geometry",
        "core geometry namespace must be reached through tomojax.geometry",
    ),
)
ALLOWED_PRODUCT_SURFACE_IMPORTS: set[tuple[str, str]] = set()


@dataclass(frozen=True)
class Violation:
    """A TomoJAX import that violates a public architecture boundary."""

    path: Path
    line: int
    imported_module: str
    importing_module: str
    reason: str

    def format(self, root: Path) -> str:
        """Format a violation for terminal output."""
        try:
            relative_path = self.path.relative_to(root)
        except ValueError:
            relative_path = self.path
        return (
            f"{relative_path}:{self.line}: {self.reason}: "
            f"{self.importing_module} imports {self.imported_module}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the private-import boundary check."""
    parser = argparse.ArgumentParser(
        description=(
            "Reject imports of tomojax.<owner>._private modules from outside tomojax.<owner>. "
            "Also reject private TomoJAX namespaces from product-facing modules. "
            "Tests may allow a specific white-box import with "
            f"'# {ALLOW_PRIVATE_MARKER}'."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=DEFAULT_SCAN_PATHS,
        help="Files or directories to scan.",
    )
    args = parser.parse_args(argv)

    root = Path.cwd()
    paths = [path if path.is_absolute() else root / path for path in args.paths]
    violations = find_violations(paths, root)
    if violations:
        for violation in violations:
            print(violation.format(root), file=sys.stderr)
        print(
            "\nPrivate implementation modules must be imported through their owner's public API. "
            "For a deliberate white-box test, keep the import local and add "
            f"'# {ALLOW_PRIVATE_MARKER}' on or directly above the import line.",
            file=sys.stderr,
        )
        return 1
    return 0


def find_violations(paths: Iterable[Path], root: Path) -> list[Violation]:
    """Find cross-boundary private imports below the given paths."""
    violations: list[Violation] = []
    for path in iter_python_files(paths):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        lines = source.splitlines()
        importing_module = module_name_for_path(path, root)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported_modules = imported_modules_from_import_from(node)
            else:
                continue

            if line_has_allowed_test_marker(lines, node.lineno, importing_module):
                continue

            for imported_module in imported_modules:
                if product_surface_import_is_allowed(importing_module, imported_module):
                    continue
                boundary_reason = product_boundary_reason(imported_module, importing_module)
                if boundary_reason is not None:
                    violations.append(
                        Violation(
                            path=path,
                            line=node.lineno,
                            imported_module=imported_module,
                            importing_module=importing_module,
                            reason=boundary_reason,
                        )
                    )
                    continue
                owner = private_owner(imported_module)
                if owner is not None and not module_is_inside_owner(importing_module, owner):
                    violations.append(
                        Violation(
                            path=path,
                            line=node.lineno,
                            imported_module=imported_module,
                            importing_module=importing_module,
                            reason=f"private import crosses tomojax.{owner} boundary",
                        )
                    )
                    continue
    return violations


def iter_python_files(paths: Iterable[Path]) -> Iterable[Path]:
    """Yield Python files from file or directory arguments."""
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            yield path
        elif path.is_dir():
            yield from sorted(path.rglob("*.py"))


def module_name_for_path(path: Path, root: Path) -> str:
    """Return a dotted module name for source and test files."""
    try:
        relative_path = path.relative_to(root)
    except ValueError:
        prefix = "tests" if path.name.startswith("test_") else "external"
        return f"{prefix}.{path.with_suffix('').name}"
    if relative_path.parts[:2] == ("src", "tomojax"):
        module_parts = relative_path.with_suffix("").parts[1:]
    else:
        module_parts = relative_path.with_suffix("").parts
    if module_parts[-1] == "__init__":
        module_parts = module_parts[:-1]
    return ".".join(module_parts)


def imported_modules_from_import_from(node: ast.ImportFrom) -> list[str]:
    """Expand an absolute from-import into concrete imported module candidates."""
    if node.level:
        return []
    module = node.module or ""
    imported_modules = [module] if module else []
    if module == "tomojax":
        imported_modules.extend(f"tomojax.{alias.name}" for alias in node.names if alias.name)
    if is_tomojax_owner_module(module):
        imported_modules.extend(f"{module}.{alias.name}" for alias in node.names if alias.name)
    return imported_modules


def private_owner(module: str) -> str | None:
    """Return the top-level TomoJAX owner for a private implementation module."""
    parts = module.split(".")
    if len(parts) < 3 or parts[0] != "tomojax":
        return None

    owner = parts[1]
    for part in parts[2:]:
        if part.startswith("_") and part != "__init__":
            return owner
    return None


def is_tomojax_owner_module(module: str) -> bool:
    """Return whether a module is exactly tomojax.<owner>."""
    parts = module.split(".")
    return len(parts) == 2 and parts[0] == "tomojax"


def module_is_inside_owner(module: str, owner: str) -> bool:
    """Return whether a module belongs to the same top-level TomoJAX owner."""
    return module == f"tomojax.{owner}" or module.startswith(f"tomojax.{owner}.")


def product_boundary_reason(imported_module: str, importing_module: str) -> str | None:
    """Return a violation reason for imports outside product-facing boundaries."""
    if not is_product_surface_module(importing_module):
        return None
    for namespace, reason in PRODUCT_BOUNDARY_IMPORT_RULES:
        if module_matches_namespace(imported_module, namespace):
            return reason
    if importing_module.startswith("tomojax.cli.") and module_matches_namespace(
        imported_module, "tomojax.bench"
    ):
        return "production CLI modules must not import benchmark helpers"
    for blocked_align_namespace in BLOCKED_ALIGNMENT_NAMESPACES:
        if module_matches_namespace(imported_module, blocked_align_namespace):
            return ALIGNMENT_FACADE_REASON
    return None


def module_matches_namespace(module: str, namespace: str) -> bool:
    """Return whether a module is a namespace or one of its descendants."""
    return module == namespace or module.startswith(f"{namespace}.")


def is_product_surface_module(module: str) -> bool:
    """Return whether a module belongs to user-facing code, tests, or examples."""
    return module.startswith(("tomojax.cli.", "tests.", "examples."))


def product_surface_import_is_allowed(importing_module: str, imported_module: str) -> bool:
    """Return whether a product-surface import has a narrow explicit exception."""
    return any(
        importing_module == allowed_importer
        and (imported_module == allowed_import or imported_module.startswith(f"{allowed_import}."))
        for allowed_importer, allowed_import in ALLOWED_PRODUCT_SURFACE_IMPORTS
    )


def line_has_allowed_test_marker(lines: list[str], lineno: int, importing_module: str) -> bool:
    """Return whether an import line has the explicit private-import allow marker."""
    if not importing_module.startswith("tests."):
        return False
    candidates = [lineno - 1, lineno - 2]
    return any(0 <= idx < len(lines) and ALLOW_PRIVATE_MARKER in lines[idx] for idx in candidates)


if __name__ == "__main__":
    raise SystemExit(main())
