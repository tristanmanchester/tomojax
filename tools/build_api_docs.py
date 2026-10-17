"""Build the Python reference from the same exported objects as the docstring lint."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import TYPE_CHECKING

from sphinx.cmd.build import build_main

from check_docstrings import PUBLIC_DATA, PUBLIC_MODULES, public_data, public_objects

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Generate API sources and build HTML under .artifacts/docs."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    source = root / ".artifacts" / "api-source"
    output = root / ".artifacts" / "docs"
    source.mkdir(parents=True, exist_ok=True)
    pages: dict[str, list[str]] = {
        module: [module, "=" * len(module), ""] for module in PUBLIC_MODULES
    }
    for name, obj in public_objects():
        # autoclass renders the public methods/properties alongside its constructor.
        module = name.rsplit(".", 1)[0]
        if module not in pages:
            continue
        directive = "autoclass" if inspect.isclass(obj) else "autofunction"
        pages[module].extend([f".. {directive}:: {name}", ""])
        if inspect.isclass(obj):
            pages[module].extend(["   :members:", "   :undoc-members:", ""])
    for name in public_data():
        module = name.rsplit(".", 1)[0]
        pages[module].extend([f".. py:data:: {name}", "", f"   {PUBLIC_DATA[name]}", ""])
    index = [
        "TomoJAX Python reference",
        "========================",
        "",
        "Volumes use ``(x, y, z)`` and projections use ``(view, v, u)``.",
        "Lengths share one physical unit; rotation angles are in degrees unless stated otherwise.",
        "",
        ".. toctree::",
        "   :maxdepth: 2",
        "",
    ]
    for module, lines in pages.items():
        index.append(f"   {module}")
        (source / f"{module}.rst").write_text("\n".join(lines), encoding="utf-8")
    (source / "index.rst").write_text("\n".join(index) + "\n", encoding="utf-8")
    return build_main(
        [
            "-W",
            "--keep-going",
            "-b",
            "html",
            "-c",
            str(root / "docs" / "api"),
            str(source),
            str(output),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
