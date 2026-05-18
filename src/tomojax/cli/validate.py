"""CLI: validate HDF5/NXtomo files.

Usage example:

  uv run tomojax validate data/sim.nxs
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TYPE_CHECKING, cast

from tomojax.io import validate_dataset

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class ValidateCommand:
    """Typed command plan for dataset validation."""

    input_path: Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate an HDF5/NXtomo (.nxs) file")
    _ = p.add_argument("input", help="Input .nxs/.h5/.hdf5 file")
    return p


def _parse_command(argv: Sequence[str] | None) -> ValidateCommand:
    """Parse CLI arguments into a typed validation command plan."""
    args = _build_parser().parse_args(argv)
    return ValidateCommand(input_path=Path(cast("str", args.input)))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dataset validation command."""
    command = _parse_command(argv)
    path = command.input_path

    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2
    if not path.is_file():
        print(f"ERROR: not a file: {path}", file=sys.stderr)
        return 2

    report = validate_dataset(path)
    issues = report["issues"]
    if not issues:
        print(f"OK: {path}")
        return 0

    issue_word = "issue" if len(issues) == 1 else "issues"
    print(f"INVALID: {path} ({len(issues)} {issue_word})")
    for issue in issues:
        print(f"- {issue}")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
