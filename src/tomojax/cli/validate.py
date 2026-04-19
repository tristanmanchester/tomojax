"""CLI: validate HDF5/NXtomo files.

Usage example:

  uv run tomojax-validate data/sim.nxs
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from ..data.io_hdf5 import validate_nxtomo


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate an HDF5/NXtomo (.nxs) file")
    p.add_argument("input", help="Input .nxs/.h5/.hdf5 file")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    path = Path(args.input)

    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2
    if not path.is_file():
        print(f"ERROR: not a file: {path}", file=sys.stderr)
        return 2

    report = validate_nxtomo(str(path))
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
