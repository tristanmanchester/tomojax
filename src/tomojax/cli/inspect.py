"""CLI: inspect HDF5/NXtomo files for quick diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from ..data.inspection import (
    format_inspection_report,
    inspect_nxtomo,
    save_projection_quicklook,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect an HDF5/NXtomo (.nxs) file before reconstruction"
    )
    parser.add_argument("input", help="Input .nxs/.h5/.hdf5 file")
    parser.add_argument(
        "--json",
        metavar="PATH",
        default=None,
        help="Write a stable machine-readable inspection report to PATH",
    )
    parser.add_argument(
        "--quicklook",
        metavar="PATH",
        default=None,
        help="Write a percentile-scaled central projection PNG to PATH",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    path = Path(args.input)

    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2
    if not path.is_file():
        print(f"ERROR: not a file: {path}", file=sys.stderr)
        return 2

    try:
        report = inspect_nxtomo(path)
    except Exception as exc:
        print(f"ERROR: could not inspect {path}: {exc}", file=sys.stderr)
        return 1

    print(format_inspection_report(report))

    if args.json is not None:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.quicklook is not None:
        try:
            save_projection_quicklook(path, args.quicklook)
        except Exception as exc:
            print(f"ERROR: could not write quicklook {args.quicklook}: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
