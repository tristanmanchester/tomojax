"""CLI: inspect HDF5/NXtomo files for quick diagnosis."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, cast

from tomojax.io.api import (
    format_inspection_report,
    inspect_dataset,
    save_projection_quicklook,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class InspectCommand:
    """Typed command plan for dataset inspection."""

    input_path: Path
    json_path: Path | None
    quicklook_path: Path | None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect an HDF5/NXtomo (.nxs) file before reconstruction"
    )
    _ = parser.add_argument("input", help="Input .nxs/.h5/.hdf5 file")
    _ = parser.add_argument(
        "--json",
        metavar="PATH",
        default=None,
        help="Write a stable machine-readable inspection report to PATH",
    )
    _ = parser.add_argument(
        "--quicklook",
        metavar="PATH",
        default=None,
        help="Write a percentile-scaled central projection PNG to PATH",
    )
    return parser


def _parse_command(argv: Sequence[str] | None) -> InspectCommand:
    """Parse CLI arguments into a typed inspection command plan."""
    args = _build_parser().parse_args(argv)
    json_arg = cast("str | None", args.json)
    quicklook_arg = cast("str | None", args.quicklook)
    return InspectCommand(
        input_path=Path(cast("str", args.input)),
        json_path=Path(json_arg) if json_arg is not None else None,
        quicklook_path=Path(quicklook_arg) if quicklook_arg is not None else None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dataset inspection command."""
    command = _parse_command(argv)
    path = command.input_path

    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2
    if not path.is_file():
        print(f"ERROR: not a file: {path}", file=sys.stderr)
        return 2

    report = inspect_dataset(path)
    print(format_inspection_report(report))

    if command.json_path is not None:
        json_path = command.json_path
        json_path.parent.mkdir(parents=True, exist_ok=True)
        _ = json_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if command.quicklook_path is not None:
        _ = save_projection_quicklook(path, command.quicklook_path)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
