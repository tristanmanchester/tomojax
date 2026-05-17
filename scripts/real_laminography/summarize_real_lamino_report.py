#!/usr/bin/env python3
"""Build the real-laminography staged artifact contract from a staged run."""

from __future__ import annotations

import argparse
from pathlib import Path

from tomojax.bench import build_real_lamino_report


def main(argv: list[str] | None = None) -> int:
    """Run the real-laminography staged report CLI."""
    args = _parse_args(argv)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "real_lamino_report"
    report = build_real_lamino_report(
        run_dir,
        out_dir=out_dir,
        require_success=bool(args.require_success),
    )
    print(f"real_lamino_report: {report['artifacts']['summary_json']}")
    print(f"success: {report['success']['passed']}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--require-success", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
