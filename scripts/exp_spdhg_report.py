from __future__ import annotations

"""Report helper for manual SPDHG experiment runs."""

import argparse
from pathlib import Path

from tomojax.bench import build_spdhg_experiment_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize benchmark results and emit a Markdown report"
    )
    parser.add_argument(
        "--indir",
        default="runs/exp_spdhg_256",
        type=Path,
        help="Directory with metrics.json and images",
    )
    parser.add_argument(
        "--out",
        default=None,
        type=Path,
        help="Output Markdown path (default: indir/REPORT.md)",
    )
    args = parser.parse_args()

    out_path = build_spdhg_experiment_report(args.indir, out=args.out)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
