"""``tomojax inspect``: describe and check a dataset before using it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, cast

from tomojax.cli._options import add_force, check_paths, options
from tomojax.cli._previews import preview_paths, write_previews
from tomojax.cli._progress import ProgressParser, add_progress, run_command
from tomojax.io.api import format_inspection_report

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomojax.io.api import Inspection
    from tomojax.wire.progress import Result


def _build_parser() -> ProgressParser:
    parser = ProgressParser(
        prog="tomojax inspect",
        description=(
            "Describe a dataset or recognized vendor acquisition (projections, angles, geometry, "
            "corrections, any reconstructed volume and memory estimates). Check storage and "
            "report missing geometry and intensity calibration for recognized raw acquisitions."
        ),
        epilog=(
            "Exit status: 0 when the dataset is valid, 1 when it has issues or a preview fails, "
            "2 for a usage error.\n\nExamples:\n"
            "  tomojax inspect scan.nxs\n"
            "  tomojax inspect recon.nxs --preview-dir previews\n"
            "  tomojax inspect scan.nxs --json > report.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ = parser.add_argument(
        "data", metavar="INPUT", help="NXtomo/HDF5 dataset or vendor acquisition file/folder"
    )
    _ = parser.add_argument(
        "--json", action="store_true", help="Print the report as JSON instead of text"
    )
    _ = parser.add_argument(
        "--preview-dir",
        metavar="DIR",
        help="Write PNGs of the central projection (and the volume's central slices) to DIR",
    )
    add_progress(parser)
    add_force(parser)
    for action in options(parser):
        if action.dest == "force":
            action.help = "Replace existing preview files in --preview-dir"
    return parser


def _display(
    report: Inspection, previews: dict[str, str], preview_error: str | None, *, as_json: bool
) -> None:
    issues = report["issues"]
    if as_json:
        payload: dict[str, object] = {**report, "previews": previews}
        if preview_error is not None:
            payload["preview_error"] = preview_error
            print(f"tomojax inspect: error: preview failed: {preview_error}", file=sys.stderr)
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_inspection_report(report))
        if previews:
            print("Previews: " + ", ".join(previews.values()))
        if preview_error is not None:
            print(f"tomojax inspect: error: preview failed: {preview_error}", file=sys.stderr)
        print("Valid: yes" if not issues else f"Issues ({len(issues)}):")
        for issue in issues:
            print(f"  - {issue}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``tomojax inspect``: :func:`tomojax.inspect`, with optional previews."""
    import tomojax as tj

    parser = _build_parser()
    parser.set_invocation(list(sys.argv[1:] if argv is None else argv))
    args = parser.parse_args(argv)
    if args.json and args.progress == "json":
        parser.error("--json cannot print an inspection document with --progress json")

    def run() -> Result | int:
        path = cast("str", args.data)
        preview_dir = cast("str | None", args.preview_dir)
        check_paths(parser, inputs=[path])
        if preview_dir is not None:
            check_paths(
                parser,
                inputs=[path],
                outputs=preview_paths(path, preview_dir).values(),
                force=cast("bool", args.force),
            )

        report = tj.inspect(path)
        issues = report["issues"]
        previews: dict[str, str] = {}
        preview_error: str | None = None
        if preview_dir is not None:
            try:
                previews = {k: str(p) for k, p in write_previews(path, preview_dir).items()}
            except (OSError, ValueError, KeyError) as exc:
                preview_error = str(exc).strip("'\"")

        if args.progress == "json":
            from tomojax.cli._results import report_result

            if preview_error or issues:
                raise ValueError(preview_error or "; ".join(issues))
            return report_result(Path(path), tuple(previews.values()), dict(report))
        _display(report, previews, preview_error, as_json=cast("bool", args.json))
        return int(bool(issues or preview_error))

    return run_command("inspect", args.progress, run)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
