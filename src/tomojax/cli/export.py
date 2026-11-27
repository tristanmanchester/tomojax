"""Export a registered member of a saved file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

from tomojax.cli._options import add_output, check_paths
from tomojax.cli._progress import ProgressParser, add_progress, run_command
from tomojax.errors import InvalidParameter
from tomojax.io.api import format_descriptions, plan_export

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomojax.wire.progress import Result


def _build_parser() -> ProgressParser:
    formats = format_descriptions()
    suffixes = ", ".join((f["suffixes"] or ["slice directory"])[0] for f in formats)
    parser = ProgressParser(
        prog="tomojax export", description=f"Export a registered array. Destinations: {suffixes}."
    )
    _ = parser.add_argument("data", metavar="INPUT", help="Saved TomoJAX file")
    add_progress(parser)
    add_output(parser, "File or slice directory")
    _ = parser.add_argument("--member", help="Array in the file (listed by inspect)")
    _ = parser.add_argument("--dry-run", action="store_true", help="Print the plan without writing")
    enabled = {name for f in formats for name in f["options"]}
    fields = {
        "dtype": {"choices": ["float32", "uint16", "uint8"], "help": "Stored sample type"},
        "range": {
            "type": float,
            "nargs": 2,
            "metavar": ("LOW", "HIGH"),
            "help": "Physical bounds for integer conversion",
        },
        "axis": {"help": "Axis to stack first (default: stored order)"},
        "binning": {"type": int, "default": 1, "help": "Mean-bin spatial axes by this factor"},
        "compress": {"action": argparse.BooleanOptionalAction, "help": "Use lossless compression"},
        "crop": {"metavar": "LO:HI,LO:HI,LO:HI", "help": "Half-open bounds in storage order"},
    }
    for name, options in fields.items():
        if name in enabled:
            parser.add_argument("--" + name, **options)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the registered array export command."""
    from tomojax._loading.files import saved_export_source
    from tomojax.cli._results import report_result
    from tomojax.cli.correct import crop_intervals

    parser = _build_parser()
    parser.set_invocation(list(sys.argv[1:] if argv is None else argv))
    args = parser.parse_args(argv)

    def run() -> Result | None:
        data, out = Path(args.data), Path(args.out)
        check_paths(parser, inputs=[data])
        from tomojax.cli._provenance import recording

        with (
            recording("export", args, {}),
            saved_export_source(data, args.member) as (source, disk),
        ):
            try:
                crop = None if args.crop is None else crop_intervals(args.crop, "Z0:Z1,Y0:Y1,X0:X1")
                plan = plan_export(
                    out,
                    source,
                    dtype=args.dtype,
                    value_range=args.range,
                    crop=crop,
                    binning=args.binning,
                    axis=args.axis,
                    compress=args.compress,
                )
            except InvalidParameter as exc:
                parser.error(str(exc))
            paths = plan.format.paths(plan.path)
            check_paths(
                parser,
                inputs=[data],
                outputs=plan.pending_paths(),
                force=args.force or args.dry_run,
            )
            report = plan.details if args.dry_run else plan.publish(force=args.force)
        if args.progress == "json":
            return report_result(
                plan.path,
                () if args.dry_run else tuple(map(str, paths)),
                report,
                arrays={f"source_{source.name}": disk},
            )
        print(json.dumps(report, indent=2))
        return None

    return run_command("export", args.progress, run)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
