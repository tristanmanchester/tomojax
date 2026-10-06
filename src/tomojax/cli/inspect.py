"""``tomojax inspect``: describe and check a dataset before using it."""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, cast

from tomojax.cli._options import add_force, check_paths
from tomojax.cli._previews import preview_paths, write_previews
from tomojax.io import validate_dataset
from tomojax.io.api import format_inspection_report, inspect_dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

_HDF5 = (".nxs", ".h5", ".hdf5")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tomojax inspect",
        description=(
            "Describe a TomoJAX dataset (projections, angles, geometry, flats and darks, "
            "corrections, any reconstructed volume and memory estimates) and check it is "
            "complete enough to reconstruct."
        ),
        epilog=(
            "Exit status: 0 when the dataset is valid, 1 when it has issues, 2 for a usage "
            "error.\n\nExamples:\n"
            "  tomojax inspect scan.nxs\n"
            "  tomojax inspect recon.nxs --preview previews\n"
            "  tomojax inspect scan.nxs --json > report.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ = parser.add_argument("data", metavar="INPUT", help="Dataset (.nxs, .h5)")
    _ = parser.add_argument(
        "--json", action="store_true", help="Print the report as JSON instead of text"
    )
    _ = parser.add_argument(
        "--preview",
        metavar="DIR",
        help="Write PNGs of the central projection (and the volume's central slices) to DIR",
    )
    add_force(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``tomojax inspect``."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    path = cast("str", args.data)
    preview = cast("str | None", args.preview)
    check_paths(parser, inputs=[path])
    if not path.lower().endswith(_HDF5):
        parser.error(f"inspect reads .nxs/.h5 datasets; convert {path} with `tomojax import`")
    if preview is not None:
        check_paths(
            parser,
            outputs=preview_paths(path, preview).values(),
            force=cast("bool", args.force),
        )

    report = inspect_dataset(path)
    issues = validate_dataset(path)["issues"]
    previews = (
        {} if preview is None else {k: str(p) for k, p in write_previews(path, preview).items()}
    )

    if cast("bool", args.json):
        payload: dict[str, object] = {
            **report,
            "valid": not issues,
            "issues": issues,
            "previews": previews,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_inspection_report(report))
        if previews:
            print("Previews: " + ", ".join(previews.values()))
        print("Valid: yes" if not issues else f"Issues ({len(issues)}):")
        for issue in issues:
            print(f"  - {issue}")
    return 0 if not issues else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
