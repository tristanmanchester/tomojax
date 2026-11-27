"""``tomojax inspect``: describe and check a dataset before using it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

from tomojax.cli import _previews as pv
from tomojax.cli._options import add_force, check_destinations, check_paths, options
from tomojax.cli._progress import ProgressParser, add_progress, run_command
from tomojax.io.api import format_inspection_report

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomojax.io.api import Inspection
    from tomojax.wire.progress import Result


class _Args(argparse.Namespace):
    data: str
    preview_dir: str | None
    slice: list[str] | None
    centers: str | None
    force: bool
    json: bool
    progress: str


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
            "  tomojax inspect scan.nxs --preview-dir previews --slice z=200 --slice x=512\n"
            "  tomojax inspect scan.nxs --preview-dir sweep --centers -20:20:2\n"
            "  tomojax inspect scan.nxs --json > report.json"
        ),
    )
    _ = parser.add_argument(
        "data",
        metavar="INPUT",
        help="NXtomo/HDF5 dataset, vendor acquisition file/folder, or checkpoint directory",
    )
    _ = parser.add_argument(
        "--json", action="store_true", help="Print the report as JSON instead of text"
    )
    _ = parser.add_argument(
        "--preview-dir",
        metavar="DIR",
        help="Write PNGs of the central projection and slices (reconstructed for a scan) to DIR",
    )
    _ = parser.add_argument(
        "--slice",
        metavar="AXIS=INDEX",
        action="append",
        help="A plane to preview, such as z=200 (repeatable; default the central z plane)",
    )
    _ = parser.add_argument(
        "--centers",
        metavar="SPEC",
        help="Sweep the detector center, in pixels from the middle: START:STOP:STEP or a,b,c",
    )
    add_progress(parser)
    add_force(parser)
    for action in options(parser):
        if action.dest == "force":
            action.help = "Replace existing preview files in --preview-dir"
    return parser


def _joined(argv: Sequence[str]) -> list[str]:
    """``--centers VALUE`` as ``--centers=VALUE``, so a negative VALUE is not read as an option."""
    args = list(argv)
    joined: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--centers" and i + 1 < len(args) and pv.is_centers_value(args[i + 1]):
            joined.append(f"--centers={args[i + 1]}")
            i += 2
            continue
        joined.append(args[i])
        i += 1
    return joined


def _display(
    report: Inspection, previews: pv.Previews | None, preview_error: str | None, *, as_json: bool
) -> None:
    issues = report["issues"]
    if as_json:
        payload: dict[str, object] = {**report}
        if previews is not None:
            payload["previews"] = previews.files
            payload["slice_requests"] = previews.slice_requests
            if previews.sweep:
                payload["sweep"] = previews.sweep
                payload["sweep_apply"] = previews.apply
        if preview_error is not None:
            payload["preview_error"] = preview_error
            print(f"tomojax inspect: error: preview failed: {preview_error}", file=sys.stderr)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(format_inspection_report(report))
    if report["checkpoint"] is not None:
        return
    if previews is not None:
        _print_previews(previews)
    if preview_error is not None:
        print(f"tomojax inspect: error: preview failed: {preview_error}", file=sys.stderr)
    print("Valid: yes" if not issues else f"Issues ({len(issues)}):")
    for issue in issues:
        print(f"  - {issue}")


def _print_previews(previews: pv.Previews) -> None:
    print("Previews")
    for note in previews.notes:
        print(f"  note: {note}")
    for name, path in previews.files.items():
        if not name.startswith("center_"):
            print(f"  {path}")
    if not previews.sweep:
        return
    apply = previews.apply or {}
    unit = apply.get("unit", "")
    print("Center sweep")
    print(f"  recorded center: {apply.get('recorded_px', 0.0):+.2f} px")
    for entry in previews.sweep:
        center, length = entry["center_px"], entry["center_length"]
        print(
            f"  {entry['ordinal']:>3}  {center:+.2f} px  U = {length:.6g} {unit}  {entry['path']}"
        )
    print(f"  sheet: {previews.files['center_sweep.png']}")
    print(f"  apply one: {apply.get('command')} (U from the list above)")


def _make_previews(
    parser: ProgressParser,
    path: Path,
    directory: Path,
    *,
    slices: list[tuple[pv.Axis, int]] | None,
    centers: tuple[float, ...] | None,
    force: bool,
    listing: bool,
) -> pv.Previews:
    import tomojax as tj
    from tomojax._loading.loader import is_raw_input
    from tomojax.io.api import saved_without_frames

    out = pv.Previews()
    saved = pv.has_saved_volume(path)  # its planes are read, not reconstructed
    frameless = saved and saved_without_frames(path)
    # Only a scan's slices or a sweep need to know (and loading the frames needs JAX).
    raw = (centers is not None or not saved) and not frameless and is_raw_input(path)
    if centers is not None and (frameless or raw):
        kind = "a reconstruction without its scan" if frameless else "raw frames"
        parser.fail(
            f"--centers sweeps a scan of line integrals; {path} holds {kind}",
            hint="tomojax correct turns raw frames into a scan" if raw else None,
        )
    scan = None if frameless or raw or (saved and centers is None) else tj.load(path)
    projection = pv.projection_image(path)
    if saved:
        wanted = pv.saved_requests(path, slices)
    else:
        wanted = [] if scan is None else slices or [("z", scan.grid.nz // 2)]
    has = {"projection": projection is not None, "listing": listing}
    paths = pv.destinations(directory, wanted, centers or (), **has)
    try:
        check_destinations({path.resolve()}, paths, force=force)
    except FileExistsError as exc:
        parser.fail(str(exc), hint="pass --force to replace the previews")
    directory.mkdir(parents=True, exist_ok=True)
    if saved:
        pv.write_saved_slices(path, directory, wanted, out)
    elif raw:
        out.notes.append("raw frames: tomojax correct makes a scan whose slices can be previewed")
    else:
        assert scan is not None
        pv.write_scan_slices(scan, directory, wanted, out)
    if scan is not None and centers is not None:
        plane = slices[0] if slices else None
        pv.write_sweep(scan, path, directory, centers, plane, out)
    if projection is not None:  # last, so a bad --slice leaves nothing that needs --force
        from tomojax.cli import _images

        image = _images.labeled(
            _images.to_uint8(projection, _images.limits(projection)), "central projection"
        )
        out.files["projection.png"] = str(_images.save_png(image, directory / "projection.png"))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``tomojax inspect``: :func:`tomojax.inspect`, with optional previews."""
    import tomojax as tj

    parser = _build_parser()
    given = _joined(sys.argv[1:] if argv is None else argv)
    parser.set_invocation(given)
    args = parser.parse_args(given, namespace=_Args())
    if args.json and args.progress == "json":
        parser.error("--json cannot print an inspection document with --progress json")
    preview_dir = args.preview_dir
    try:
        slices = None if args.slice is None else [pv.parse_slice(s) for s in args.slice]
        centers = None if args.centers is None else pv.parse_centers(args.centers)
    except ValueError as exc:
        parser.fail(str(exc), hint="for example --slice z=200 or --centers -10:10:1")
    if (slices or centers) and preview_dir is None:
        parser.fail(
            "--slice and --centers write images", hint="give a folder for them with --preview-dir"
        )

    def run() -> Result | int:
        path = Path(args.data)
        check_paths(parser, inputs=[path])
        report = tj.inspect(path)
        issues = report["issues"]
        previews: pv.Previews | None = None
        preview_error: str | None = None
        if preview_dir is not None:
            try:
                previews = _make_previews(
                    parser,
                    path,
                    Path(preview_dir),
                    slices=slices,
                    centers=centers,
                    force=args.force,
                    listing=args.progress == "json",
                )
            except (OSError, ValueError, KeyError) as exc:
                preview_error = str(exc).strip("'\"")

        if args.progress == "json":
            return _json_result(path, report, previews, preview_error, preview_dir)
        _display(report, previews, preview_error, as_json=args.json)
        return int(bool(issues or preview_error))

    return run_command("inspect", args.progress, run)


def _json_result(
    path: Path,
    report: Inspection,
    previews: pv.Previews | None,
    preview_error: str | None,
    preview_dir: str | None,
) -> Result:
    from tomojax.cli._results import report_result

    if preview_error or report["issues"]:
        raise ValueError(preview_error or "; ".join(report["issues"]))
    summary: dict[str, object] = dict(report)
    outputs: tuple[str, ...] = ()
    if previews is not None and preview_dir is not None:
        listing = Path(preview_dir) / pv.LISTING
        document = {
            "previews": previews.files,
            "slice_requests": previews.slice_requests,
            "sweep": previews.sweep,
            "sweep_apply": previews.apply,
        }
        listing.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        summary["preview_count"] = len(previews.files)
        summary["preview_listing"] = str(listing)
        outputs = (*previews.files.values(), str(listing))
    return report_result(path, outputs, summary)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
