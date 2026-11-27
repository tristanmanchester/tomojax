"""``tomojax correct``: import raw scans and apply corrections, selection, crop and binning.

A thin layer over :func:`tomojax.load_frames` and :meth:`tomojax.Frames.corrected`
for raw frames, and :func:`tomojax.load` and :meth:`tomojax.Scan.corrected` for a
dataset of line integrals, with :meth:`~tomojax.Scan.selected`,
:meth:`~tomojax.Scan.cropped` and binning; the scan written records each step.
Bracketing multi-frame flat sets default to ``--flat-weights views`` (pass
``--flat-weights position`` or ``flat_weights = "position"`` in ``--config`` to
blend by position alone); ``--flats`` and ``--darks`` accept another scan's file
and take its flat or dark frames.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tomojax import progress
from tomojax._parameters import FLAT_WEIGHTS
from tomojax.cli._inputs import (
    PUBLIC as INPUTS,
    InputArgs,
    add_inputs,
    check_scan_calibration,
    input_paths,
    load_keywords,
    validate_inputs,
)
from tomojax.cli._options import add_config, add_output, check_paths, hide_expert, options
from tomojax.cli._progress import ProgressParser, add_progress, run_command
from tomojax.cli.config import parse_args_with_config

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from tomojax import Frames, Scan
    from tomojax._loading.geometry import DeclaredGeometry
    from tomojax._parameters import FlatWeighting
    from tomojax.corrections import Step
    from tomojax.wire.progress import Result


class _Args(InputArgs):
    data: str
    out: str | None
    force: bool
    dry_run: bool
    progress: str
    views: str | None
    crop: str | None
    binning: int
    preview_rows: str | None
    preview: str | None
    paganin: tuple[float, ...] | None
    zingers: float | None
    reject_views: float | None
    rings: bool
    epsilon: float
    flat_weights: FlatWeighting


_PUBLIC = (
    *(option for option in INPUTS if option != "--apply-poses"),
    "--dry-run",
    "--progress",
    "--views",
    "--crop",
    "--reject-views",
    "--zingers",
    "--rings",
    "--binning",
    "--preview",
)
# Steps on transmission, before the log: they need raw frames.
_TRANSMISSION = ("zingers", "paganin")
_PAGANIN_KEYS = ("paganin_delta_beta", "paganin_distance", "paganin_energy_kev", "paganin_pixel")


def _settings() -> dict[str, object]:
    """Expert TOML keys for ``tomojax correct --config``."""
    return {
        "paganin_delta_beta": (None, "Paganin delta/beta ratio (requires all four paganin_* keys)"),
        "paganin_distance": (None, "Paganin effective propagation distance in meters"),
        "paganin_energy_kev": (None, "Paganin X-ray energy in keV"),
        "paganin_pixel": (None, "Paganin effective pixel size at the sample in meters"),
        "zingers_size": (3, "Median-filter neighborhood size for --zingers (odd int >= 3)"),
    }


def build_parser() -> ProgressParser:
    """Build the ``tomojax correct`` parser."""
    parser = ProgressParser(
        prog="tomojax correct",
        description=(
            "Import, convert or correct a scan, with optional view selection, cropping and "
            "binning. On raw frames (HDF5 with image_key, Nikon, SkyScan, Zeiss, ESRF, a TIFF "
            "stack), runs dark subtraction, flat division and -log plus any requested steps; on "
            "saved line integrals, converts formats or runs line-integral steps ("
            "--rings, --reject-views). The scan written records each step."
        ),
        epilog=(
            "Examples:\n"
            "  tomojax correct scan/scan.xtekct -o scan.nxs --darks 0\n"
            "  tomojax correct raw.nxs -o scan.nxs --rings --zingers --reject-views\n"
            "  tomojax correct scan.nxs -o rings.nxs --rings\n"
            "  tomojax correct raw.nxs -o scan.nxs --crop 100:900,50:1950 --views ::4 --binning 2"
        ),
    )
    _ = parser.add_argument(
        "data", metavar="INPUT", help="Raw acquisition, TIFF stack, or a dataset of line integrals"
    )
    add_output(parser, "Corrected scan to write (.nxs, .h5 or .npz); optional with --dry-run")
    for action in options(parser):
        if action.dest == "out":
            action.required = False  # a dry run writes no scan
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written and stop (with --preview, write only the preview)",
    )
    add_config(parser, settings=_settings)
    add_inputs(parser)
    _ = parser.add_argument(
        "--binning", type=int, default=1, help="Average BINNING x BINNING detector pixels into one"
    )
    _ = parser.add_argument(
        "--views",
        metavar="RANGES",
        help="Keep views by 0-based indices and Python slices (e.g. ::8 or 0:90,120:180:2; "
        "deduplicated in acquisition order)",
    )
    _ = parser.add_argument(
        "--crop",
        metavar="Y0:Y1,X0:X1",
        help="Keep this detector block (rows y then columns x; omitted endpoints allowed)",
    )
    _ = parser.add_argument(
        "--reject-views",
        type=float,
        nargs="?",
        const=6.0,
        metavar="Z",
        help="Drop views inside an abrupt step of the mean, Z robust deviations (6)",
    )
    _ = parser.add_argument(
        "--zingers",
        type=float,
        nargs="?",
        const=0.1,
        metavar="THRESHOLD",
        help="Replace specks brighter than their neighbors' median by THRESHOLD "
        "(a fraction of the open beam; 0.1)",
    )
    _ = parser.add_argument(
        "--rings",
        action="store_true",
        help="Correct pixel gain and response errors using the acquisition geometry",
    )
    _ = parser.add_argument(
        "--preview",
        metavar="PNG",
        help="Write sinograms of a row band before and after the steps, and their difference",
    )
    _ = parser.add_argument(
        "--preview-rows",
        metavar="A:B",
        help="The band --preview shows, in output rows (default: 8 at the detector middle)",
    )
    _ = parser.add_argument(
        "--paganin",
        type=_numbers,
        metavar="DELTA_BETA,DISTANCE,ENERGY_KEV,PIXEL",
        help="Single-material phase retrieval: delta/beta, effective propagation distance (m), "
        "energy (keV), pixel at the sample (m). For cone beams divide both the "
        "sample-detector distance and the detector pixel by magnification",
    )
    _ = parser.add_argument(
        "--epsilon", type=float, default=1e-6, help="Floor of the flat field and transmission"
    )
    _ = parser.add_argument(
        "--flat-weights",
        choices=("views", "position"),
        default=FLAT_WEIGHTS,
        help="Blend the flat sets around a view by what the views show of each, or by the "
        f"view's position between them (default {FLAT_WEIGHTS})",
    )
    add_progress(parser)
    hide_expert(parser, _PUBLIC)
    return parser


def _numbers(text: str) -> tuple[float, ...]:
    parts = text.split(",")
    if not parts or any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {text!r}")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {text!r}") from exc


def _parse_bound(part: str, label: str) -> int | None:
    stripped = part.strip()
    if not stripped:
        return None
    try:
        value = int(stripped)
    except ValueError as exc:
        raise ValueError(f"{label}: {part!r} is not an integer") from exc
    if value < 0:
        raise ValueError(f"{label}: negative index {value} is not supported")
    return value


def _parse_view_tokens(text: str) -> list[tuple[int | None, int | None, int | None] | int]:
    """Parse view index/slice tokens without knowing total views yet (for upfront CLI checks)."""
    tokens = text.replace(",", " ").split()
    if not tokens:
        raise ValueError("views: expected at least one index or slice")
    parsed: list[tuple[int | None, int | None, int | None] | int] = []
    for token in tokens:
        parts = token.split(":")
        if len(parts) == 1:
            index = _parse_bound(parts[0], "views")
            if index is None:
                raise ValueError(f"views: {token!r} is not an index or slice")
            parsed.append(index)
        elif len(parts) in (2, 3):
            start = _parse_bound(parts[0], "views")
            stop = _parse_bound(parts[1], "views")
            step = _parse_bound(parts[2], "views") if len(parts) == 3 else None
            if step is not None and step < 1:
                raise ValueError(f"views: step must be positive in {token!r}")
            parsed.append((start, stop, step))
        else:
            raise ValueError(f"views: {token!r} is not an index or start:stop[:step] slice")
    return parsed


def view_ranges(text: str, views: int) -> np.ndarray:
    """Increasing 0-based view indices from indices and Python slices (deduplicated union)."""
    chosen: set[int] = set()
    for item in _parse_view_tokens(text):
        if isinstance(item, int):
            if item >= views:
                raise ValueError(f"views: {item} is outside the {views} views (0 to {views - 1})")
            chosen.add(item)
            continue
        start, stop, step = item
        if stop is not None and stop > views:
            raise ValueError(f"views: stop {stop} exceeds the {views} views")
        values = range(*slice(start, stop, step).indices(views))
        if not values:
            raise ValueError(f"views: empty range in {text!r} for {views} views")
        chosen.update(values)
    return np.asarray(sorted(chosen), np.int64)


def detector_block(text: str) -> tuple[slice, slice]:
    """The rows and columns of ``[Y0]:[Y1],[X0]:[X1]``."""
    rows, columns = crop_intervals(text, "Y0:Y1,X0:X1")
    return rows, columns


def crop_intervals(text: str, pattern: str) -> tuple[slice, ...]:
    """Parse one half-open interval per storage axis."""
    parts = text.split(",")
    if len(parts) != len(pattern.split(",")):
        raise ValueError(f"--crop {text!r}: expected {pattern}")
    slices: list[slice] = []
    for piece in parts:
        bounds = piece.split(":")
        if len(bounds) != 2:
            raise ValueError(f"--crop {text!r}: expected {pattern}")
        start = _parse_bound(bounds[0], "--crop")
        stop = _parse_bound(bounds[1], "--crop")
        if start is not None and stop is not None and stop <= start:
            raise ValueError(f"--crop {text!r}: stop must exceed start")
        slices.append(slice(start, stop))
    return tuple(slices)


def _resolve_paganin(
    cli_paganin: tuple[float, ...] | None,
    settings: Mapping[str, object],
) -> tuple[float, float, float, float] | None:
    """Resolve Paganin parameters from either ``--paganin`` or TOML ``paganin_*`` keys."""
    paganin_fields = {k: settings[k] for k in _PAGANIN_KEYS if k in settings}
    if paganin_fields and cli_paganin is not None:
        raise ValueError("pass either --paganin or paganin_* config keys, not both")
    if paganin_fields:
        if len(paganin_fields) != 4:
            missing = [k for k in _PAGANIN_KEYS if k not in paganin_fields]
            raise ValueError(f"paganin config requires all four keys; missing {', '.join(missing)}")
        values: list[float] = []
        for key in _PAGANIN_KEYS:
            value = paganin_fields[key]
            if not isinstance(value, int | float | str):
                raise ValueError(f"{key} must be a number")
            values.append(float(value))
        cli_paganin = tuple(values)
    if cli_paganin is None:
        return None
    if len(cli_paganin) != 4:
        raise ValueError("--paganin takes DELTA_BETA,DISTANCE,ENERGY_KEV,PIXEL")
    delta_beta, distance, energy, pixel = cli_paganin
    return delta_beta, distance, energy, pixel


def _steps(args: _Args, settings: Mapping[str, object]) -> list[Step]:
    """The correction steps the options and TOML settings ask for, in run order."""
    from tomojax.corrections import Paganin, RejectViews, Zingers

    zingers = args.zingers
    paganin = _resolve_paganin(args.paganin, settings)
    outliers = args.reject_views
    zingers_size = settings.get("zingers_size", 3)
    if isinstance(zingers_size, bool) or not isinstance(zingers_size, int):
        raise ValueError("zingers_size must be an odd integer >= 3")
    steps: list[Step] = []
    if zingers is not None:
        steps.append(Zingers(threshold=zingers, size=zingers_size))
    if paganin is not None:
        steps.append(Paganin(*paganin))
    if outliers is not None:
        steps.append(RejectViews(z=outliers))
    return steps


def _band(text: str) -> tuple[int, int]:
    """``A:B`` as a half-open row interval."""
    lower, sep, upper = text.partition(":")
    if not sep or not lower.isdigit() or not upper.isdigit() or int(upper) <= int(lower):
        raise ValueError(f"--preview-rows takes A:B with B > A >= 0, not {text!r}")
    return int(lower), int(upper)


def _checked(
    parser: ProgressParser, args: _Args, settings: Mapping[str, object]
) -> tuple[tuple[slice, slice] | None, list[Step], tuple[int, int] | None]:
    """The crop, the steps and the preview band, or a usage error before anything loads."""
    try:
        if args.binning < 1:
            raise ValueError("--binning must be at least 1")
        if args.views is not None:
            _ = _parse_view_tokens(args.views)
        crop = None if args.crop is None else detector_block(args.crop)
        steps = _steps(args, settings)
        rows = None if args.preview_rows is None else _band(args.preview_rows)
    except ValueError as exc:
        parser.error(str(exc))
    if rows is not None and args.preview is None:
        parser.fail("--preview-rows chooses the band --preview draws", hint="add --preview PNG")
    return crop, steps, rows


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``tomojax correct``."""
    parser = build_parser()
    args, metadata = parse_args_with_config(parser, argv, namespace=_Args())
    settings = metadata["settings"]
    declared = validate_inputs(parser, args)
    dry = args.dry_run
    output = args.out
    if output is None and not dry:
        parser.error("the following arguments are required: -o/--output")
    preview = args.preview
    crop_slices, steps, rows = _checked(parser, args, settings)
    source = args.data

    def run() -> Result | None:
        for path in [source, *input_paths(args)]:
            if path is not None and not Path(path).exists():
                raise FileNotFoundError(f"input not found: {path}")
        check_paths(
            parser,
            inputs=[source, *input_paths(args), metadata["config_path"]],
            outputs=[None if dry else output, preview],
            force=args.force,
        )
        progress.check_canceled()
        prepared = _prepared(source, args, declared, steps, crop_slices)
        check_paths(parser, outputs=[None if dry else output, preview], force=args.force)
        band = _preview_band(parser, prepared, rows, args.binning)
        if dry:
            return _dry_run(args, prepared, band, preview)
        assert output is not None
        scan = _applied(prepared, args)
        progress.check_canceled()
        import tomojax as tj

        tj.save(output, scan)
        if preview is not None:
            progress.check_canceled()
            write_band_preview(Path(preview), prepared, band, args)
        if args.progress == "json":
            from tomojax.cli._results import saved_result

            outputs = (output,) if preview is None else (output, preview)
            return saved_result(output, scan, outputs=outputs)
        print(f"wrote {output}: {scan!r}")
        return None

    return run_command("correct", args.progress, run)


@dataclass(frozen=True)
class _Prepared:
    """The input after view selection and crop, and the steps to run on it."""

    item: Frames | Scan
    steps: tuple[Step, ...]
    selection: np.ndarray  # source view index of each selected view
    total: int  # views in the source

    @property
    def raw(self) -> bool:
        from tomojax import Frames

        return isinstance(self.item, Frames)


def _prepared(
    source: str,
    args: _Args,
    declared: DeclaredGeometry | None,
    steps: list[Step],
    crop: tuple[slice, slice] | None,
) -> _Prepared:
    """Load raw frames or a scan, select views and crop; Rings runs first when asked."""
    from tomojax import Frames
    from tomojax._loading.loader import load_native

    select = args.views
    with progress.task("metadata", Path(source).name, None, ""):
        item = load_native(source, declared=declared, **load_keywords(args))
    if not isinstance(item, Frames):
        check_scan_calibration(args.epsilon, args.flat_weights)
    if not isinstance(item, Frames) and any(s.domain == "transmission" for s in steps):
        step = next(type(s).__name__.lower() for s in steps if s.domain == "transmission")
        raise ValueError(
            f"{step} acts on transmission, before the log; {source} holds line integrals. "
            "Run tomojax correct on the raw acquisition to apply it."
        )
    progress.check_canceled()
    total = int(item.views) if isinstance(item, Frames) else len(item.angles)
    selection = np.arange(total)
    if select is not None:
        selection = view_ranges(select, total)
        item = item.selected(selection)
    if crop is not None:
        item = item.cropped(*crop)
    ordered = list(steps)
    if args.rings:
        from tomojax.corrections import Rings

        ordered.insert(0, Rings())
    return _Prepared(item, tuple(ordered), selection, total)


def _applied(prepared: _Prepared, args: _Args) -> Scan:
    """Every step on the prepared input, then binning (raw frames bin as they correct)."""
    from tomojax import Frames

    binning = args.binning
    item = prepared.item
    if isinstance(item, Frames):
        return item.corrected(
            *prepared.steps, epsilon=args.epsilon, flat_weights=args.flat_weights, binning=binning
        )
    scan = item.corrected(*prepared.steps) if prepared.steps else item
    progress.check_canceled()
    return scan.binned(binning)


def _detector_shape(prepared: _Prepared) -> tuple[int, int]:
    from tomojax import Frames

    item = prepared.item
    shape = item.counts.shape if isinstance(item, Frames) else np.shape(item.projections)
    return int(shape[1]), int(shape[2])


def _preview_band(
    parser: ProgressParser, prepared: _Prepared, rows: tuple[int, int] | None, binning: int
) -> tuple[int, int]:
    """The output rows ``--preview`` draws: as asked, or up to 8 at the detector middle."""
    height = _detector_shape(prepared)[0] // binning
    if rows is None:
        count = min(8, height)
        lower = (height - count) // 2
        return lower, lower + count
    if rows[1] > height:
        message = f"--preview-rows {rows[0]}:{rows[1]} is past the {height} output rows"
        parser.fail(message, hint=f"choose rows below {height}")
    return rows


def _step_names(prepared: _Prepared) -> list[str]:
    """The steps in the order they run: by domain, then as given."""
    order = {"counts": 0, "transmission": 1, "line integrals": 2}
    named = ["flat and dark correction"] if prepared.raw else []
    logged = not prepared.raw
    for step in sorted(prepared.steps, key=lambda s: order.get(s.domain, 2)):
        if not logged and step.domain == "line integrals":
            named.append("-log")
            logged = True
        record = step.record()
        settings = ", ".join(f"{k}={v}" for k, v in record.settings.items() if v is not None)
        named.append(f"{record.name}({settings})")
    if not logged:
        named.append("-log")
    return named


def _dry_run(
    args: _Args, prepared: _Prepared, band: tuple[int, int], preview: str | None
) -> Result | None:
    """What a run would write, and the preview when asked; no scan is written."""
    from tomojax.cli._plan_view import size as readable
    from tomojax.corrections import RejectViews
    from tomojax.corrections.api import preview_window

    binning = args.binning
    height, width = _detector_shape(prepared)
    shape = (len(prepared.selection), height // binning, width // binning)
    size = 4 * shape[0] * shape[1] * shape[2]
    bound = any(isinstance(step, RejectViews) for step in prepared.steps)
    window, _ = preview_window(prepared.steps, (height, width), *band, binning)
    full_rows = window == slice(0, height) and (band[1] - band[0]) * binning < height
    if preview is not None:
        write_band_preview(Path(preview), prepared, band, args)
    names = _step_names(prepared)
    if args.progress == "json":
        from tomojax.cli._results import report_result

        summary: dict[str, object] = {
            "steps": names,
            "shape": list(shape),
            "bytes": size,
            "upper_bound": bound,
            "full_rows": full_rows,
        }
        return report_result(Path(args.data), () if preview is None else (preview,), summary)
    at_most = "at most " if bound else ""
    print(f"steps    {', '.join(names) or 'none'}")
    print(f"output   {at_most}{shape[0]} views of {shape[1]} x {shape[2]} pixels")
    print(f"size     {at_most}{readable(size)} of float32 line integrals")
    reads = "every detector row" if full_rows else "only its band"
    print(f"preview  output rows {band[0]}:{band[1]}, which read {reads}")
    if preview is not None:
        print(f"wrote    {preview}")
    return None


def write_band_preview(path: Path, prepared: _Prepared, band: tuple[int, int], args: _Args) -> None:
    """Sinograms of the band's middle row: without the steps, with them, and the difference.

    Rows are the selected views in acquisition order; a strip on the left marks the
    views ``--reject-views`` drops. The legend says which source views were selected.
    """
    from PIL import Image

    from tomojax import interactive
    from tomojax.cli import _images
    from tomojax.wire.arrays import Axis, Uniform

    before, after = (
        interactive.correction_preview(
            prepared.item,
            band,
            *steps,
            binning=args.binning,
            epsilon=args.epsilon,
            flat_weights=args.flat_weights,
        )  # fmt: skip
        for steps in ((), prepared.steps)
    )
    middle = (band[1] - band[0]) // 2
    views = len(prepared.selection)
    rows: dict[str, np.ndarray] = {}
    for name, result in (("before", before), ("after", after)):
        full = np.full((views, result.after.shape[2]), np.nan, np.float32)
        full[result.views] = result.after[:, middle, :]  # indices into the selected views
        rows[name] = full
    difference = rows["after"] - rows["before"]
    shared = _images.limits(rows["before"], rows["after"])
    finite = np.abs(difference[np.isfinite(difference)])
    spread = float(np.percentile(finite, 99.5)) if finite.size else 0.0
    rejected = np.ones(views, bool)
    rejected[after.views] = False
    axes = (Axis("view", "acquisition", "index", Uniform(0, 1)),) * 2
    shape = _images.display_shape(rows["before"].shape, axes, _images.PLANE_SIDE)
    marks = np.where(rejected, 255, 0).astype(np.float32)[:, None].repeat(6, axis=1)
    baseline = "flat, dark and -log only" if prepared.raw else "input line integrals"
    steps = ", ".join(type(step).__name__.lower() for step in prepared.steps) or "no steps"
    pictures = [
        Image.fromarray(
            _images.to_uint8(_images.resample(marks, (shape[0], 6), nearest=True), (0.0, 255.0))
        )
    ]
    for image, limits, label in (
        (rows["before"], shared, f"before: {baseline}"),
        (rows["after"], shared, f"after: {steps}"),
        (difference, (-(spread or 1.0), spread or 1.0), "after - before"),
    ):
        shown = _images.resample(image, shape, nearest=True)
        pictures.append(_images.labeled(_images.to_uint8(shown, limits), label))
    height = max(p.height for p in pictures)
    canvas = Image.new("L", (sum(p.width for p in pictures) + 8 * len(pictures), height), 0)
    x = 0
    for picture in pictures:
        canvas.paste(picture, (x, height - picture.height))
        x += picture.width + 8
    chosen = (
        f"all {prepared.total} views"
        if views == prepared.total
        else f"{views} of {prepared.total} views (--views)"
    )
    legend = (
        f"output rows {band[0]}:{band[1]}, row {band[0] + middle} shown; {chosen}, downwards; "
        f"white left strip: {int(rejected.sum())} dropped by --reject-views"
    )
    _ = _images.save_png(_images.labeled(np.asarray(canvas), legend), path)


__all__ = ["build_parser", "detector_block", "main", "view_ranges"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
