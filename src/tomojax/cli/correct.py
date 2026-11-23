"""``tomojax correct``: import raw scans and apply corrections, selection, crop and binning.

A thin layer over :func:`tomojax.load_frames` and :meth:`tomojax.Frames.corrected`
for raw frames, and :func:`tomojax.load` and :meth:`tomojax.Scan.corrected` for a
dataset of line integrals, with :meth:`~tomojax.Scan.selected`,
:meth:`~tomojax.Scan.cropped` and binning; the scan written records each step.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from tomojax import progress
from tomojax.cli._inputs import (
    PUBLIC as INPUTS,
    add_inputs,
    input_paths,
    load_keywords,
    validate_inputs,
)
from tomojax.cli._options import add_config, add_output, check_paths, hide_expert
from tomojax.cli._progress import add_progress, run_command
from tomojax.cli.config import parse_args_with_config

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from tomojax._loading_geometry import DeclaredGeometry
    from tomojax.corrections import Step

_PUBLIC = (
    *INPUTS,
    "--progress",
    "--views",
    "--crop",
    "--reject-views",
    "--zingers",
    "--stripes",
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
        "paganin_distance": (None, "Paganin effective propagation distance in metres"),
        "paganin_energy_kev": (None, "Paganin X-ray energy in keV"),
        "paganin_pixel": (None, "Paganin effective pixel size at the sample in metres"),
        "stripes_varying": (False, "Use slow-varying stripe removal with --stripes"),
        "zingers_size": (3, "Median-filter neighbourhood size for --zingers (odd int >= 3)"),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the ``tomojax correct`` parser."""
    parser = argparse.ArgumentParser(
        prog="tomojax correct",
        description=(
            "Import, convert or correct a scan, with optional view selection, cropping and "
            "binning. On raw frames (HDF5 with image_key, Nikon, SkyScan, Zeiss, ESRF, a TIFF "
            "stack), runs dark subtraction, flat division and -log plus any requested steps; on "
            "saved line integrals, converts formats or runs line-integral steps (--stripes, "
            "--rings, --reject-views). The scan written records each step."
        ),
        epilog=(
            "Examples:\n"
            "  tomojax correct scan/scan.xtekct -o scan.nxs --darks 0\n"
            "  tomojax correct raw.nxs -o scan.nxs --stripes 9 --zingers --reject-views\n"
            "  tomojax correct scan.nxs -o rings.nxs --rings\n"
            "  tomojax correct raw.nxs -o scan.nxs --crop 100:900,50:1950 --views ::4 --binning 2"
        ),
    )
    _ = parser.add_argument(
        "data", metavar="INPUT", help="Raw acquisition, TIFF stack, or a dataset of line integrals"
    )
    add_output(parser, "Corrected scan to write (.nxs, .h5 or .npz)")
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
        help="Replace specks brighter than their neighbours' median by THRESHOLD "
        "(a fraction of the open beam; 0.1)",
    )
    _ = parser.add_argument(
        "--stripes",
        type=int,
        metavar="WIDTH",
        help="Remove rings: each pixel's offset against a median over WIDTH columns (e.g. 9)",
    )
    _ = parser.add_argument(
        "--rings",
        action="store_true",
        help="Correct pixel gain and response errors using the acquisition geometry",
    )
    _ = parser.add_argument(
        "--preview", metavar="PNG", help="Write the central corrected projection as a PNG"
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
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError(f"--crop {text!r}: expected Y0:Y1,X0:X1")
    slices: list[slice] = []
    for piece in parts:
        bounds = piece.split(":")
        if len(bounds) != 2:
            raise ValueError(f"--crop {text!r}: expected Y0:Y1,X0:X1")
        start = _parse_bound(bounds[0], "--crop")
        stop = _parse_bound(bounds[1], "--crop")
        if start is not None and stop is not None and stop <= start:
            raise ValueError(f"--crop {text!r}: stop must exceed start")
        slices.append(slice(start, stop))
    return slices[0], slices[1]


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
        values = tuple(float(cast("Any", paganin_fields[k])) for k in _PAGANIN_KEYS)
        return cast("tuple[float, float, float, float]", values)
    if cli_paganin is None:
        return None
    if len(cli_paganin) != 4:
        raise ValueError("--paganin takes DELTA_BETA,DISTANCE,ENERGY_KEV,PIXEL")
    return cast("tuple[float, float, float, float]", cli_paganin)


def _steps(args: argparse.Namespace, settings: Mapping[str, object]) -> list[Step]:
    """The correction steps the options and TOML settings ask for, in run order."""
    from tomojax.corrections import Paganin, RejectViews, Stripes, Zingers

    zingers = cast("float | None", args.zingers)
    paganin = _resolve_paganin(cast("tuple[float, ...] | None", args.paganin), settings)
    outliers = cast("float | None", args.reject_views)
    stripes = cast("int | None", args.stripes)
    zingers_size = settings.get("zingers_size", 3)
    if isinstance(zingers_size, bool) or not isinstance(zingers_size, int):
        raise ValueError("zingers_size must be an odd integer >= 3")
    stripes_varying = settings.get("stripes_varying", False)
    if not isinstance(stripes_varying, bool):
        raise ValueError("stripes_varying must be a boolean")
    steps: list[Step] = []
    if zingers is not None:
        steps.append(Zingers(threshold=zingers, size=zingers_size))
    if paganin is not None:
        steps.append(Paganin(*paganin))
    if outliers is not None:
        steps.append(RejectViews(z=outliers))
    if stripes is not None:
        steps.append(Stripes(width=stripes, varying=stripes_varying))
    return steps


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``tomojax correct``."""
    import tomojax as tj
    from tomojax.core import setup_logging
    from tomojax.io.api import save_projection_quicklook

    parser = build_parser()
    args, metadata = parse_args_with_config(parser, argv)
    settings = cast("dict[str, object]", metadata["settings"])
    declared = validate_inputs(parser, args)
    source, output = cast("str", args.data), cast("str", args.out)
    preview = cast("str | None", args.preview)
    try:
        if cast("int", args.binning) < 1:
            raise ValueError("--binning must be at least 1")
        if args.views is not None:
            _ = _parse_view_tokens(cast("str", args.views))
        crop_slices = None if args.crop is None else detector_block(cast("str", args.crop))
        steps = _steps(args, settings)
    except ValueError as exc:
        parser.error(str(exc))

    def run() -> None:
        for path in [source, *input_paths(args)]:
            if path is not None and not Path(path).exists():
                raise FileNotFoundError(f"input not found: {path}")
        check_paths(
            parser,
            inputs=[source, *input_paths(args)],
            outputs=[output, preview],
            force=cast("bool", args.force),
        )
        setup_logging()
        progress.check_cancelled()
        scan = _corrected(source, args, declared, steps, crop_slices)
        progress.check_cancelled()
        tj.save(output, scan)
        if preview is not None:
            progress.check_cancelled()
            _ = save_projection_quicklook(Path(output), Path(preview))
        if args.progress != "json":
            print(f"wrote {output}: {scan!r}")

    return run_command("correct", args.progress, run)


def _corrected(
    source: str,
    args: argparse.Namespace,
    declared: DeclaredGeometry | None,
    steps: list[Step],
    crop: tuple[slice, slice] | None,
) -> Any:
    """The scan: raw frames corrected with every step, or a dataset with its own."""
    from tomojax._loading import (
        is_raw_input,
        load_raw_frames_with_declaration,
        load_scan_with_declaration,
    )

    select = cast("str | None", args.views)
    binning = cast("int", args.binning)
    keywords = load_keywords(args)
    progress.check_cancelled()
    raw = is_raw_input(
        source,
        flats=keywords["flats"],
        darks=keywords["darks"],
        data_path=keywords["data_path"],
        image_key_path=keywords["image_key_path"],
        angles_path=keywords["angles_path"],
    )
    progress.check_cancelled()
    if not raw:
        has_paganin = any(s.domain == "transmission" for s in steps)
        if has_paganin:
            step = next(type(s).__name__.lower() for s in steps if s.domain == "transmission")
            raise ValueError(
                f"{step} acts on transmission, before the log; {source} holds line integrals. "
                "Run tomojax correct on the raw acquisition to apply it."
            )
        scan = load_scan_with_declaration(source, declared=declared, **keywords)
        progress.check_cancelled()
        if select is not None:
            scan = scan.selected(view_ranges(select, len(scan.angles)))
        if crop is not None:
            scan = scan.cropped(*crop)
        if args.rings:
            from tomojax.corrections import Rings

            steps.insert(0, Rings())
        scan = scan.corrected(*steps) if steps else scan
        progress.check_cancelled()
        return scan.binned(binning)
    frames = load_raw_frames_with_declaration(source, declared=declared, **keywords)
    progress.check_cancelled()
    if select is not None:
        frames = frames.selected(view_ranges(select, frames.views))
    if crop is not None:
        frames = frames.cropped(*crop)
    if args.rings:
        from tomojax.corrections import Rings

        steps.insert(0, Rings())
    return frames.corrected(*steps, epsilon=cast("float", args.epsilon), binning=binning)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
