"""``tomojax correct``: the optional corrections, on raw frames or a saved scan.

A thin layer over :func:`tomojax.load_frames` and :meth:`tomojax.Frames.corrected`
for raw frames, and :func:`tomojax.load` and :meth:`tomojax.Scan.corrected` for a
dataset of line integrals, with :meth:`~tomojax.Scan.selected`,
:meth:`~tomojax.Scan.cropped` and binning; the scan written records each step.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from tomojax.cli._inputs import PUBLIC as INPUTS, add_inputs, declared, input_paths, load_keywords
from tomojax.cli._options import add_output, check_paths, fail, hide_expert

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomojax.corrections import Step

_PUBLIC = (
    *INPUTS,
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tomojax correct",
        description=(
            "Apply the optional corrections and cut the scan down. On raw frames (HDF5 with "
            "image_key, Nikon, SkyScan, Zeiss, ESRF, a TIFF stack) every step runs, with the "
            "standard dark, flat and -log; on a dataset of line integrals, the steps that act "
            "on them (--stripes, --rings, --reject-views). The scan written records each step."
        ),
        epilog=(
            "Examples:\n"
            "  tomojax correct raw.nxs -o scan.nxs --stripes 9 --zingers --reject-views\n"
            "  tomojax correct skyscan/ --flats WHITE --darks DARK -o scan.nxs\n"
            "  tomojax correct scan.nxs -o rings.nxs --rings\n"
            "  tomojax correct raw.nxs -o scan.nxs --crop 100:900,50:1950 --views 0:1200:4 "
            "--binning 2"
        ),
    )
    _ = parser.add_argument(
        "data", metavar="INPUT", help="Raw acquisition, TIFF stack, or a dataset of line integrals"
    )
    add_output(parser, "Corrected scan to write (.nxs)")
    add_inputs(parser)
    _ = parser.add_argument(
        "--binning", type=int, default=1, help="Average BINNING x BINNING detector pixels into one"
    )
    _ = parser.add_argument(
        "--views",
        metavar="RANGES",
        help="Keep only these views, as indices and ranges (for example 0:90,120:180:2)",
    )
    _ = parser.add_argument(
        "--crop", metavar="Y0:Y1,X0:X1", help="Keep this detector block (rows y, then columns x)"
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
    hide_expert(parser, _PUBLIC)
    return parser


def _numbers(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {text!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("expected at least one number")
    return values


def view_ranges(text: str, views: int) -> np.ndarray:
    """The increasing view indices of ``text``: indices and ``start:stop[:step]`` ranges."""
    chosen: set[int] = set()
    for token in text.replace(",", " ").split():
        parts = token.split(":")
        try:
            numbers = [int(p) for p in parts]
        except ValueError as exc:
            raise ValueError(f"views: {token!r} is not an index or start:stop[:step]") from exc
        if len(numbers) == 1:
            values = range(numbers[0], numbers[0] + 1)
        elif len(numbers) in (2, 3) and (len(numbers) == 2 or numbers[2] > 0):
            values = range(*numbers)
        else:
            raise ValueError(f"views: {token!r} is not an index or start:stop[:step]")
        if not values or min(values) < 0 or max(values) >= views:
            raise ValueError(f"views: {token!r} is outside the {views} views (0 to {views - 1})")
        chosen.update(values)
    return np.asarray(sorted(chosen), np.int64)


def detector_block(text: str) -> tuple[slice, slice]:
    """The rows and columns of ``Y0:Y1,X0:X1``."""
    try:
        rows, cols = (tuple(int(n) for n in part.split(":")) for part in text.split(","))
        (y0, y1), (x0, x1) = rows, cols
    except ValueError as exc:
        raise ValueError(f"--crop {text!r}: expected Y0:Y1,X0:X1") from exc
    return slice(y0, y1), slice(x0, x1)


def _steps(args: argparse.Namespace) -> list[Step]:
    """The correction steps the options ask for, in the order they run."""
    from tomojax.corrections import Paganin, RejectViews, Stripes, Zingers

    zingers = cast("float | None", args.zingers)
    paganin = cast("tuple[float, ...] | None", args.paganin)
    outliers = cast("float | None", args.reject_views)
    stripes = cast("int | None", args.stripes)
    steps: list[Step] = []
    if zingers is not None:
        steps.append(Zingers(threshold=zingers))
    if paganin is not None:
        if len(paganin) != 4:
            raise ValueError("--paganin takes DELTA_BETA,DISTANCE,ENERGY_KEV,PIXEL")
        steps.append(Paganin(*paganin))
    if outliers is not None:
        steps.append(RejectViews(z=outliers))
    if stripes is not None:
        steps.append(Stripes(width=stripes))
    return steps


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``tomojax correct``."""
    import tomojax as tj
    from tomojax.core import setup_logging
    from tomojax.io.api import save_projection_quicklook

    parser = _build_parser()
    args = parser.parse_args(argv)
    source, output = cast("str", args.data), cast("str", args.out)
    preview = cast("str | None", args.preview)
    check_paths(
        parser,
        inputs=[source, *input_paths(args)],
        outputs=[output, preview],
        force=cast("bool", args.force),
    )
    try:
        steps = _steps(args)
    except ValueError as exc:
        parser.error(str(exc))
    setup_logging()
    try:
        scan = _corrected(tj, source, args, steps)
    except (KeyError, ValueError, OSError) as exc:
        return fail("correct", str(exc).strip("'\""))
    tj.save(output, scan)
    if preview is not None:
        _ = save_projection_quicklook(Path(output), Path(preview))
    print(f"wrote {output}: {scan!r}")
    return 0


def _corrected(tj: Any, source: str, args: argparse.Namespace, steps: list[Step]) -> Any:
    """The scan: raw frames corrected with every step, or a dataset with its own."""
    select, crop = cast("str | None", args.views), cast("str | None", args.crop)
    binning = cast("int", args.binning)
    keywords = load_keywords(args)
    if not _raw(source, args):
        named = [name for name in _TRANSMISSION if getattr(args, name) is not None]
        if named:
            raise ValueError(
                f"--{named[0]} acts on transmission, before the log; {source} holds line "
                "integrals. Run tomojax correct on the raw acquisition (with the same --flats/"
                "--darks/--angles/--geometry flags as tomojax load) to apply it."
            )
        scan = tj.load(source, **keywords)
        scan = replace(scan, geometry=declared(scan.geometry, args))
        if select is not None:
            scan = scan.selected(view_ranges(select, len(scan.angles)))
        if crop is not None:
            scan = scan.cropped(*detector_block(crop))
        if args.rings:
            from tomojax.corrections import Rings

            steps.insert(0, Rings(scan.geometry))
        scan = scan.corrected(*steps) if steps else scan
        return scan.binned(binning)
    keywords.pop("apply_poses")
    frames = tj.load_frames(source, **keywords)
    frames = replace(frames, geometry=declared(frames.geometry, args))
    if select is not None:
        frames = frames.selected(view_ranges(select, frames.views))
    if crop is not None:
        frames = frames.cropped(*detector_block(crop))
    if args.rings:
        from tomojax.corrections import Rings

        # Fit the selected native views before rejection changes their geometry.
        steps.insert(0, Rings(frames.geometry))
    return frames.corrected(*steps, epsilon=cast("float", args.epsilon), binning=binning)


def _raw(source: str, args: argparse.Namespace) -> bool:
    """Whether ``source`` holds detector counts (with flats and darks), not line integrals."""
    from tomojax.io.api import locate_frames, read_acquisition

    path = Path(source)
    if args.flats is not None or args.darks is not None or read_acquisition(path) is not None:
        return True
    if path.is_dir() or path.suffix.lower() in {".tif", ".tiff"}:
        return True
    if path.suffix.lower() == ".npz":
        return False
    located = locate_frames(source)
    keys = located.image_key
    labelled = keys is not None and bool(np.any(keys != 0))
    return labelled or located.flats is not None or located.darks is not None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
