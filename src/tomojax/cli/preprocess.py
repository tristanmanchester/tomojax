"""``tomojax preprocess``: correct raw detector frames into a scan of line integrals.

A thin layer over :func:`tomojax.load_frames`, :meth:`tomojax.Frames.selected`,
:meth:`~tomojax.Frames.cropped` and :meth:`~tomojax.Frames.corrected` with
steps from :mod:`tomojax.corrections`; the scan written records them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from tomojax.cli._options import add_config, add_output, check_paths, fail, hide_expert
from tomojax.cli.config import parse_args_with_config

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomojax.corrections import Step

_PUBLIC = (
    "--flats",
    "--darks",
    "--angles",
    "--select-views",
    "--reject-views",
    "--crop",
    "--reject-outliers",
    "--zingers",
    "--remove-stripes",
    "--beam-hardening",
    "--preview",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tomojax preprocess",
        description=(
            "Correct raw detector frames into line integrals -log((I - D) / (F - D)) ready "
            "to reconstruct, with each view's flat interpolated between the flat sets "
            "around it. INPUT is an HDF5 file whose image_key marks flats (1) and darks "
            "(2), a Nikon .xtekct scan, or a TIFF file or folder (with --angles and "
            "--flats). The scan written records each correction."
        ),
        epilog=(
            "Examples:\n"
            "  tomojax preprocess raw.nxs -o scan.nxs\n"
            "  tomojax preprocess raw.nxs -o scan.nxs --remove-stripes 9 --zingers\n"
            "  tomojax preprocess frames/ --flats flats/ --darks darks/ --angles angles.csv "
            "-o scan.nxs\n"
            "  tomojax preprocess raw.nxs -o scan.nxs --crop 100:900,50:1950 "
            "--reject-views 12,57:61"
        ),
    )
    _ = parser.add_argument(
        "data", metavar="INPUT", help="Raw HDF5 file, Nikon .xtekct, or a TIFF file or folder"
    )
    add_output(parser, "Corrected scan to write (.nxs)")
    add_config(parser)
    _ = parser.add_argument(
        "--flats", metavar="TIFF|LEVEL", help="Flat frames (TIFF file or folder), or one level"
    )
    _ = parser.add_argument(
        "--darks", metavar="TIFF|LEVEL", help="Dark frames (TIFF file or folder), or one level"
    )
    _ = parser.add_argument(
        "--angles", metavar="FILE", help="Angles in degrees (.npy, or one per line)"
    )
    _ = parser.add_argument(
        "--select-views",
        metavar="RANGES",
        help="Keep only these views, as indices and ranges (for example 0:90,120:180:2)",
    )
    _ = parser.add_argument(
        "--reject-views", metavar="RANGES", help="Drop these views (for example 12,57:61)"
    )
    _ = parser.add_argument(
        "--crop", metavar="Y0:Y1,X0:X1", help="Keep this detector block (rows y, then columns x)"
    )
    _ = parser.add_argument(
        "--reject-outliers",
        type=float,
        nargs="?",
        const=6.0,
        metavar="Z",
        help="Drop views whose median is more than Z robust deviations from the views' (6)",
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
        "--remove-stripes",
        type=int,
        metavar="WIDTH",
        help="Remove rings: each pixel's offset against a median over WIDTH columns (e.g. 9)",
    )
    _ = parser.add_argument(
        "--beam-hardening",
        type=_numbers,
        metavar="C1,C2,...",
        help="Linearise beam hardening: p -> C1 p + C2 p^2 + ... (for example 1,0.05)",
    )
    _ = parser.add_argument(
        "--preview", metavar="PNG", help="Write the central corrected projection as a PNG"
    )
    _ = parser.add_argument(
        "--paganin",
        type=_numbers,
        metavar="DELTA_BETA,DISTANCE,ENERGY_KEV,PIXEL",
        help="Single-material phase retrieval: delta/beta, sample-detector distance (m), "
        "energy (keV), pixel at the sample (m)",
    )
    _ = parser.add_argument(
        "--epsilon", type=float, default=1e-6, help="Floor of the flat field and transmission"
    )
    _ = parser.add_argument("--data-path", help="HDF5 path of the stack of frames")
    _ = parser.add_argument("--image-key-path", help="HDF5 path of the image_key")
    _ = parser.add_argument("--angles-path", help="HDF5 path of the rotation angles")
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


def _level_or_path(value: str | None) -> float | str | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return value


def _steps(args: argparse.Namespace) -> list[Step]:
    """The correction steps the options ask for, in the order they run."""
    from tomojax.corrections import BeamHardening, Paganin, RejectViews, Stripes, Zingers

    zingers = cast("float | None", args.zingers)
    paganin = cast("tuple[float, ...] | None", args.paganin)
    outliers = cast("float | None", args.reject_outliers)
    stripes = cast("int | None", args.remove_stripes)
    hardening = cast("tuple[float, ...] | None", args.beam_hardening)
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
    if hardening is not None:
        steps.append(BeamHardening(hardening))
    return steps


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``tomojax preprocess``."""
    import tomojax as tj
    from tomojax.core import setup_logging
    from tomojax.io.api import load_angles, save_projection_quicklook

    parser = _build_parser()
    args, _ = parse_args_with_config(parser, argv)
    source, output = cast("str", args.data), cast("str", args.out)
    flats = _level_or_path(cast("str | None", args.flats))
    darks = _level_or_path(cast("str | None", args.darks))
    angles, preview = cast("str | None", args.angles), cast("str | None", args.preview)
    select, reject = cast("str | None", args.select_views), cast("str | None", args.reject_views)
    crop = cast("str | None", args.crop)
    check_paths(
        parser,
        inputs=[source, *(p for p in (flats, darks) if isinstance(p, str)), angles],
        outputs=[output, preview],
        force=cast("bool", args.force),
    )
    try:
        steps = _steps(args)
    except ValueError as exc:
        parser.error(str(exc))
    setup_logging()
    try:
        frames = tj.load_frames(
            source,
            flats=flats,
            darks=darks,
            angles=None if angles is None else load_angles(Path(angles)),
            data_path=cast("str | None", args.data_path),
            image_key_path=cast("str | None", args.image_key_path),
            angles_path=cast("str | None", args.angles_path),
        )
        keep = np.ones(frames.views, bool)
        if select is not None:
            keep[:] = False
            keep[view_ranges(select, frames.views)] = True
        if reject is not None:
            keep[view_ranges(reject, frames.views)] = False
        if not keep.all():
            frames = frames.selected(keep)
        if crop is not None:
            frames = frames.cropped(*detector_block(crop))
        scan = frames.corrected(*steps, epsilon=cast("float", args.epsilon))
    except (KeyError, ValueError, OSError) as exc:
        return fail("preprocess", str(exc).strip("'\""))
    tj.save(output, scan)
    if preview is not None:
        _ = save_projection_quicklook(Path(output), Path(preview))
    print(f"wrote {output}: {scan!r}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
