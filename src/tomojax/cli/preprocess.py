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
from tomojax.geometry import Detector

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomojax.corrections import Step

_PUBLIC = (
    "--flats",
    "--darks",
    "--angles",
    "--select-views",
    "--crop",
    "--reject-views",
    "--zingers",
    "--stripes",
    "--rings",
    "--beam-hardening",
    "--preview",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tomojax preprocess",
        description=(
            "Correct raw detector frames into line integrals -log((I - D) / (F - D)) ready "
            "to reconstruct, with each view's flat interpolated between the flat sets "
            "around it. INPUT may be HDF5 with image_key calibration, or a Nikon, "
            "SkyScan, Zeiss or ESRF acquisition with its own metadata. Generic TIFF "
            "stacks need --angles, --flats and a detector in --config; "
            "SkyScan exports need explicit white and dark levels. "
            "The scan written records each correction."
        ),
        epilog=(
            "Examples:\n"
            "  tomojax preprocess raw.nxs -o scan.nxs\n"
            "  tomojax preprocess skyscan/ --flats WHITE --darks DARK -o scan.nxs\n"
            "  tomojax preprocess raw.nxs -o scan.nxs --stripes 9 --zingers --reject-views\n"
            "  tomojax preprocess frames/ --flats flats/ --darks darks/ --angles angles.csv "
            "-o scan.nxs\n"
            "  tomojax preprocess raw.nxs -o scan.nxs --crop 100:900,50:1950 "
            "--select-views 0:1200:4"
        ),
    )
    _ = parser.add_argument(
        "data", metavar="INPUT", help="Acquisition file/folder (including SkyScan), or TIFF stack"
    )
    add_output(parser, "Corrected scan to write (.nxs)")
    add_config(parser, settings={"binning": 1, "detector": None})
    _ = parser.add_argument(
        "--flats", metavar="PATH|LEVEL", help="Flat frames (HDF5, TIFF or XRM), or one level"
    )
    _ = parser.add_argument(
        "--darks", metavar="PATH|LEVEL", help="Dark frames (HDF5, TIFF or XRM), or one level"
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
        "--crop", metavar="Y0:Y1,X0:X1", help="Keep this detector block (rows y, then columns x)"
    )
    _ = parser.add_argument(
        "--reject-views",
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
        "--beam-hardening",
        choices=("recorded",),
        help="Linearise single-material beam hardening with the curve the vendor recorded",
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


def _detector(settings: dict[str, object]) -> Detector | None:
    """The same explicit detector override accepted by load_frames, from TOML."""
    value = settings.get("detector")
    if value is None:
        return None
    if not isinstance(value, dict) or not {"nu", "nv", "du", "dv", "det_center"} <= value.keys():
        raise ValueError("detector in --config needs nu, nv, du, dv and det_center")
    return Detector.from_dict(value)


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
    """Run ``tomojax preprocess``."""
    import tomojax as tj
    from tomojax.core import setup_logging
    from tomojax.io.api import load_angles, save_projection_quicklook

    parser = _build_parser()
    args, config_meta = parse_args_with_config(parser, argv)
    settings = cast("dict[str, object]", config_meta["settings"])
    source, output = cast("str", args.data), cast("str", args.out)
    flats = _level_or_path(cast("str | None", args.flats))
    darks = _level_or_path(cast("str | None", args.darks))
    angles, preview = cast("str | None", args.angles), cast("str | None", args.preview)
    select = cast("str | None", args.select_views)
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
            detector=_detector(settings),
            data_path=cast("str | None", args.data_path),
            image_key_path=cast("str | None", args.image_key_path),
            angles_path=cast("str | None", args.angles_path),
        )
        if select is not None:
            frames = frames.selected(view_ranges(select, frames.views))
        if crop is not None:
            frames = frames.cropped(*detector_block(crop))
        if args.rings:
            from tomojax.corrections import Rings

            # Fit the selected native views before rejection changes their geometry.
            steps.insert(0, Rings(frames.geometry))
        scan = frames.corrected(
            *steps,
            epsilon=cast("float", args.epsilon),
            binning=cast("int", settings.get("binning", 1)),
        )
        if args.beam_hardening is not None:
            from tomojax.corrections import BeamHardening

            scan = scan.corrected(BeamHardening.recorded(scan))
    except (KeyError, ValueError, OSError) as exc:
        return fail("preprocess", str(exc).strip("'\""))
    tj.save(output, scan)
    if preview is not None:
        _ = save_projection_quicklook(Path(output), Path(preview))
    print(f"wrote {output}: {scan!r}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
