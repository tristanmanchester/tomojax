"""``tomojax preprocess``: flat- and dark-correct raw frames."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from tomojax.cli._options import add_config, add_output, check_paths, hide_expert
from tomojax.cli.config import parse_args_with_config
from tomojax.core import setup_logging
from tomojax.io import (
    PreprocessConfig,
    preprocess_nxtomo,
    preprocess_tiff_stack,
)
from tomojax.io.api import (
    save_projection_quicklook,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class PreprocessCommand:
    """Typed command plan for raw NXtomo preprocessing."""

    input_path: Path
    output_path: Path
    input_format: str
    flats_path: Path | None
    darks_path: Path | None
    angles_sidecar_path: Path | None
    quicklook_path: Path | None
    config: PreprocessConfig


_PUBLIC = (
    "--flats",
    "--darks",
    "--angles",
    "--transmission",
    "--select-views",
    "--reject-views",
    "--crop",
    "--beam-hardening",
    "--remove-stripes",
    "--preview",
)
_TIFF_SUFFIXES = (".tif", ".tiff")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tomojax preprocess",
        description=(
            "Flat- and dark-correct raw frames into absorption projections ready to "
            "reconstruct. INPUT is a raw NXtomo file (frames labelled by image_key) or a "
            "TIFF file or directory, which needs --flats, --darks and --angles."
        ),
        epilog=(
            "Examples:\n"
            "  tomojax preprocess raw.nxs -o scan.nxs\n"
            "  tomojax preprocess raw.nxs -o scan.nxs --remove-stripes 9 --beam-hardening 1,0.05\n"
            "  tomojax preprocess frames/ --flats flats/ --darks darks/ --angles angles.csv "
            "-o scan.nxs"
        ),
    )
    _ = parser.add_argument(
        "data", metavar="INPUT", help="Raw .nxs/.h5 file, or a TIFF file or directory"
    )
    add_output(parser, "Corrected dataset to write (.nxs)")
    add_config(parser)
    _ = parser.add_argument("--flats", metavar="TIFF", help="TIFF input: flat-field frames")
    _ = parser.add_argument("--darks", metavar="TIFF", help="TIFF input: dark-field frames")
    _ = parser.add_argument(
        "--angles", metavar="FILE", help="TIFF input: angles in degrees (.npy, or one per line)"
    )
    _ = parser.add_argument(
        "--transmission",
        action="store_true",
        help="Write transmission (I / flat) instead of absorption",
    )
    _ = parser.add_argument(
        "--preview", metavar="PNG", help="Write the central corrected projection as a PNG"
    )
    _ = parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Positive floor for flat-dark denominator and log safeguard",
    )
    _ = parser.add_argument(
        "--clip-min",
        type=float,
        default=None,
        help="Optional positive floor applied to transmission before writing/log",
    )
    _ = parser.add_argument(
        "--dtype",
        dest="output_dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Output projection dtype",
    )
    _ = parser.add_argument(
        "--data-path",
        default=None,
        help="Override HDF5 path to raw frame stack [n_frames, nv, nu]",
    )
    _ = parser.add_argument(
        "--angles-path",
        default=None,
        help="Override HDF5 path to rotation angles [n_frames]",
    )
    _ = parser.add_argument(
        "--image-key-path",
        default=None,
        help="Override HDF5 path to image_key [n_frames] with 0=sample, 1=flat, 2=dark",
    )
    _ = parser.add_argument(
        "--assume-dark-field",
        type=float,
        default=None,
        help="Explicit constant dark field to use when no dark frames are present",
    )
    _ = parser.add_argument(
        "--assume-flat-field",
        type=float,
        default=None,
        help="Explicit constant flat field to use when no flat frames are present",
    )
    _ = parser.add_argument(
        "--select-views",
        default=None,
        metavar="RANGES",
        help="Keep only these views, as indices and ranges (for example 0:90,120:180:2)",
    )
    _ = parser.add_argument(
        "--reject-views",
        default=None,
        metavar="RANGES",
        help="Drop these views (for example 12,57:61)",
    )
    _ = parser.add_argument(
        "--select-views-file",
        default=None,
        help=(
            "File containing sample-view indices/ranges to keep; commas, "
            "whitespace, and # comments allowed"
        ),
    )
    _ = parser.add_argument(
        "--reject-views-file",
        default=None,
        help=(
            "File containing sample-view indices/ranges to reject; commas, "
            "whitespace, and # comments allowed"
        ),
    )
    _ = parser.add_argument(
        "--auto-reject",
        choices=["off", "nonfinite", "outliers", "both"],
        default="off",
        help=(
            "Optionally reject corrected sample views with non-finite values "
            "and/or robust intensity outliers"
        ),
    )
    _ = parser.add_argument(
        "--outlier-z-threshold",
        type=float,
        default=6.0,
        help="Robust z-score threshold for --auto-reject outliers/both",
    )
    _ = parser.add_argument(
        "--crop",
        default=None,
        metavar="Y0:Y1,X0:X1",
        help="Keep this detector region (rows y, then columns x)",
    )
    _ = parser.add_argument(
        "--beam-hardening",
        type=_coefficients,
        default=None,
        metavar="C1,C2,...",
        help=(
            "Absorption output: linearise beam hardening, mapping each value p to "
            "C1*p + C2*p^2 + ... (for example 1,0.05)"
        ),
    )
    _ = parser.add_argument(
        "--remove-stripes",
        type=int,
        default=None,
        metavar="WIDTH",
        help=(
            "Absorption output: remove detector-fixed errors that reconstruct as rings "
            "(sorting-based stripe removal, median over WIDTH columns; wider than the "
            "defects, for example 9)"
        ),
    )
    hide_expert(parser, _PUBLIC)
    return parser


def _coefficients(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {text!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("expected at least one coefficient")
    return values


def _optional_str(value: object) -> str | None:
    return cast("str | None", value)


def _optional_float(value: object) -> float | None:
    return cast("float | None", value)


def _parse_command(argv: Sequence[str] | None) -> PreprocessCommand:
    """Parse CLI arguments into a typed preprocessing command plan."""
    parser = _build_parser()
    args, _ = parse_args_with_config(parser, argv)
    source = cast("str", args.data)
    check_paths(
        parser,
        inputs=[source, cast("str | None", args.flats), cast("str | None", args.darks)],
        outputs=[cast("str", args.out), cast("str | None", args.preview)],
        force=cast("bool", args.force),
    )
    tiff = Path(source).is_dir() or source.lower().endswith(_TIFF_SUFFIXES)
    sidecars = cast("tuple[str | None, ...]", (args.flats, args.darks, args.angles))
    if tiff and None in sidecars:
        parser.error("TIFF input needs --flats, --darks and --angles")
    clip_min = cast("float | None", args.clip_min)
    data_path = cast("str | None", args.data_path)
    angles_path = cast("str | None", args.angles_path)
    image_key_path = cast("str | None", args.image_key_path)
    assume_dark_field = cast("float | None", args.assume_dark_field)
    assume_flat_field = cast("float | None", args.assume_flat_field)
    select_views = cast("str | None", args.select_views)
    reject_views = cast("str | None", args.reject_views)
    select_views_file = cast("str | None", args.select_views_file)
    reject_views_file = cast("str | None", args.reject_views_file)
    crop = cast("str | None", args.crop)
    output_domain = "transmission" if cast("bool", args.transmission) else "absorption"
    config = PreprocessConfig(
        output_domain=output_domain,
        epsilon=cast("float", args.epsilon),
        clip_min=_optional_float(clip_min),
        output_dtype=cast("str", args.output_dtype),
        data_path=_optional_str(data_path),
        angles_path=_optional_str(angles_path),
        image_key_path=_optional_str(image_key_path),
        assume_dark_field=_optional_float(assume_dark_field),
        assume_flat_field=_optional_float(assume_flat_field),
        select_views=_optional_str(select_views),
        reject_views=_optional_str(reject_views),
        select_views_file=_optional_str(select_views_file),
        reject_views_file=_optional_str(reject_views_file),
        auto_reject=cast("str", args.auto_reject),
        outlier_z_threshold=cast("float", args.outlier_z_threshold),
        crop=_optional_str(crop),
        beam_hardening=cast("tuple[float, ...] | None", args.beam_hardening),
        stripe_width=cast("int | None", args.remove_stripes),
    )
    return PreprocessCommand(
        input_path=Path(source),
        output_path=Path(cast("str", args.out)),
        input_format="tiff-stack" if tiff else "nxtomo",
        flats_path=Path(cast("str", args.flats)) if cast("str | None", args.flats) else None,
        darks_path=Path(cast("str", args.darks)) if cast("str | None", args.darks) else None,
        angles_sidecar_path=Path(cast("str", args.angles))
        if cast("str | None", args.angles)
        else None,
        quicklook_path=Path(cast("str", args.preview))
        if cast("str | None", args.preview)
        else None,
        config=config,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the raw NXtomo preprocessing command."""
    command = _parse_command(argv)
    input_path = command.input_path
    output_path = command.output_path

    setup_logging()
    if command.input_format == "tiff-stack":
        if (
            command.flats_path is None
            or command.darks_path is None
            or command.angles_sidecar_path is None
        ):
            raise RuntimeError("validated TIFF-stack command is missing required sidecars")
        result = preprocess_tiff_stack(
            input_path,
            flats_path=command.flats_path,
            darks_path=command.darks_path,
            angles_path=command.angles_sidecar_path,
            output_path=output_path,
            config=command.config,
        )
    else:
        result = preprocess_nxtomo(input_path, output_path, command.config)
    if command.quicklook_path is not None:
        _ = save_projection_quicklook(output_path, command.quicklook_path)

    print(
        f"wrote {output_path}: {result.output_domain} projections "
        f"(samples={result.sample_count}, flats={result.flat_count}, darks={result.dark_count}, "
        f"shape={list(result.output_shape)})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
