"""CLI: preprocess raw NXtomo sample/flat/dark frames."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TYPE_CHECKING, cast

from tomojax.core import setup_logging
from tomojax.io import (
    PreprocessConfig,
    preprocess_nxtomo,
    preprocess_tiff_stack,
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Flat/dark-correct raw NXtomo/image_key or explicit TIFF stacks into "
            "sample-only absorption projections by default"
        )
    )
    _ = parser.add_argument(
        "input",
        help=(
            "Input raw .nxs/.h5/.hdf5 file, or TIFF projection file/directory "
            "with --format tiff-stack"
        ),
    )
    _ = parser.add_argument("output", help="Output corrected .nxs/.h5/.hdf5 file")
    _ = parser.add_argument(
        "--format",
        dest="input_format",
        choices=["nxtomo", "tiff-stack"],
        default="nxtomo",
        help="Input layout: NXtomo/HDF5 with image_key, or explicit TIFF projections/flats/darks",
    )
    _ = parser.add_argument(
        "--domain",
        dest="output_domain",
        choices=["absorption", "transmission"],
        default="absorption",
        help="Output projection domain; absorption is reconstruction-ready and the default",
    )
    _ = parser.add_argument(
        "--log",
        action="store_true",
        help="Alias for --domain absorption; absorption is already the default",
    )
    _ = parser.add_argument(
        "--transmission",
        action="store_true",
        help="Write normalized transmission instead of reconstruction-ready absorption",
    )
    _ = parser.add_argument(
        "--flats",
        default=None,
        help="TIFF-stack mode: flat-field TIFF file or directory",
    )
    _ = parser.add_argument(
        "--darks",
        default=None,
        help="TIFF-stack mode: dark-field TIFF file or directory",
    )
    _ = parser.add_argument(
        "--angles",
        default=None,
        help="TIFF-stack mode: angle sidecar (.npy or CSV/text degrees)",
    )
    _ = parser.add_argument(
        "--quicklook",
        default=None,
        help="Write a percentile-scaled central corrected-projection PNG",
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
        help=(
            "Keep only these sample-view indices/ranges after image_key filtering "
            "(e.g. '0:90,120:180:2')"
        ),
    )
    _ = parser.add_argument(
        "--reject-views",
        default=None,
        help=(
            "Reject these sample-view indices/ranges after image_key filtering (e.g. '12,57:61')"
        ),
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
        help="Detector ROI crop in projection axis order y0:y1,x0:x1",
    )
    return parser


def _optional_str(value: object) -> str | None:
    return cast("str | None", value)


def _optional_float(value: object) -> float | None:
    return cast("float | None", value)


def _parse_command(argv: Sequence[str] | None) -> PreprocessCommand:
    """Parse CLI arguments into a typed preprocessing command plan."""
    args = _build_parser().parse_args(argv)
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
    output_domain = cast("str", args.output_domain)
    if cast("bool", args.log):
        output_domain = "absorption"
    if cast("bool", args.transmission):
        output_domain = "transmission"
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
    )
    return PreprocessCommand(
        input_path=Path(cast("str", args.input)),
        output_path=Path(cast("str", args.output)),
        input_format=cast("str", args.input_format),
        flats_path=Path(cast("str", args.flats)) if cast("str | None", args.flats) else None,
        darks_path=Path(cast("str", args.darks)) if cast("str | None", args.darks) else None,
        angles_sidecar_path=Path(cast("str", args.angles))
        if cast("str | None", args.angles)
        else None,
        quicklook_path=Path(cast("str", args.quicklook))
        if cast("str | None", args.quicklook)
        else None,
        config=config,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the raw NXtomo preprocessing command."""
    command = _parse_command(argv)
    input_path = command.input_path
    output_path = command.output_path

    if not input_path.exists():
        print(f"ERROR: file not found: {input_path}", file=sys.stderr)
        return 2
    if command.input_format == "nxtomo" and not input_path.is_file():
        print(f"ERROR: not a file: {input_path}", file=sys.stderr)
        return 2
    if output_path.exists() and not output_path.is_file():
        print(f"ERROR: output path exists and is not a file: {output_path}", file=sys.stderr)
        return 2

    setup_logging()
    try:
        if command.input_format == "tiff-stack":
            tiff_error = _tiff_command_error(command)
            if tiff_error is not None:
                print(tiff_error, file=sys.stderr)
                return 2
            flats_path = command.flats_path
            darks_path = command.darks_path
            angles_sidecar_path = command.angles_sidecar_path
            assert flats_path is not None
            assert darks_path is not None
            assert angles_sidecar_path is not None
            result = preprocess_tiff_stack(
                input_path,
                flats_path,
                darks_path,
                angles_sidecar_path,
                output_path,
                command.config,
            )
        else:
            result = preprocess_nxtomo(input_path, output_path, command.config)
        if command.quicklook_path is not None:
            _ = save_projection_quicklook(output_path, command.quicklook_path)
    except Exception as exc:
        print(f"ERROR: could not preprocess {input_path}: {exc}", file=sys.stderr)
        return 1

    print(
        "Wrote corrected "
        f"{result.output_domain} projections to {output_path} "
        f"(samples={result.sample_count}, flats={result.flat_count}, darks={result.dark_count}, "
        f"shape={list(result.output_shape)})"
    )
    return 0


def _tiff_command_error(command: PreprocessCommand) -> str | None:
    if command.flats_path is None or command.darks_path is None:
        return "ERROR: --format tiff-stack requires --flats and --darks"
    if command.angles_sidecar_path is None:
        return "ERROR: --format tiff-stack requires --angles"
    return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
