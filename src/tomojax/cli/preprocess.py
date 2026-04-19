"""CLI: preprocess raw NXtomo sample/flat/dark frames."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from ..data.preprocess import PreprocessConfig, preprocess_nxtomo
from ..utils.logging import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Flat/dark-correct a raw mixed-frame NXtomo file into sample-only "
            "transmission or absorption projections"
        )
    )
    parser.add_argument("input", help="Input raw .nxs/.h5/.hdf5 file")
    parser.add_argument("output", help="Output corrected .nxs/.h5/.hdf5 file")
    parser.add_argument(
        "--log",
        action="store_true",
        help="Write absorption projections (-log transmission) instead of transmission",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Positive floor for flat-dark denominator and log safeguard",
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=None,
        help="Optional positive floor applied to transmission before writing/log",
    )
    parser.add_argument(
        "--dtype",
        dest="output_dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Output projection dtype",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Override HDF5 path to raw frame stack [n_frames, nv, nu]",
    )
    parser.add_argument(
        "--angles-path",
        default=None,
        help="Override HDF5 path to rotation angles [n_frames]",
    )
    parser.add_argument(
        "--image-key-path",
        default=None,
        help="Override HDF5 path to image_key [n_frames] with 0=sample, 1=flat, 2=dark",
    )
    parser.add_argument(
        "--assume-dark-field",
        type=float,
        default=None,
        help="Explicit constant dark field to use when no dark frames are present",
    )
    parser.add_argument(
        "--assume-flat-field",
        type=float,
        default=None,
        help="Explicit constant flat field to use when no flat frames are present",
    )
    parser.add_argument(
        "--select-views",
        default=None,
        help=(
            "Keep only these sample-view indices/ranges after image_key filtering "
            "(e.g. '0:90,120:180:2')"
        ),
    )
    parser.add_argument(
        "--reject-views",
        default=None,
        help=(
            "Reject these sample-view indices/ranges after image_key filtering (e.g. '12,57:61')"
        ),
    )
    parser.add_argument(
        "--select-views-file",
        default=None,
        help="File containing sample-view indices/ranges to keep; commas, whitespace, and # comments allowed",
    )
    parser.add_argument(
        "--reject-views-file",
        default=None,
        help="File containing sample-view indices/ranges to reject; commas, whitespace, and # comments allowed",
    )
    parser.add_argument(
        "--auto-reject",
        choices=["off", "nonfinite", "outliers", "both"],
        default="off",
        help="Optionally reject corrected sample views with non-finite values and/or robust intensity outliers",
    )
    parser.add_argument(
        "--outlier-z-threshold",
        type=float,
        default=6.0,
        help="Robust z-score threshold for --auto-reject outliers/both",
    )
    parser.add_argument(
        "--crop",
        default=None,
        help="Detector ROI crop in projection axis order y0:y1,x0:x1",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: file not found: {input_path}", file=sys.stderr)
        return 2
    if not input_path.is_file():
        print(f"ERROR: not a file: {input_path}", file=sys.stderr)
        return 2
    if output_path.exists() and not output_path.is_file():
        print(f"ERROR: output path exists and is not a file: {output_path}", file=sys.stderr)
        return 2

    setup_logging()
    config = PreprocessConfig(
        log=bool(args.log),
        epsilon=float(args.epsilon),
        clip_min=args.clip_min,
        output_dtype=str(args.output_dtype),
        data_path=args.data_path,
        angles_path=args.angles_path,
        image_key_path=args.image_key_path,
        assume_dark_field=args.assume_dark_field,
        assume_flat_field=args.assume_flat_field,
        select_views=args.select_views,
        reject_views=args.reject_views,
        select_views_file=args.select_views_file,
        reject_views_file=args.reject_views_file,
        auto_reject=str(args.auto_reject),
        outlier_z_threshold=float(args.outlier_z_threshold),
        crop=args.crop,
    )
    try:
        result = preprocess_nxtomo(input_path, output_path, config)
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
