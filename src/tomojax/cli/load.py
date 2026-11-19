"""``tomojax load``: :func:`tomojax.load`, then :func:`tomojax.save`."""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import TYPE_CHECKING, cast

from tomojax.cli._inputs import PUBLIC, add_inputs, declared, input_paths, load_keywords
from tomojax.cli._options import add_output, check_paths, fail, hide_expert

if TYPE_CHECKING:
    from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tomojax load",
        description=(
            "Load a scan and save it as a TomoJAX dataset: tj.load, then tj.save. INPUT is "
            "any file or folder tj.load reads: raw frames (HDF5 with image_key, Nikon, "
            "SkyScan, Zeiss, ESRF, a TIFF stack), corrected with the standard chain of "
            "dark, flat and -log, or a dataset of line integrals. The flags supply what "
            "INPUT does not record; tomojax correct applies the optional corrections."
        ),
        epilog=(
            "Examples:\n"
            "  tomojax load scan/scan.xtekct -o scan.nxs --darks 0\n"
            "  tomojax load raw.nxs -o scan.nxs --geometry lamino --tilt 30 --pixel-size 3.25\n"
            "  tomojax load frames/ --flats flats/ --darks darks/ --angles angles.csv "
            "--pixel-size 0.65 -o scan.nxs\n"
            "  tomojax load scan.npz -o scan.nxs"
        ),
    )
    _ = parser.add_argument("data", metavar="INPUT", help="Raw acquisition, TIFF stack or dataset")
    add_output(parser, "Dataset to write (.nxs, .h5 or .npz)")
    add_inputs(parser)
    hide_expert(parser, PUBLIC)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``tomojax load``."""
    import tomojax as tj

    parser = _build_parser()
    args = parser.parse_args(argv)
    source, output = cast("str", args.data), cast("str", args.out)
    check_paths(
        parser,
        inputs=[source, *input_paths(args)],
        outputs=[output],
        force=cast("bool", args.force),
    )
    try:
        scan = tj.load(source, **load_keywords(args))
        scan = replace(scan, geometry=declared(scan.geometry, args))
    except (KeyError, ValueError, OSError) as exc:
        return fail("load", str(exc).strip("'\""))
    tj.save(output, scan)
    print(f"wrote {output}: {scan!r}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
