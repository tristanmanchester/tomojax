"""``tomojax import``: turn raw scans into a TomoJAX dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, cast

from tomojax.cli._options import add_output, check_paths
from tomojax.geometry import ConeBeam, Detector
from tomojax.io import load_dataset, load_nikon_xtekct, load_tiff_stack, save_dataset
from tomojax.io.api import load_angles

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomojax.io import ProjectionDataset

_DATASETS = (".nxs", ".h5", ".hdf5", ".npz")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tomojax import",
        description=(
            "Write a TomoJAX dataset (.nxs, or .npz) from a Nikon .xtekct scan, a TIFF "
            "projection stack, or another TomoJAX dataset. Nikon scans carry their own "
            "geometry; TIFF stacks need --angles and, for cone beam, the distances."
        ),
        epilog=(
            "Lengths (--pixel-size, --source-to-axis, --source-to-detector, --axis-offset) "
            "share one unit, which is the reconstruction's length unit.\n\n"
            "Exit status: 0 success, 1 failure, 2 usage error.\n\nExamples:\n"
            "  tomojax import scan/scan.xtekct -o scan.nxs\n"
            "  tomojax import projections/ --angles angles.csv --pixel-size 0.65 -o scan.nxs\n"
            "  tomojax import projections/ --angles angles.csv --geometry cone \\\n"
            "      --source-to-axis 120 --source-to-detector 800 --pixel-size 0.2 -o scan.nxs\n"
            "  tomojax import scan.npz -o scan.nxs"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ = parser.add_argument(
        "data",
        metavar="INPUT",
        help="Nikon .xtekct file, TIFF file or directory of TIFFs, or a .nxs/.npz dataset",
    )
    add_output(parser, "Dataset to write (.nxs, .h5 or .npz)")
    tiff = parser.add_argument_group("TIFF stacks")
    _ = tiff.add_argument(
        "--angles",
        metavar="FILE",
        help="Rotation angles in degrees: .npy, or text/CSV with one per line (required)",
    )
    _ = tiff.add_argument("--geometry", choices=["parallel", "lamino", "cone"], default="parallel")
    _ = tiff.add_argument(
        "--pixel-size",
        type=float,
        nargs="+",
        metavar=("SIZE", "SIZE_V"),
        default=[1.0],
        help="Detector pixel size (one value, or u then v)",
    )
    _ = tiff.add_argument("--source-to-axis", type=float, help="Cone beam: source to axis")
    _ = tiff.add_argument("--source-to-detector", type=float, help="Cone beam: source to detector")
    _ = tiff.add_argument(
        "--axis-offset",
        type=float,
        default=0.0,
        help="Cone beam: rotation axis offset from the central ray (tomojax align --mode cor "
        "estimates it)",
    )
    for angle in ("roll", "pitch", "yaw"):
        _ = tiff.add_argument(
            f"--detector-{angle}",
            type=float,
            default=0.0,
            help=f"Cone beam: detector {angle}, degrees",
        )
    nikon = parser.add_argument_group("Nikon scans")
    _ = nikon.add_argument(
        "--transmission",
        action="store_true",
        help="Keep transmission intensities instead of -log(I / WhiteLevel)",
    )
    _ = nikon.add_argument(
        "--reverse-angles",
        action="store_true",
        help="Negate the stage angles (for a stage turning the other way)",
    )
    _ = parser.add_argument("--name", help="Sample name to record")
    return parser


def _tiff_stack(parser: argparse.ArgumentParser, args: argparse.Namespace) -> ProjectionDataset:
    angles = cast("str | None", args.angles)
    if angles is None:
        parser.error("TIFF stacks need --angles")
    sizes = cast("list[float]", args.pixel_size)
    if len(sizes) > 2:
        parser.error("--pixel-size takes one value, or two (u then v)")
    du, dv = sizes[0], sizes[-1]
    geometry = cast("str", args.geometry)
    record = load_tiff_stack(
        cast("str", args.data), angles=load_angles(Path(angles)), geometry_type=geometry
    )
    nv, nu = cast("tuple[int, int, int]", record.projections.shape)[1:]
    record.detector = Detector(nu=nu, nv=nv, du=du, dv=dv)
    record.geometry_metadata = {"ingest_source": "tiff_stack"}
    if geometry == "cone":
        sod = cast("float | None", args.source_to_axis)
        sdd = cast("float | None", args.source_to_detector)
        if sod is None or sdd is None:
            parser.error("--geometry cone needs --source-to-axis and --source-to-detector")
        try:
            beam = ConeBeam(
                sod,
                sdd,
                detector_roll_deg=cast("float", args.detector_roll),
                detector_pitch_deg=cast("float", args.detector_pitch),
                detector_yaw_deg=cast("float", args.detector_yaw),
                axis_offset=cast("float", args.axis_offset),
            )
        except ValueError as exc:
            parser.error(str(exc))
        record.geometry_metadata["cone_beam"] = beam.to_dict()
    return record


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``tomojax import``."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    source, output = cast("str", args.data), cast("str", args.out)
    check_paths(parser, inputs=[source], outputs=[output], force=cast("bool", args.force))
    suffix = Path(source).suffix.lower()
    if suffix == ".xtekct":
        record = load_nikon_xtekct(
            source,
            absorption=not cast("bool", args.transmission),
            reverse_angles=cast("bool", args.reverse_angles),
        )
        hint = "; estimate the rotation axis with `tomojax align --mode cor`"
    elif suffix in _DATASETS:
        record, hint = load_dataset(source), ""
    else:
        record, hint = _tiff_stack(parser, args), ""
    name = cast("str | None", args.name)
    if name is not None:
        record.sample_name = name
    save_dataset(output, record)
    print(f"wrote {output} ({record.projections.shape[0]} projections){hint}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
