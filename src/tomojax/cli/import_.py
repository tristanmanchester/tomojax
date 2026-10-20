"""``tomojax import``: turn raw scans into a TomoJAX dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, cast

from tomojax.cli._options import add_config, add_output, check_paths
from tomojax.cli.config import parse_args_with_config
from tomojax.geometry import ConeBeam, Detector
from tomojax.io import load_dataset, load_tiff_stack, save_dataset
from tomojax.io.api import load_angles

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from tomojax.io import ProjectionDataset

_DATASETS = (".nxs", ".h5", ".hdf", ".hdf5", ".npz")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tomojax import",
        description=(
            "Write a TomoJAX dataset (.nxs, or .npz) from a TIFF line-integral "
            "stack or another TomoJAX dataset. Raw acquisitions use tomojax preprocess. "
            "TIFF stacks need --angles, --pixel-size, --geometry and detector_center in --config."
        ),
        epilog=(
            "Lengths (--pixel-size, --source-to-axis, --source-to-detector, --axis-offset) "
            "share one unit, which is the reconstruction's length unit.\n\n"
            "Exit status: 0 success, 1 failure, 2 usage error.\n\nExamples:\n"
            "  tomojax import projections/ --angles angles.csv --pixel-size 0.65 "
            "--geometry parallel --config calibration.toml -o scan.nxs\n"
            "  tomojax import projections/ --angles angles.csv --geometry cone \\\n"
            "      --source-to-axis 120 --source-to-detector 800 --pixel-size 0.2 "
            "--config calibration.toml -o scan.nxs\n"
            "  tomojax import scan.npz -o scan.nxs"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ = parser.add_argument(
        "data",
        metavar="INPUT",
        help="TIFF line-integral stack, or a .nxs/.npz dataset",
    )
    add_output(parser, "Dataset to write (.nxs, .h5 or .npz)")
    add_config(
        parser,
        settings={
            "detector_center": None,
            "tilt_deg": None,
            "tilt_about": "x",
            "detector_roll_deg": 0.0,
            "detector_pitch_deg": 0.0,
            "detector_yaw_deg": 0.0,
        },
    )
    tiff = parser.add_argument_group("TIFF stacks")
    _ = tiff.add_argument(
        "--angles",
        metavar="FILE",
        help="Rotation angles in degrees: .npy, or text/CSV with one per line (required)",
    )
    _ = tiff.add_argument("--geometry", choices=["parallel", "lamino", "cone"])
    _ = tiff.add_argument(
        "--pixel-size",
        type=float,
        nargs="+",
        metavar=("SIZE", "SIZE_V"),
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
    _ = parser.add_argument("--name", help="Sample name to record")
    return parser


def _geometry_metadata(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    settings: Mapping[str, object],
    *,
    geometry: str,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "ingest_source": "tiff_stack",
        "detector_center_status": "provided",
    }
    if geometry == "lamino":
        if settings.get("tilt_deg") is None:
            parser.error("Laminography needs tilt_deg in --config")
        metadata.update(
            tilt_deg=float(cast("float", settings["tilt_deg"])),
            tilt_about=str(settings.get("tilt_about", "x")),
        )
    if geometry == "cone":
        sod = cast("float | None", args.source_to_axis)
        sdd = cast("float | None", args.source_to_detector)
        if sod is None or sdd is None:
            parser.error("--geometry cone needs --source-to-axis and --source-to-detector")
        try:
            beam = ConeBeam(
                sod,
                sdd,
                detector_roll_deg=float(cast("float", settings.get("detector_roll_deg", 0.0))),
                detector_pitch_deg=float(cast("float", settings.get("detector_pitch_deg", 0.0))),
                detector_yaw_deg=float(cast("float", settings.get("detector_yaw_deg", 0.0))),
                axis_offset=cast("float", args.axis_offset),
            )
        except ValueError as exc:
            parser.error(str(exc))
        metadata["cone_beam"] = beam.to_dict()
    return metadata


def _tiff_stack(
    parser: argparse.ArgumentParser, args: argparse.Namespace, settings: Mapping[str, object]
) -> ProjectionDataset:
    angles = cast("str | None", args.angles)
    if angles is None:
        parser.error("TIFF stacks need --angles")
    sizes = cast("list[float] | None", args.pixel_size)
    if sizes is None:
        parser.error("TIFF stacks need --pixel-size")
    if len(sizes) > 2:
        parser.error("--pixel-size takes one value, or two (u then v)")
    du, dv = sizes[0], sizes[-1]
    geometry = cast("str | None", args.geometry)
    if geometry is None:
        parser.error("TIFF stacks need --geometry")
    if settings.get("detector_center") is None:
        parser.error("TIFF stacks need detector_center = [u, v] in --config")
    center = tuple(float(v) for v in cast("Sequence[float]", settings["detector_center"]))
    if len(center) != 2:
        parser.error("detector_center must contain u and v in the pixel-size length unit")
    metadata = _geometry_metadata(parser, args, settings, geometry=geometry)
    record = load_tiff_stack(
        cast("str", args.data), angles=load_angles(Path(angles)), geometry_type=geometry
    )
    nv, nu = cast("tuple[int, int, int]", record.projections.shape)[1:]
    record.detector = Detector(nu=nu, nv=nv, du=du, dv=dv, center=(center[0], center[1]))
    record.geometry_metadata = metadata
    return record


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``tomojax import``."""
    parser = _build_parser()
    args, config_meta = parse_args_with_config(parser, argv)
    settings = cast("Mapping[str, object]", config_meta["settings"])
    source, output = cast("str", args.data), cast("str", args.out)
    check_paths(parser, inputs=[source], outputs=[output], force=cast("bool", args.force))
    suffix = Path(source).suffix.lower()
    if suffix == ".xtekct":
        parser.error("Raw Nikon counts need tomojax preprocess INPUT --darks ... -o OUTPUT")
    record = load_dataset(source) if suffix in _DATASETS else _tiff_stack(parser, args, settings)
    name = cast("str | None", args.name)
    if name is not None:
        record.sample_name = name
    save_dataset(output, record)
    print(f"wrote {output} ({record.projections.shape[0]} projections)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
