"""CLI: ingest external projection stacks into the TomoJAX dataset contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, cast

from tomojax.geometry import ConeBeam, Detector, Grid
from tomojax.io import load_nikon_xtekct, load_tiff_stack, save_dataset
from tomojax.io.api import load_angles

if TYPE_CHECKING:
    from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest a TIFF projection stack, or a Nikon .xtekct scan, into a TomoJAX "
            ".nxs/.h5/.hdf5 dataset."
        )
    )
    _ = parser.add_argument(
        "input",
        help=(
            "Input TIFF file or directory of TIFF projections, or a Nikon .xtekct file "
            "(its geometry, angles and projections are read from the scan folder)"
        ),
    )
    _ = parser.add_argument("output", nargs="?", help="Output .nxs/.h5/.hdf5 or .npz dataset")
    _ = parser.add_argument("--out", dest="out", default=None, help="Output dataset path")
    _ = parser.add_argument(
        "--angles",
        default=None,
        help=(
            "Angle sidecar: .npy array or text/CSV file with one angle in degrees per row "
            "(required for TIFF stacks)"
        ),
    )
    _ = parser.add_argument(
        "--transmission",
        action="store_true",
        help="Nikon .xtekct: keep transmission intensities instead of -log(I / WhiteLevel)",
    )
    _ = parser.add_argument(
        "--reverse-angles",
        action="store_true",
        help="Nikon .xtekct: negate the stage angles (for a stage turning the other way)",
    )
    _ = parser.add_argument(
        "--geometry",
        choices=["parallel", "lamino", "cone"],
        default="parallel",
        help="Acquisition geometry type recorded in metadata",
    )
    _ = parser.add_argument(
        "--source-to-axis",
        type=float,
        default=None,
        help="Cone beam: source to rotation axis distance, in detector-pixel-size units",
    )
    _ = parser.add_argument(
        "--source-to-detector",
        type=float,
        default=None,
        help="Cone beam: source to detector distance, in the same units",
    )
    _ = parser.add_argument(
        "--axis-offset",
        type=float,
        default=0.0,
        help=(
            "Cone beam: lateral offset of the rotation axis from the source-detector "
            "centre line (centre of rotation), in the same units; "
            "`tomojax align --mode cor` estimates it"
        ),
    )
    for angle in ("roll", "pitch", "yaw"):
        _ = parser.add_argument(
            f"--detector-{angle}",
            type=float,
            default=0.0,
            help=f"Cone beam: detector {angle} in degrees (see tomojax.geometry.ConeBeam)",
        )
    _ = parser.add_argument("--du", type=float, default=1.0, help="Detector pixel size along u")
    _ = parser.add_argument("--dv", type=float, default=1.0, help="Detector pixel size along v")
    _ = parser.add_argument(
        "--det-center-u",
        type=float,
        default=0.0,
        help="Initial detector centre offset along u, in detector pixels",
    )
    _ = parser.add_argument(
        "--det-center-v",
        type=float,
        default=0.0,
        help="Initial detector centre offset along v, in detector pixels",
    )
    _ = parser.add_argument(
        "--grid",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=None,
        help="Optional reconstruction grid size to record in metadata",
    )
    _ = parser.add_argument(
        "--voxel-size",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "VZ"),
        default=(1.0, 1.0, 1.0),
        help="Voxel sizes used with --grid",
    )
    _ = parser.add_argument(
        "--sample-name", default="sample", help="Sample name stored in metadata"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the TIFF-stack ingestion command."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    output = cast("str | None", args.out) or cast("str | None", args.output)
    if output is None:
        parser.error("the following arguments are required: output or --out")

    if cast("str", args.input).lower().endswith(".xtekct"):
        dataset = load_nikon_xtekct(
            cast("str", args.input),
            absorption=not cast("bool", args.transmission),
            reverse_angles=cast("bool", args.reverse_angles),
        )
        save_dataset(output, dataset)
        print(
            f"wrote {output} from {dataset.projections.shape[0]} Nikon projections; "
            "calibrate the rotation axis with `tomojax align --mode cor`"
        )
        return 0
    angles_path = cast("str | None", args.angles)
    if angles_path is None:
        parser.error("--angles is required for TIFF stacks")
    angles = load_angles(Path(angles_path))
    probe = load_tiff_stack(
        cast("str", args.input),
        angles_deg=angles,
        geometry_type=cast("str", args.geometry),
    )
    n_dims = cast("tuple[int, int, int]", probe.projections.shape)
    _, nv, nu = n_dims
    detector = Detector(
        nu=int(nu),
        nv=int(nv),
        du=float(cast("float", args.du)),
        dv=float(cast("float", args.dv)),
        det_center=(
            float(cast("float", args.det_center_u)) * float(cast("float", args.du)),
            float(cast("float", args.det_center_v)) * float(cast("float", args.dv)),
        ),
    )
    grid = None
    raw_grid = cast("Sequence[int] | None", args.grid)
    if raw_grid is not None:
        vx, vy, vz = (float(v) for v in cast("Sequence[float]", args.voxel_size))
        grid_shape = tuple(int(v) for v in raw_grid)
        grid = Grid(
            nx=int(grid_shape[0]),
            ny=int(grid_shape[1]),
            nz=int(grid_shape[2]),
            vx=vx,
            vy=vy,
            vz=vz,
        )

    probe.detector = detector
    probe.grid = grid
    probe.geometry_type = str(cast("str", args.geometry))
    probe.geometry_metadata = {"ingest_source": "tiff_stack"}
    if probe.geometry_type == "cone":
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
        probe.geometry_metadata["cone_beam"] = beam.to_dict()
    probe.sample_name = str(cast("str", args.sample_name))
    save_dataset(output, probe)
    print(f"wrote {output} from {probe.projections.shape[0]} TIFF projections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
