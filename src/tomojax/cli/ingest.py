"""CLI: ingest external projection stacks into the TomoJAX dataset contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from tomojax.core.geometry import Detector, Grid
from tomojax.io import load_tiff_stack, save_dataset

if TYPE_CHECKING:
    from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest a TIFF projection stack into a TomoJAX .nxs/.h5/.hdf5 dataset."
    )
    _ = parser.add_argument("input", help="Input TIFF file or directory of TIFF projections")
    _ = parser.add_argument("output", nargs="?", help="Output .nxs/.h5/.hdf5 or .npz dataset")
    _ = parser.add_argument("--out", dest="out", default=None, help="Output dataset path")
    _ = parser.add_argument(
        "--angles",
        required=True,
        help="Angle sidecar: .npy array or text/CSV file with one angle in degrees per row",
    )
    _ = parser.add_argument(
        "--geometry",
        choices=["parallel", "lamino"],
        default="parallel",
        help="Acquisition geometry type recorded in metadata",
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

    angles = _load_angles(Path(cast("str", args.angles)))
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
            float(cast("float", args.det_center_u)),
            float(cast("float", args.det_center_v)),
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
    probe.sample_name = str(cast("str", args.sample_name))
    save_dataset(output, probe)
    print(f"wrote {output} from {probe.projections.shape[0]} TIFF projections")
    return 0


def _load_angles(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        values = np.asarray(np.load(path), dtype=np.float32)
    else:
        rows: list[float] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            token = text.split(",", 1)[0].strip()
            try:
                rows.append(float(token))
            except ValueError:
                if not rows:
                    continue
                raise
        values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("angle sidecar must be one-dimensional")
    return values


if __name__ == "__main__":
    raise SystemExit(main())
