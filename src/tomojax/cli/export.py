"""``tomojax export``: write a reconstruction as TIFF slices or a raw file, slice by slice."""
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import h5py
import numpy as np

from tomojax.cli._options import add_output, check_paths
from tomojax.cli._progress import ProgressParser, add_progress, run_command
from tomojax.cli._results import stored_grid
from tomojax.cli.products import VOLUME_DATASET

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomojax.wire.progress import Result


class _Args(argparse.Namespace):
    data: str
    out: str
    dtype: str
    range: list[float] | None
    prefix: str | None
    force: bool
    progress: str


def _build_parser() -> ProgressParser:
    parser = ProgressParser(
        prog="tomojax export",
        description=(
            "Export a reconstruction or alignment image for other software: a directory of "
            "z-slice TIFFs (y rows, x columns), or one little-endian z-major file when OUTPUT "
            "ends in .raw. A JSON sidecar gives the shape, voxel size, grid placement and value "
            "scaling."
        ),
        epilog=(
            "Exit status: 0 success, 1 failure, 2 usage error.\n\nExamples:\n"
            "  tomojax export recon.nxs -o slices/\n"
            "  tomojax export recon.nxs -o recon.raw --dtype uint16"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ = parser.add_argument("data", metavar="INPUT", help="Reconstruction (.nxs, .h5)")
    add_progress(parser)
    add_output(parser, "Directory for TIFF slices, or a .raw file")
    _ = parser.add_argument(
        "--dtype",
        choices=["float32", "uint16"],
        default="float32",
        help="uint16 maps --range (default: the 0.1 and 99.9 percentiles) to 0..65535",
    )
    _ = parser.add_argument(
        "--range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Values mapped to 0 and 65535 for --dtype uint16 (finite, LOW < HIGH)",
    )
    _ = parser.add_argument("--prefix", default=None, help="TIFF file name prefix (default slice)")
    return parser


def _z_slice(dataset: h5py.Dataset, axes: str, z: int) -> np.ndarray:
    """The ``(ny, nx)`` slice at height ``z`` of a volume stored with ``axes``."""
    selection: list[int | slice] = [slice(None)] * 3
    selection[axes.index("z")] = z
    plane = np.asarray(dataset[tuple(selection)], np.float32)
    rest = [axis for axis in axes if axis != "z"]
    return plane if rest == ["y", "x"] else plane.T


def main(argv: Sequence[str] | None = None) -> int:
    """Run the export command: :func:`tomojax.export`, reading the volume a slice at a time."""
    from tomojax._files import validate_export_range, write_volume

    parser = _build_parser()
    parser.set_invocation(list(sys.argv[1:] if argv is None else argv))
    args = parser.parse_args(argv, namespace=_Args())

    def run() -> Result | None:
        data, out = Path(args.data), Path(args.out)
        raw = out.suffix.lower() == ".raw"
        dtype = args.dtype
        value_range = args.range
        explicit_prefix = args.prefix
        if raw and explicit_prefix is not None:
            parser.error("--prefix applies only to TIFF directory exports, not .raw files")
        try:
            _ = validate_export_range(dtype, value_range)
        except ValueError as exc:
            parser.error(str(exc))
        prefix = "slice" if explicit_prefix is None else explicit_prefix
        # A raw volume writes a JSON sidecar beside it; refuse to replace either.
        outputs = [out, out.with_suffix(".json")] if raw else [out]
        check_paths(parser, inputs=[data], outputs=outputs, force=args.force)
        registered = {}
        with h5py.File(data, "r") as handle:
            dataset = handle.get(VOLUME_DATASET)
            if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 3:
                raise ValueError(
                    f"{data} holds no reconstructed volume; "
                    "reconstruct it with tomojax reconstruct",
                )
            grid, grid_meta, geom_meta, axes = stored_grid(handle, data)
            shape = {axis: int(n) for axis, n in zip(axes, dataset.shape, strict=True)}
            voxel = (
                (grid.vx, grid.vy, grid.vz)
                if grid is not None
                else (
                    float(grid_meta.get("vx", 1.0)),
                    float(grid_meta.get("vy", 1.0)),
                    float(grid_meta.get("vz", 1.0)),
                )
            )
            if args.progress == "json":
                from tomojax.cli._results import disk_volume

                registered["source_volume"] = disk_volume(
                    data, grid, axes, tuple(dataset.shape), dataset.dtype.str, geom_meta
                )
            sidecar = write_volume(
                out,
                lambda z: _z_slice(dataset, axes, z),
                (shape["z"], shape["y"], shape["x"]),
                voxel,
                dtype=dtype,
                value_range=value_range,
                prefix=prefix,
                source=str(data),
                grid=grid,
            )
        if args.progress == "json":
            from tomojax.cli._results import report_result

            written = [out] if raw else [out / f"{prefix}_{z:05d}.tif" for z in range(shape["z"])]
            return report_result(
                out,
                tuple(str(p) for p in [*written, sidecar]),
                json.loads(sidecar.read_text()),
                arrays=registered,
            )
        print(f"exported {shape['z']} z-slices of {shape['y']}x{shape['x']} to {out}")
        return None

    return run_command("export", args.progress, run)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
