"""``tomojax export``: write a reconstruction as TIFF slices or a raw file, slice by slice."""
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import h5py
import numpy as np

from tomojax.cli._options import add_output, check_paths
from tomojax.io.api import saved_volume_axes

if TYPE_CHECKING:
    from collections.abc import Sequence

_VOLUME_PATH = "/entry/processing/tomojax/volume"
_AXES_ATTR = "volume_axes_order"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tomojax export",
        description=(
            "Export a reconstruction for other software: a directory of z-slice TIFFs "
            "(y rows, x columns), or one little-endian z-major file when OUTPUT ends in "
            ".raw. A JSON sidecar gives the shape, voxel size and value scaling."
        ),
        epilog=(
            "Exit status: 0 success, 1 failure, 2 usage error.\n\nExamples:\n"
            "  tomojax export recon.nxs -o slices/\n"
            "  tomojax export recon.nxs -o recon.raw --dtype uint16"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ = parser.add_argument("data", metavar="INPUT", help="Reconstruction (.nxs, .h5)")
    add_output(parser, "Directory for TIFF slices, or a .raw file")
    _ = parser.add_argument(
        "--dtype",
        choices=["float32", "uint16"],
        default="float32",
        help="uint16 maps --range (default: the 0.1 and 99.9 percentiles) to 0..65535",
    )
    _ = parser.add_argument("--range", type=float, nargs=2, metavar=("LOW", "HIGH"), default=None)
    _ = parser.add_argument("--prefix", default="slice", help="TIFF file name prefix")
    return parser


def _decode(value: object, default: str) -> str:
    if value is None:
        return default
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _z_slice(dataset: h5py.Dataset, axes: str, z: int) -> np.ndarray:
    """The ``(ny, nx)`` slice at height ``z`` of a volume stored with ``axes``."""
    selection: list[int | slice] = [slice(None)] * 3
    selection[axes.index("z")] = z
    plane = np.asarray(dataset[tuple(selection)], np.float32)
    rest = [axis for axis in axes if axis != "z"]
    return plane if rest == ["y", "x"] else plane.T


def main(argv: Sequence[str] | None = None) -> int:
    """Run the export command: :func:`tomojax.export`, reading the volume a slice at a time."""
    from tomojax._files import write_volume

    parser = _build_parser()
    args = parser.parse_args(argv)
    data, out = Path(cast("str", args.data)), Path(cast("str", args.out))
    # A raw volume writes a JSON sidecar beside it; refuse to replace either.
    outputs = [out, out.with_suffix(".json")] if out.suffix.lower() == ".raw" else [out]
    check_paths(parser, inputs=[data], outputs=outputs, force=cast("bool", args.force))
    with h5py.File(data, "r") as handle:
        dataset = handle.get(_VOLUME_PATH)
        if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 3:
            parser.error(
                f"{data} holds no reconstructed volume; reconstruct it with tomojax reconstruct"
            )
        grid = cast(
            "dict[str, float]",
            json.loads(_decode(handle["/entry"].attrs.get("grid_meta_json"), "{}")),
        )
        attribute = handle["/entry/processing/tomojax"].attrs.get(_AXES_ATTR)
        try:
            axes = saved_volume_axes(None if attribute is None else _decode(attribute, ""), data)
        except ValueError as exc:
            parser.error(str(exc))
        shape = {axis: int(n) for axis, n in zip(axes, dataset.shape, strict=True)}
        _ = write_volume(
            out,
            lambda z: _z_slice(dataset, axes, z),
            (shape["z"], shape["y"], shape["x"]),
            (grid.get("vx", 1.0), grid.get("vy", 1.0), grid.get("vz", 1.0)),
            dtype=cast("str", args.dtype),
            value_range=cast("list[float] | None", args.range),
            prefix=cast("str", args.prefix),
            source=str(data),
        )
    print(f"exported {shape['z']} z-slices of {shape['y']}x{shape['x']} to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
