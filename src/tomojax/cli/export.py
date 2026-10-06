"""``tomojax export``: write a reconstruction as TIFF slices or a raw file, slice by slice."""
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import h5py
import imageio.v3 as iio
import numpy as np

from tomojax.cli._options import add_output, check_paths

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


def _value_range(
    dataset: h5py.Dataset, axes: str, nz: int, requested: Sequence[float] | None
) -> tuple[float, float]:
    if requested is not None:
        return float(requested[0]), float(requested[1])
    # Percentiles of a subsample of slices keep the pass over the volume cheap.
    count = min(nz, 16)
    picks = sorted({round(i * (nz - 1) / max(1, count - 1)) for i in range(count)})
    values = np.concatenate([_z_slice(dataset, axes, z).ravel() for z in picks])
    low, high = cast(
        "list[float]", np.percentile(values[np.isfinite(values)], [0.1, 99.9]).tolist()
    )
    return low, high if high > low else low + 1.0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the export command."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    data, out = Path(cast("str", args.data)), Path(cast("str", args.out))
    check_paths(parser, inputs=[data], outputs=[out], force=cast("bool", args.force))
    fmt = "raw" if out.suffix.lower() == ".raw" else "tiff"
    dtype, prefix = cast("str", args.dtype), cast("str", args.prefix)
    with h5py.File(data, "r") as handle:
        dataset = handle.get(_VOLUME_PATH)
        if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 3:
            parser.error(f"{data} holds no reconstructed volume; reconstruct it with tomojax recon")
        axes = _decode(handle["/entry/processing/tomojax"].attrs.get(_AXES_ATTR), "zyx").lower()
        if sorted(axes) != ["x", "y", "z"]:
            parser.error(f"unsupported saved volume axes {axes!r}")
        shape = {axis: int(n) for axis, n in zip(axes, dataset.shape, strict=True)}
        grid = cast(
            "dict[str, float]",
            json.loads(_decode(handle["/entry"].attrs.get("grid_meta_json"), "{}")),
        )
        low, high = (
            _value_range(dataset, axes, shape["z"], cast("list[float] | None", args.range))
            if dtype == "uint16"
            else (0.0, 1.0)
        )

        def convert(plane: np.ndarray) -> np.ndarray:
            if dtype == "float32":
                return plane
            scaled = (plane - low) / (high - low) * 65535.0
            return np.clip(np.round(scaled), 0, 65535).astype(np.uint16)

        if fmt == "tiff":
            out.mkdir(parents=True, exist_ok=True)
            for z in range(shape["z"]):
                path = out / f"{prefix}_{z:05d}.tif"
                _ = iio.imwrite(path, convert(_z_slice(dataset, axes, z)))
            sidecar = out / f"{prefix}.json"
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            little = np.dtype(dtype).newbyteorder("<")
            with out.open("wb") as stream:
                for z in range(shape["z"]):
                    _ = stream.write(convert(_z_slice(dataset, axes, z)).astype(little).tobytes())
            sidecar = out.with_suffix(".json")
    info: dict[str, object] = {
        "source": str(data),
        "format": fmt,
        "dtype": dtype,
        "shape_zyx": [shape["z"], shape["y"], shape["x"]],
        "voxel_size_xyz": [grid.get("vx"), grid.get("vy"), grid.get("vz")],
        "byte_order": "little" if fmt == "raw" else None,
        "value_range": [low, high] if dtype == "uint16" else None,
        "slice_axis": "z, from the bottom of the volume",
    }
    _ = sidecar.write_text(json.dumps(info, indent=2) + "\n")
    print(f"exported {shape['z']} z-slices of {shape['y']}x{shape['x']} to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
