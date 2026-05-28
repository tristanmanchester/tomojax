"""Extract labelled reconstruction slices without loading whole volumes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import h5py
import imageio.v3 as iio
import numpy as np

from tomojax.recon.quicklook import scale_to_uint8

if TYPE_CHECKING:
    from collections.abc import Sequence


_VOLUME_PATH = "/entry/processing/tomojax/volume"
_AXES_ATTR = "volume_axes_order"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract orthogonal PNG slices from a TomoJAX reconstruction."
    )
    _ = parser.add_argument("--data", required=True, help="Input reconstruction .nxs/.h5 file")
    _ = parser.add_argument("--out", required=True, help="Output directory for PNG slices")
    _ = parser.add_argument("--z", type=int, default=None, help="z index for y-x slice")
    _ = parser.add_argument("--y", type=int, default=None, help="y index for z-x slice")
    _ = parser.add_argument("--x", type=int, default=None, help="x index for z-y slice")
    _ = parser.add_argument("--prefix", default="slice", help="Output filename prefix")
    _ = parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing PNG/JSON outputs.",
    )
    _ = parser.add_argument("--lower-percentile", type=float, default=1.0)
    _ = parser.add_argument("--upper-percentile", type=float, default=99.0)
    return parser


def _decode_attr(value: object, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _slice_plane(
    dataset: h5py.Dataset,
    *,
    axes: str,
    fixed_axis: str,
    index: int,
) -> np.ndarray:
    if fixed_axis not in axes:
        raise ValueError(f"saved volume axes {axes!r} do not contain {fixed_axis!r}")
    if index < 0 or index >= int(dataset.shape[axes.index(fixed_axis)]):
        raise ValueError(
            f"{fixed_axis} index {index} outside saved volume shape "
            f"{tuple(int(v) for v in dataset.shape)} with axes {axes!r}"
        )

    selection: list[int | slice] = [slice(None)] * int(dataset.ndim)
    selection[axes.index(fixed_axis)] = int(index)
    plane = np.asarray(dataset[tuple(selection)], dtype=np.float32)
    remaining_axes = [axis for axis in axes if axis != fixed_axis]
    desired_axes = {
        "z": ("y", "x"),
        "y": ("z", "x"),
        "x": ("z", "y"),
    }[fixed_axis]
    if tuple(remaining_axes) != desired_axes:
        order = [remaining_axes.index(axis) for axis in desired_axes]
        plane = np.transpose(plane, order)
    return plane


def _central_index(shape: tuple[int, ...], axes: str, axis: str) -> int:
    return int(shape[axes.index(axis)]) // 2


def _temporary_path(path: Path) -> Path:
    suffix = path.suffix or ".tmp"
    return path.with_name(f".{path.stem}.tmp{suffix}")


def _prepare_slice_outputs(
    *,
    data_path: Path,
    out_dir: Path,
    prefix: str,
    z_index: int | None,
    y_index: int | None,
    x_index: int | None,
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[dict[str, Any], list[tuple[Path, np.ndarray]]]:
    outputs: dict[str, Any] = {
        "input_path": str(data_path),
        "volume_path": _VOLUME_PATH,
        "slices": {},
    }
    planned_images: list[tuple[Path, np.ndarray]] = []
    with h5py.File(data_path, "r") as handle:
        if _VOLUME_PATH not in handle:
            raise ValueError(f"{data_path} does not contain {_VOLUME_PATH}")
        dataset_obj = handle[_VOLUME_PATH]
        if not isinstance(dataset_obj, h5py.Dataset) or dataset_obj.ndim != 3:
            raise ValueError(f"{_VOLUME_PATH} must be a 3-D HDF5 dataset")
        dataset = cast("h5py.Dataset", dataset_obj)
        tomojax_group = handle["/entry/processing/tomojax"]
        axes = _decode_attr(tomojax_group.attrs.get(_AXES_ATTR), "zyx").lower()
        if sorted(axes) != ["x", "y", "z"]:
            raise ValueError(f"unsupported saved volume axes {axes!r}")
        shape = tuple(int(v) for v in dataset.shape)
        outputs["saved_axes"] = axes
        outputs["saved_shape"] = list(shape)
        requested = {
            "z": z_index if z_index is not None else _central_index(shape, axes, "z"),
            "y": y_index if y_index is not None else _central_index(shape, axes, "y"),
            "x": x_index if x_index is not None else _central_index(shape, axes, "x"),
        }
        for axis, index in requested.items():
            plane = _slice_plane(dataset, axes=axes, fixed_axis=axis, index=int(index))
            image = scale_to_uint8(
                plane,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
            )
            out_path = out_dir / f"{prefix}_{axis}{int(index):04d}.png"
            planned_images.append((out_path, image))
            outputs["slices"][axis] = {
                "index": int(index),
                "path": str(out_path),
                "display_axes": {"z": "yx", "y": "zx", "x": "zy"}[axis],
            }
    return outputs, planned_images


def _write_slice_outputs(
    *,
    planned_images: list[tuple[Path, np.ndarray]],
    summary_path: Path,
    outputs: dict[str, Any],
) -> None:
    temp_paths: list[tuple[Path, Path]] = []
    try:
        for out_path, image in planned_images:
            tmp_path = _temporary_path(out_path)
            iio.imwrite(tmp_path, image)
            temp_paths.append((tmp_path, out_path))
        summary_tmp = _temporary_path(summary_path)
        summary_tmp.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
        temp_paths.append((summary_tmp, summary_path))
        for tmp_path, out_path in temp_paths:
            tmp_path.replace(out_path)
    except OSError:
        for tmp_path, _out_path in temp_paths:
            tmp_path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Run the slice extraction CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    data_path = Path(str(args.data))
    out_dir = Path(str(args.out))
    prefix = str(args.prefix)
    summary_path = out_dir / f"{prefix}_slices.json"

    try:
        outputs, planned_images = _prepare_slice_outputs(
            data_path=data_path,
            out_dir=out_dir,
            prefix=prefix,
            z_index=args.z,
            y_index=args.y,
            x_index=args.x,
            lower_percentile=float(args.lower_percentile),
            upper_percentile=float(args.upper_percentile),
        )
    except OSError as exc:
        parser.error(f"could not read {data_path}: {exc}")
    except ValueError as exc:
        parser.error(str(exc))

    planned_paths = [path for path, _image in planned_images] + [summary_path]
    existing = [path for path in planned_paths if path.exists()]
    if existing and not bool(args.force):
        rendered = ", ".join(str(path) for path in existing[:5])
        extra = "" if len(existing) <= 5 else f", and {len(existing) - 5} more"
        parser.error(f"output file(s) already exist: {rendered}{extra}; pass --force to overwrite")

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        _write_slice_outputs(
            planned_images=planned_images,
            summary_path=summary_path,
            outputs=outputs,
        )
    except OSError as exc:
        parser.error(f"could not write slice outputs: {exc}")
    _ = print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
