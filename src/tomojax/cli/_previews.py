"""Preview PNGs of a dataset: its central projection and the volume's central slices.

Slices are read plane by plane, so previews of large reconstructions stay cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import h5py
import numpy as np

from tomojax._typed_arrays import numpy_float32_array, write_image
from tomojax.recon.quicklook import scale_to_uint8

_VOLUME_PATH = "/entry/processing/tomojax/volume"
_PROJECTION_PATHS = (
    "/entry/instrument/detector/data",
    "/entry/data/projections",
    "/entry/projections",
)
# Each slice is displayed with these axes as (rows, columns).
_DISPLAY_AXES = {"z": ("y", "x"), "y": ("z", "x"), "x": ("z", "y")}


def preview_paths(path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """The previews :func:`write_previews` makes for ``path``, by name."""
    out = Path(out_dir)
    paths = {"projection": out / "projection.png"}
    with h5py.File(path, "r") as file:
        if isinstance(file.get(_VOLUME_PATH), h5py.Dataset):
            paths.update({f"slice_{axis}": out / f"slice_{axis}.png" for axis in _DISPLAY_AXES})
    return paths


def write_previews(path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Write percentile-scaled PNG previews of ``path`` into ``out_dir``."""
    paths = preview_paths(path, out_dir)
    images: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as file:
        projections = next(
            (
                obj
                for obj in (file.get(key) for key in _PROJECTION_PATHS)
                if isinstance(obj, h5py.Dataset) and obj.ndim == 3
            ),
            None,
        )
        if projections is None:
            raise ValueError(f"{path} holds no 3-D projection stack")
        images["projection"] = numpy_float32_array(projections[int(projections.shape[0]) // 2])
        volume = file.get(_VOLUME_PATH)
        if isinstance(volume, h5py.Dataset):
            axes = str(file["/entry/processing/tomojax"].attrs.get("volume_axes_order", "zyx"))
            images.update({f"slice_{axis}": _central_slice(volume, axes, axis) for axis in "zyx"})
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        write_image(paths[name], scale_to_uint8(image))
    return paths


def _central_slice(volume: h5py.Dataset, axes: str, axis: str) -> np.ndarray:
    axes = axes.lower()
    if sorted(axes) != ["x", "y", "z"]:
        raise ValueError(f"unsupported saved volume axes {axes!r}")
    selection: list[int | slice] = [slice(None)] * 3
    selection[axes.index(axis)] = int(volume.shape[axes.index(axis)]) // 2
    plane = np.asarray(cast("np.ndarray", volume[tuple(selection)]), dtype=np.float32)
    remaining = [a for a in axes if a != axis]
    return np.transpose(plane, [remaining.index(a) for a in _DISPLAY_AXES[axis]])


__all__ = ["preview_paths", "write_previews"]
