"""Preview PNGs of a dataset: its central projection and the volume's central slices.

Slices are read plane by plane, so previews of large reconstructions stay cheap.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from tomojax._typed_arrays import write_image
from tomojax.io.api import (
    locate_frames,
    read_acquisition,
    saved_volume_axes,
    saved_without_frames,
)
from tomojax.io.projection_layout import saved_npz_layout, saved_projection_layout
from tomojax.recon.quicklook import scale_to_uint8

_VOLUME_PATH = "/entry/processing/tomojax/volume"
# Each slice is displayed with these axes as (rows, columns).
_DISPLAY_AXES = {"z": ("y", "x"), "y": ("z", "x"), "x": ("z", "y")}


def preview_paths(path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """The previews :func:`write_previews` makes for ``path``, by name."""
    out = Path(out_dir)
    paths = {"projection": out / "projection.png"}
    if Path(path).suffix.lower() == ".npz":
        _, members = saved_npz_layout(Path(path))
        volume, projections = "volume" in members, "projections" in members
    elif h5py.is_hdf5(path):
        with h5py.File(path, "r") as file:
            volume = isinstance(file.get(_VOLUME_PATH), h5py.Dataset)
        projections = not saved_without_frames(path)
    else:
        return paths
    if volume:
        paths.update({f"slice_{axis}": out / f"slice_{axis}.png" for axis in _DISPLAY_AXES})
    if not projections:
        del paths["projection"]
    return paths


def write_previews(path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Write percentile-scaled PNG previews of ``path`` into ``out_dir``."""
    images = {}
    file_path = Path(path)
    if file_path.suffix.lower() == ".npz":
        with ExitStack() as resources:
            record, members = saved_npz_layout(file_path, resources=resources)
            if (stack := members.get("projections")) is not None:
                key = record.image_key
                samples = stack if key is None else stack.frames_at(key == 0)
                images["projection"] = _middle_projection(samples, file_path)
            if (volume := members.get("volume")) is not None:
                windows = []
                for i in range(3):
                    ranges = [range(n) for n in volume.shape]
                    mid = volume.shape[i] // 2
                    ranges[i] = range(mid, mid + 1)
                    windows.append(ranges)
                planes = volume.sampled_windows(windows)
                images.update(
                    {f"slice_{axis}": planes[i].squeeze(i) for i, axis in enumerate("zyx")}
                )
        paths = {name: Path(out_dir) / f"{name}.png" for name in images}
    else:
        paths = preview_paths(path, out_dir)
        if "slice_z" in paths:
            with h5py.File(path, "r") as file:
                images.update(_slices(file, path))
        if "projection" in paths:
            images["projection"] = _middle_projection(_projection_samples(file_path), file_path)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        write_image(paths[name], scale_to_uint8(image))
    return paths


def _middle_projection(samples, path):
    views = samples.shape[0]
    if not views:
        raise ValueError(f"{path} holds no sample projections")
    return samples[views // 2]


def _projection_samples(path: Path):
    """Locate logical sample frames even when the scan lacks geometry or calibration."""
    if (raw := read_acquisition(path)) is not None:
        return raw.counts
    if path.suffix.lower() == ".mat":
        _, samples = saved_projection_layout(path)
        return samples
    return locate_frames(str(path)).samples


def _slices(file: h5py.File, path: str | Path) -> dict[str, np.ndarray]:
    """The saved volume's central slices, by name; none without a volume."""
    volume = file.get(_VOLUME_PATH)
    if not isinstance(volume, h5py.Dataset):
        return {}
    attribute = file["/entry/processing/tomojax"].attrs.get("volume_axes_order")
    axes = saved_volume_axes(None if attribute is None else str(attribute), path)
    return {f"slice_{axis}": _central_slice(volume, axes, axis) for axis in "zyx"}


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
