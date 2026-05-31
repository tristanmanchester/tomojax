"""Quicklook image writers for projection datasets."""

from __future__ import annotations

from pathlib import Path

import h5py

from tomojax._typed_arrays import numpy_float32_array, write_image
from tomojax.recon.quicklook import scale_to_uint8

type PathLike = str | Path

_PROJECTION_PATHS = (
    "/entry/instrument/detector/data",
    "/entry/data/projections",
    "/entry/projections",
)


def _projection_dataset(file: h5py.File) -> h5py.Dataset | None:
    for path in _PROJECTION_PATHS:
        obj = file.get(path)
        if isinstance(obj, h5py.Dataset):
            return obj
    return None


def save_projection_quicklook(input_path: PathLike, output_path: PathLike) -> Path:
    """Save a percentile-scaled central projection PNG."""
    in_path = Path(input_path)
    out_path = Path(output_path)
    with h5py.File(in_path, "r") as file:
        dataset = _projection_dataset(file)
        if dataset is None:
            raise KeyError("Could not find projections dataset under /entry")
        if dataset.ndim != 3:
            raise ValueError(
                f"projection dataset must be 3D (n_views, nv, nu), got {dataset.shape}"
            )
        central = numpy_float32_array(dataset[int(dataset.shape[0]) // 2])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_image(out_path, scale_to_uint8(central))
    return out_path


__all__ = ["save_projection_quicklook"]
