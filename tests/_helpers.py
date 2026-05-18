from __future__ import annotations

import json
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np

from tomojax.geometry import Detector, Grid
from tomojax.io import ProjectionDataset, save_dataset


def tiny_detector(*, nu: int = 4, nv: int = 2) -> Detector:
    return Detector(nu=nu, nv=nv, du=1.0, dv=1.0, det_center=(0.0, 0.0))


def tiny_grid(*, nx: int = 4, ny: int = 4, nz: int = 2) -> Grid:
    return Grid(nx=nx, ny=ny, nz=nz, vx=1.0, vy=1.0, vz=1.0)


def make_projection_dataset(
    *,
    projections: np.ndarray | None = None,
    angles_deg: np.ndarray | None = None,
    detector: Detector | None = None,
    grid: Grid | None = None,
    geometry_type: str = "parallel",
    geometry_metadata: dict[str, object] | None = None,
) -> ProjectionDataset:
    if projections is None:
        projections = np.arange(2 * 2 * 4, dtype=np.float32).reshape(2, 2, 4)
    if angles_deg is None:
        angles_deg = np.asarray([0.0, 90.0], dtype=np.float32)
    return ProjectionDataset(
        projections=np.asarray(projections, dtype=np.float32),
        angles_deg=np.asarray(angles_deg, dtype=np.float32),
        detector=detector
        or tiny_detector(nu=int(projections.shape[2]), nv=int(projections.shape[1])),
        grid=grid or tiny_grid(nz=int(projections.shape[1])),
        geometry_type=geometry_type,
        geometry_metadata=dict(geometry_metadata or {}),
        sample_name="fixture",
    )


def write_projection_dataset(path: Path, **kwargs: object) -> ProjectionDataset:
    dataset = make_projection_dataset(**kwargs)
    save_dataset(path, dataset)
    return dataset


def write_tiff_stack(path: Path, values: list[float], *, shape: tuple[int, int] = (2, 2)) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for index, value in enumerate(values, start=1):
        iio.imwrite(path / f"{index:04d}.tif", np.full(shape, value, dtype=np.float32))


def write_angle_csv(path: Path, values: list[float]) -> None:
    path.write_text("angle\n" + "\n".join(str(value) for value in values) + "\n", encoding="utf-8")


def write_raw_nxtomo(path: Path) -> None:
    frames = np.stack(
        [
            np.full((2, 2), 5.0, dtype=np.float32),
            np.full((2, 2), 11.0, dtype=np.float32),
            np.full((2, 2), 9.0, dtype=np.float32),
            np.full((2, 2), 1.0, dtype=np.float32),
        ],
        axis=0,
    )
    image_key = np.asarray([0, 1, 0, 2], dtype=np.int32)
    angles = np.asarray([0.0, 0.0, 90.0, 0.0], dtype=np.float32)
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["definition"] = "NXtomo"
        entry.attrs["grid_meta_json"] = json.dumps(tiny_grid().to_dict())
        geometry = entry.create_group("geometry")
        geometry.attrs["type"] = "parallel"
        instrument = entry.create_group("instrument")
        detector = instrument.create_group("detector")
        detector.create_dataset("data", data=frames)
        detector.create_dataset("image_key", data=image_key)
        detector.create_dataset("x_pixel_size", data=np.asarray(1.0, dtype=np.float32))
        detector.create_dataset("y_pixel_size", data=np.asarray(1.0, dtype=np.float32))
        detector.attrs["detector_meta_json"] = json.dumps(tiny_detector(nu=2, nv=2).to_dict())
        sample = entry.create_group("sample")
        sample.create_dataset("name", data="fixture")
        transformations = sample.create_group("transformations")
        rotation_angle = transformations.create_dataset("rotation_angle", data=angles)
        rotation_angle.attrs["units"] = "degree"
