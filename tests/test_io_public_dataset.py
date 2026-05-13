from __future__ import annotations

import imageio.v3 as iio
import numpy as np
import pytest

from tomojax.core.geometry import Detector, Grid
from tomojax.data.io_hdf5 import NXTomoMetadata, save_nxtomo
from tomojax.io import (
    ProjectionDataset,
    load_dataset,
    load_projection_payload,
    load_tiff_stack,
    save_dataset,
    validate_dataset,
)


def test_public_io_loads_nxtomo_as_projection_dataset(tmp_path):
    path = tmp_path / "scan.nxs"
    projections = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    detector = Detector(nu=4, nv=2, du=0.5, dv=0.75, det_center=(1.0, -2.0))
    grid = Grid(nx=4, ny=4, nz=2, vx=1.0, vy=1.0, vz=2.0)
    metadata = NXTomoMetadata(
        thetas_deg=np.asarray([0.0, 45.0, 90.0], dtype=np.float32),
        detector=detector,
        grid=grid,
        geometry_type="parallel",
        geometry_meta={"beamline": "test"},
        sample_name="fixture",
    )
    save_nxtomo(path, projections=projections, metadata=metadata)

    dataset = load_dataset(path)

    assert isinstance(dataset, ProjectionDataset)
    np.testing.assert_allclose(dataset.projections, projections)
    np.testing.assert_allclose(dataset.angles_deg, [0.0, 45.0, 90.0])
    assert dataset.detector == detector
    assert dataset.grid == grid
    assert dataset.geometry_metadata == {"beamline": "test"}
    assert dataset.source_format == "nxtomo"
    geometry_inputs = dataset.geometry_inputs()
    assert geometry_inputs["detector"] == detector.to_dict()
    np.testing.assert_allclose(geometry_inputs["thetas_deg"], [0.0, 45.0, 90.0])
    assert geometry_inputs["geometry_type"] == "parallel"


def test_public_io_roundtrips_dataset_to_nxtomo(tmp_path):
    path = tmp_path / "roundtrip.nxs"
    dataset = ProjectionDataset(
        projections=np.ones((2, 3, 4), dtype=np.float32),
        angles_deg=np.asarray([0.0, 180.0], dtype=np.float32),
        detector=Detector(nu=4, nv=3, du=1.0, dv=1.0),
        grid=Grid(nx=4, ny=4, nz=3, vx=1.0, vy=1.0, vz=1.0),
        sample_name="roundtrip",
    )

    save_dataset(path, dataset)
    loaded = load_dataset(path)

    np.testing.assert_allclose(loaded.projections, dataset.projections)
    np.testing.assert_allclose(loaded.angles_deg, dataset.angles_deg)
    assert validate_dataset(path)["issues"] == []


def test_public_io_projection_payload_preserves_solver_metadata(tmp_path):
    path = tmp_path / "payload.nxs"
    projections = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    detector = Detector(nu=2, nv=2, du=0.5, dv=0.75)
    grid = Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0)
    align_params = np.ones((3, 5), dtype=np.float32)
    metadata = NXTomoMetadata(
        thetas_deg=np.asarray([0.0, 45.0, 90.0], dtype=np.float32),
        detector=detector,
        grid=grid,
        geometry_type="lamino",
        geometry_meta={
            "tilt_deg": 55.0,
            "tilt_about": "x",
            "detector_roll_deg": 1.25,
        },
        angle_offset_deg=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        align_params=align_params,
        align_gauge={"mode": "mean_zero"},
    )
    save_nxtomo(path, projections=projections, metadata=metadata)

    payload = load_projection_payload(path)

    assert isinstance(payload, ProjectionDataset)
    geometry_inputs = payload.geometry_inputs()
    assert geometry_inputs["geometry_type"] == "lamino"
    assert geometry_inputs["tilt_deg"] == pytest.approx(55.0)
    assert geometry_inputs["tilt_about"] == "x"
    assert geometry_inputs["detector_roll_deg"] == pytest.approx(1.25)
    np.testing.assert_allclose(geometry_inputs["angle_offset_deg"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(geometry_inputs["align_params"], align_params)
    assert geometry_inputs["align_gauge"] == {"mode": "mean_zero"}

    copied = payload.copy_metadata()
    assert copied.geometry_type == "lamino"
    np.testing.assert_allclose(copied.align_params, align_params)
    assert copied.align_gauge == {"mode": "mean_zero"}


def test_public_io_loads_tiff_stack_with_explicit_angles(tmp_path):
    stack_dir = tmp_path / "tiffs"
    stack_dir.mkdir()
    iio.imwrite(stack_dir / "0002.tif", np.full((2, 3), 2, dtype=np.float32))
    iio.imwrite(stack_dir / "0001.tif", np.full((2, 3), 1, dtype=np.float32))

    dataset = load_tiff_stack(stack_dir, angles_deg=[0.0, 90.0])

    assert dataset.source_format == "tiff_stack"
    assert dataset.projections.shape == (2, 2, 3)
    np.testing.assert_allclose(dataset.projections[:, 0, 0], [1.0, 2.0])
    np.testing.assert_allclose(dataset.angles_deg, [0.0, 90.0])


def test_public_io_rejects_tiff_stack_without_angles(tmp_path):
    stack_dir = tmp_path / "tiffs"
    stack_dir.mkdir()
    iio.imwrite(stack_dir / "0001.tif", np.ones((2, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="TIFF inputs require angle metadata"):
        load_dataset(stack_dir)

    with pytest.raises(ValueError, match="does not match projection count"):
        load_tiff_stack(stack_dir, angles_deg=[0.0, 90.0])
