from __future__ import annotations

import imageio.v3 as iio
import numpy as np
import pytest

from tomojax.core.geometry import Detector, Grid
from tomojax.io import (
    PreprocessConfig,
    ProjectionDataset,
    load_dataset,
    load_projection_payload,
    load_tiff_stack,
    preprocess_tiff_stack,
    save_dataset,
    save_projection_payload,
    validate_dataset,
)


def test_public_io_loads_nxtomo_as_projection_dataset(tmp_path):
    path = tmp_path / "scan.nxs"
    projections = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    detector = Detector(nu=4, nv=2, du=0.5, dv=0.75, det_center=(1.0, -2.0))
    grid = Grid(nx=4, ny=4, nz=2, vx=1.0, vy=1.0, vz=2.0)
    save_dataset(
        path,
        ProjectionDataset(
            projections=projections,
            angles_deg=np.asarray([0.0, 45.0, 90.0], dtype=np.float32),
            detector=detector,
            grid=grid,
            geometry_type="parallel",
            geometry_metadata={"beamline": "test"},
            sample_name="fixture",
        ),
    )

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


def test_public_io_projection_dataset_preserves_solver_metadata(tmp_path):
    path = tmp_path / "payload_from_dataset.nxs"
    projections = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    detector = Detector(nu=2, nv=2, du=0.5, dv=0.75)
    grid = Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0)
    align_params = np.ones((3, 5), dtype=np.float32)
    dataset = ProjectionDataset(
        projections=projections,
        angles_deg=np.asarray([0.0, 45.0, 90.0], dtype=np.float32),
        detector=detector,
        grid=grid,
        geometry_type="lamino",
        geometry_metadata={
            "tilt_deg": 55.0,
            "tilt_about": "x",
            "detector_roll_deg": 1.25,
        },
        angle_offset_deg=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        align_params=align_params,
        align_gauge={"mode": "mean_zero"},
    )
    save_dataset(path, dataset)

    dataset = load_dataset(path)

    np.testing.assert_allclose(dataset.projections, projections)
    np.testing.assert_allclose(dataset.angles_deg, [0.0, 45.0, 90.0])
    np.testing.assert_allclose(dataset.angle_offset_deg, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(dataset.align_params, align_params)
    assert dataset.align_gauge == {"mode": "mean_zero"}
    geometry_inputs = dataset.geometry_inputs()
    assert geometry_inputs["geometry_type"] == "lamino"
    assert geometry_inputs["tilt_deg"] == pytest.approx(55.0)
    assert geometry_inputs["tilt_about"] == "x"
    assert geometry_inputs["detector_roll_deg"] == pytest.approx(1.25)
    np.testing.assert_allclose(geometry_inputs["angle_offset_deg"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(geometry_inputs["align_params"], align_params)
    assert geometry_inputs["align_gauge"] == {"mode": "mean_zero"}


def test_public_io_roundtrips_dataset_to_nxtomo(tmp_path):
    path = tmp_path / "roundtrip.nxs"
    volume = np.arange(48, dtype=np.float32).reshape(4, 4, 3)
    dataset = ProjectionDataset(
        projections=np.ones((2, 3, 4), dtype=np.float32),
        angles_deg=np.asarray([0.0, 180.0], dtype=np.float32),
        volume=volume,
        detector=Detector(nu=4, nv=3, du=1.0, dv=1.0),
        grid=Grid(nx=4, ny=4, nz=3, vx=1.0, vy=1.0, vz=1.0),
        sample_name="roundtrip",
    )

    save_dataset(path, dataset)
    loaded = load_dataset(path)

    np.testing.assert_allclose(loaded.projections, dataset.projections)
    np.testing.assert_allclose(loaded.angles_deg, dataset.angles_deg)
    np.testing.assert_allclose(loaded.volume, volume)
    assert validate_dataset(path)["issues"] == []


def test_public_io_projection_payload_preserves_solver_metadata(tmp_path):
    path = tmp_path / "payload.nxs"
    projections = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    detector = Detector(nu=2, nv=2, du=0.5, dv=0.75)
    grid = Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0)
    align_params = np.ones((3, 5), dtype=np.float32)
    dataset = ProjectionDataset(
        projections=projections,
        angles_deg=np.asarray([0.0, 45.0, 90.0], dtype=np.float32),
        detector=detector,
        grid=grid,
        geometry_type="lamino",
        geometry_metadata={
            "tilt_deg": 55.0,
            "tilt_about": "x",
            "detector_roll_deg": 1.25,
        },
        angle_offset_deg=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        align_params=align_params,
        align_gauge={"mode": "mean_zero"},
    )
    save_projection_payload(
        path,
        projections=dataset.projections,
        metadata=dataset.to_nxtomo_metadata(),
    )

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


def test_public_io_preprocesses_tiff_stack_to_absorption_nxtomo(tmp_path):
    projections = tmp_path / "projections"
    flats = tmp_path / "flats"
    darks = tmp_path / "darks"
    for path in (projections, flats, darks):
        path.mkdir()
    iio.imwrite(projections / "0001.tif", np.full((2, 2), 5.0, dtype=np.float32))
    iio.imwrite(projections / "0002.tif", np.full((2, 2), 9.0, dtype=np.float32))
    iio.imwrite(flats / "0001.tif", np.full((2, 2), 11.0, dtype=np.float32))
    iio.imwrite(darks / "0001.tif", np.full((2, 2), 1.0, dtype=np.float32))
    angles = tmp_path / "angles.csv"
    angles.write_text("angle\n0\n90\n", encoding="utf-8")
    out_path = tmp_path / "corrected.nxs"

    result = preprocess_tiff_stack(projections, flats, darks, angles, out_path)

    assert result.output_domain == "absorption"
    loaded = load_dataset(out_path)
    np.testing.assert_allclose(
        loaded.projections[:, 0, 0],
        -np.log(np.array([0.4, 0.8], dtype=np.float32)),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(loaded.angles_deg, [0.0, 90.0])
    assert loaded.detector == Detector(nu=2, nv=2, du=1.0, dv=1.0)


def test_public_io_preprocesses_tiff_stack_can_write_transmission(tmp_path):
    projections = tmp_path / "projections"
    flats = tmp_path / "flats"
    darks = tmp_path / "darks"
    for path in (projections, flats, darks):
        path.mkdir()
    iio.imwrite(projections / "0001.tif", np.full((1, 2), 5.0, dtype=np.float32))
    iio.imwrite(flats / "0001.tif", np.full((1, 2), 11.0, dtype=np.float32))
    iio.imwrite(darks / "0001.tif", np.full((1, 2), 1.0, dtype=np.float32))
    angles = tmp_path / "angles.csv"
    angles.write_text("0\n", encoding="utf-8")
    out_path = tmp_path / "transmission.nxs"

    preprocess_tiff_stack(
        projections,
        flats,
        darks,
        angles,
        out_path,
        PreprocessConfig(output_domain="transmission"),
    )

    loaded = load_dataset(out_path)
    np.testing.assert_allclose(loaded.projections, np.full((1, 1, 2), 0.4))
