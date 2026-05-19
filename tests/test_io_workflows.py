from __future__ import annotations

from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
import pytest

from tomojax.geometry import Detector, Grid
from tomojax.io import (
    ProjectionDataset,
    absorption_to_transmission,
    convert_dataset,
    flat_dark_to_absorption,
    load_dataset,
    load_nxtomo,
    load_real_laminography_input,
    load_tiff_stack,
    save_dataset,
    save_projection_payload,
    validate_dataset,
)

from ._helpers import make_projection_dataset, write_projection_dataset


def test_projection_dataset_roundtrips_nxtomo_with_solver_metadata(tmp_path: Path) -> None:
    path = tmp_path / "scan.nxs"
    detector = Detector(nu=4, nv=2, du=0.5, dv=0.75, det_center=(1.0, -2.0))
    grid = Grid(nx=4, ny=4, nz=2, vx=1.0, vy=1.0, vz=2.0)
    align_params = np.ones((2, 5), dtype=np.float32)
    dataset = make_projection_dataset(
        detector=detector,
        grid=grid,
        geometry_type="lamino",
        geometry_metadata={"tilt_deg": 55.0, "tilt_about": "x", "detector_roll_deg": 1.25},
    )
    dataset.angle_offset_deg = np.asarray([1.0, 2.0], dtype=np.float32)
    dataset.align_params = align_params
    dataset.align_gauge = {"mode": "mean_zero"}

    save_dataset(path, dataset)
    loaded = load_dataset(path)

    assert isinstance(loaded, ProjectionDataset)
    np.testing.assert_allclose(loaded.projections, dataset.projections)
    np.testing.assert_allclose(loaded.angles_deg, dataset.angles_deg)
    assert loaded.detector == detector
    assert loaded.grid == grid
    geometry_inputs = loaded.geometry_inputs()
    assert geometry_inputs["geometry_type"] == "lamino"
    assert geometry_inputs["tilt_deg"] == pytest.approx(55.0)
    assert geometry_inputs["tilt_about"] == "x"
    assert geometry_inputs["detector_roll_deg"] == pytest.approx(1.25)
    np.testing.assert_allclose(geometry_inputs["angle_offset_deg"], [1.0, 2.0])
    np.testing.assert_allclose(geometry_inputs["align_params"], align_params)
    assert geometry_inputs["align_gauge"] == {"mode": "mean_zero"}


def test_projection_payload_save_load_preserves_metadata_copy(tmp_path: Path) -> None:
    path = tmp_path / "payload.nxs"
    dataset = make_projection_dataset(geometry_metadata={"axis_unit_lab": [0.0, 0.0, 1.0]})
    save_projection_payload(
        path, projections=dataset.projections, metadata=dataset.to_nxtomo_metadata()
    )

    payload = load_dataset(path)

    copied = payload.copy_metadata()
    assert copied.sample_name == "fixture"
    assert copied.detector == payload.detector
    assert copied.grid == payload.grid
    assert payload.geometry_inputs()["axis_unit_lab"] == [0.0, 0.0, 1.0]


def test_dataset_roundtrips_npz_and_converts_to_nxtomo(tmp_path: Path) -> None:
    npz_path = tmp_path / "scan.npz"
    nxs_path = tmp_path / "scan.nxs"
    dataset = make_projection_dataset()

    save_dataset(npz_path, dataset)
    loaded_npz = load_dataset(npz_path)
    convert_dataset(npz_path, nxs_path)
    loaded_nxs = load_dataset(nxs_path)

    np.testing.assert_allclose(loaded_npz.projections, dataset.projections)
    np.testing.assert_allclose(loaded_nxs.projections, dataset.projections)
    assert validate_dataset(nxs_path)["issues"] == []


def test_nxtomo_loader_does_not_synthesize_grid_from_volume_only(tmp_path: Path) -> None:
    path = tmp_path / "missing_grid.nxs"
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["definition"] = "NXtomo"
        detector = entry.create_group("instrument").create_group("detector")
        detector.create_dataset("data", data=np.zeros((2, 3, 4), dtype=np.float32))
        detector.create_dataset("image_key", data=np.zeros((2,), dtype=np.int32))
        sample = entry.create_group("sample")
        sample.create_dataset("name", data="fixture")
        transformations = sample.create_group("transformations")
        rotation_angle = transformations.create_dataset(
            "rotation_angle",
            data=np.asarray([0.0, 90.0], dtype=np.float32),
        )
        rotation_angle.attrs["units"] = "degree"
        tomojax = entry.create_group("processing").create_group("tomojax")
        tomojax.create_dataset("volume", data=np.zeros((5, 6, 7), dtype=np.float32))

    loaded = load_nxtomo(str(path))

    assert loaded.volume is not None
    assert loaded.grid is None
    assert loaded.disk_volume_axes_order == "unknown"
    assert loaded.volume_axes_source == "heuristic"


def test_load_tiff_stack_requires_explicit_angles_and_sorts_files(tmp_path: Path) -> None:
    stack_dir = tmp_path / "tiffs"
    stack_dir.mkdir()
    iio.imwrite(stack_dir / "0002.tif", np.full((2, 3), 2.0, dtype=np.float32))
    iio.imwrite(stack_dir / "0001.tif", np.full((2, 3), 1.0, dtype=np.float32))

    dataset = load_tiff_stack(stack_dir, angles_deg=[0.0, 90.0])

    assert dataset.source_format == "tiff_stack"
    assert dataset.projections.shape == (2, 2, 3)
    np.testing.assert_allclose(dataset.projections[:, 0, 0], [1.0, 2.0])
    np.testing.assert_allclose(dataset.angles_deg, [0.0, 90.0])
    with pytest.raises(ValueError, match="TIFF inputs require angle metadata"):
        load_dataset(stack_dir)
    with pytest.raises(ValueError, match="does not match projection count"):
        load_tiff_stack(stack_dir, angles_deg=[0.0])


def test_validate_dataset_reports_schema_failures(tmp_path: Path) -> None:
    valid = tmp_path / "valid.nxs"
    invalid = tmp_path / "invalid.nxs"
    write_projection_dataset(valid)
    with h5py.File(invalid, "w") as handle:
        handle.create_group("not_entry")

    assert validate_dataset(valid)["issues"] == []
    assert validate_dataset(invalid)["issues"] == ["Missing /entry"]


def test_validate_dataset_reports_scalar_axis_datasets(tmp_path: Path) -> None:
    path = tmp_path / "scalar_axes.nxs"
    write_projection_dataset(path)
    with h5py.File(path, "a") as handle:
        detector = handle["entry/instrument/detector"]
        del detector["image_key"]
        detector.create_dataset("image_key", data=np.asarray(0, dtype=np.int32))
        transformations = handle["entry/sample/transformations"]
        del transformations["rotation_angle"]
        rotation_angle = transformations.create_dataset(
            "rotation_angle",
            data=np.asarray(0.0, dtype=np.float32),
        )
        rotation_angle.attrs["units"] = "degree"

    issues = validate_dataset(path)["issues"]

    assert "instrument/detector/image_key must be 1D (n_views,)" in issues
    assert "rotation_angle must be 1D (n_views,)" in issues


def test_validate_dataset_reports_invalid_image_key_values(tmp_path: Path) -> None:
    path = tmp_path / "invalid_image_key.nxs"
    write_projection_dataset(path)
    with h5py.File(path, "a") as handle:
        detector = handle["entry/instrument/detector"]
        del detector["image_key"]
        detector.create_dataset("image_key", data=np.asarray([0, 99], dtype=np.int32))

    issues = validate_dataset(path)["issues"]

    assert "image_key values must be in {0, 1, 2, 3}; found 99 (1 frame)" in issues


def test_contrast_helpers_roundtrip_flat_dark_absorption() -> None:
    projections = np.asarray([[[5.0]], [[9.0]]], dtype=np.float32)
    flats = np.asarray([[[11.0]]], dtype=np.float32)
    darks = np.asarray([[[1.0]]], dtype=np.float32)

    absorption = flat_dark_to_absorption(projections, flats, darks)
    transmission = absorption_to_transmission(absorption)

    np.testing.assert_allclose(absorption[:, 0, 0], -np.log([0.4, 0.8]), rtol=1e-6)
    np.testing.assert_allclose(transmission[:, 0, 0], [0.4, 0.8], rtol=1e-6)


def _write_real_lamino_fixture(path: Path) -> tuple[np.ndarray, np.ndarray]:
    projections = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    thetas = np.asarray([10.0, 20.0], dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("entry/imaging/data", data=projections)
        handle.create_dataset("entry/imaging_sum/smaract_zrot", data=thetas)
    return projections, thetas


def test_real_laminography_loader_reads_angles_and_detector_orientation(tmp_path: Path) -> None:
    path = tmp_path / "real_lamino.nxs"
    projections, thetas = _write_real_lamino_fixture(path)

    loaded = load_real_laminography_input(path)
    transposed = load_real_laminography_input(path, transpose_detector=True)
    flipped_u = load_real_laminography_input(path, flip_u=True)
    flipped_v = load_real_laminography_input(path, flip_v=True)

    np.testing.assert_array_equal(loaded.projections, projections)
    np.testing.assert_array_equal(loaded.thetas_deg, thetas)
    np.testing.assert_array_equal(transposed.projections, np.transpose(projections, (0, 2, 1)))
    np.testing.assert_array_equal(flipped_u.projections, projections[:, :, ::-1])
    np.testing.assert_array_equal(flipped_v.projections, projections[:, ::-1, :])


def test_real_laminography_loader_rejects_angle_count_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "bad_real_lamino.nxs"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("entry/imaging/data", data=np.zeros((2, 3, 4), dtype=np.float32))
        handle.create_dataset(
            "entry/imaging_sum/smaract_zrot", data=np.zeros((3,), dtype=np.float32)
        )

    with pytest.raises(ValueError, match="projection count 2 does not match angle count 3"):
        load_real_laminography_input(path)
