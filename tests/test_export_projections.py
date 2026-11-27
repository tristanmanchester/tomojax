"""Use detector coordinates and measured angles to check exported scan registration."""

from __future__ import annotations

from dataclasses import replace

import h5py
import numpy as np
import pytest
import tifffile

import tomojax as tj
from tomojax.io import ProjectionDataset, build_geometry_from_dataset_metadata


def measured():
    det = tj.Detector(8, 6, 0.4, 0.5, center=(0.3, -0.2))
    angles = np.array([-15.0, 0.0001, 24.0, 67.0, 95.0], np.float64)
    geom = tj.ParallelGeometry(tj.Grid(7, 5, 3, 1, 1, 2), det, angles, detector_roll=11)
    view, v, u = np.indices((5, 6, 8))
    data = (100 * view + 10 * v + u).astype("f4")
    source = ProjectionDataset(data, angles, geometry_metadata={"length_unit": "mm"})
    scan = tj.Scan(data, geom, source=source, view_positions=np.array([3.5, 4.5, 6.5, 8.5, 9.5]))
    poses = np.zeros((5, 6))
    poses[:, 3] = [0, 0.2, 0.1, 0, -0.4]
    return scan.with_poses(poses)


def test_sinogram_pages_keep_the_actual_view_rows_without_physical_page_spacing(tmp_path):
    scan = measured()
    path = tmp_path / "sinograms.tif"
    receipt = tj.export(path, scan, axis="v")
    assert [a["name"] for a in receipt["axes"]] == ["v", "view", "u"]
    with tifffile.TiffFile(path) as file:
        assert file.imagej_metadata["unit"] == "pixel"
        assert "spacing" not in file.imagej_metadata
        for v, page in enumerate(file.pages):
            np.testing.assert_array_equal(page.asarray(), scan.projections[:, v, :])
            assert page.tags["ResolutionUnit"].value == 1
    for suffix in (".am", ".zarr"):
        with pytest.raises(tj.InvalidParameter, match="needs a spatial volume"):
            tj.export(tmp_path / ("cannot" + suffix), scan, dry_run=True)


def test_exported_cropped_views_rebuild_without_the_source(tmp_path):
    scan = measured()
    native = tmp_path / "aligned.nxs"
    tj.save(native, scan)
    path = tmp_path / "sinograms.h5"
    crop = ((1, 5), (1, 6), (0, 7))
    tj.export(path, tj.load(native), crop=crop, binning=2, axis="v")
    native.unlink()
    record = tj.inspect(path)["record"]
    tables = record["tables"]
    np.testing.assert_array_equal(tables["angles"]["values"], [0.0001, 24, 67, 95])
    np.testing.assert_array_equal(tables["view_positions"]["values"], [4.5, 6.5, 8.5, 9.5])
    np.testing.assert_array_equal(tables["poses"]["values"], scan.poses[1:])
    with h5py.File(path) as file:
        group = file["entry/data"]
        assert list(group.attrs["axes"]) == ["v", "rotation_angle", "u"]
        np.testing.assert_array_equal(group["rotation_angle"][:], tables["angles"]["values"])
        np.testing.assert_allclose(group["u"][:], [-0.9, -0.1, 0.7])
        np.testing.assert_allclose(group["v"][:], [-0.7, 0.3])
        pixels = np.moveaxis(group["data"][:], 1, 0)
    g = record["geometry"]
    _, _, geometry = build_geometry_from_dataset_metadata(
        {
            **g["meta"],
            "geometry_type": g["type"],
            "grid": g["grid"],
            "detector": g["detector"],
            "angles": tables["angles"]["values"],
            "align_params": tables["poses"]["values"],
            "align_gauge": {"pose_translation_frame": tables["poses"]["frame"]},
        },
        apply_poses=True,
    )
    rebuilt = tj.Scan(pixels, geometry, view_positions=np.array(tables["view_positions"]["values"]))
    expected = scan.selected(slice(1, 5)).cropped(slice(1, 6), slice(0, 7)).binned(2)
    np.testing.assert_array_equal(rebuilt.projections, expected.projections)
    for i in range(4):
        np.testing.assert_array_equal(
            rebuilt.geometry.pose_matrix(i), expected.geometry.pose_matrix(i)
        )
    assert rebuilt.detector == expected.detector
    np.testing.assert_array_equal(rebuilt.view_positions, expected.view_positions)


def test_lazy_stack_reads_only_selected_detector_windows(tmp_path, monkeypatch):
    scan = measured()
    reads = []

    from tomojax.io.api import locate_frames

    acquisition = tmp_path / "measured.nxs"
    tj.save(acquisition, scan)
    source = locate_frames(str(acquisition), data_path="/entry/data/data").stack
    read = type(source)._read

    def window(stack, indices, rows, cols):
        reads.append((indices.tolist(), rows, cols))
        return read(stack, indices, rows, cols)

    def whole(*args, **kwargs):
        raise AssertionError("materialized detector stack")

    monkeypatch.setattr(type(source), "_read", window)
    monkeypatch.setattr(type(source), "__array__", whole)
    path = tmp_path / "sinograms.raw"
    tj.export(path, replace(scan, projections=source), crop=((1, 4), (1, 3), (2, 8)), axis="v")
    np.testing.assert_array_equal(
        np.fromfile(path, "<f4").reshape(2, 3, 6),
        scan.projections[1:4, 1:3, 2:8].transpose(1, 0, 2),
    )
    assert reads == [([1, 2, 3], slice(1, 3), slice(2, 8))]
