"""SkyScan content discovery, metadata geometry and lazy corrected TIFF ingestion."""

from pathlib import Path

import numpy as np
import pytest
import tifffile

import tomojax as tj
from tomojax.geometry import ConeGeometry


def test_skyscan_log_drives_geometry_and_selects_only_sample_tiffs(tmp_path: Path) -> None:
    counts = (20000 + np.arange(2 * 6 * 8).reshape(2, 6, 8)).astype(np.uint16)
    for view, frame in enumerate(counts):
        tifffile.imwrite(tmp_path / f"grain{view:04d}.tif", frame)
    tifffile.imwrite(tmp_path / "grain_arc.tif", np.zeros((1, 2), np.uint16))
    tifffile.imwrite(tmp_path / "grain0000.iif", np.zeros((1, 1), np.uint8))
    path = tmp_path / "renamed.metadata"
    path.write_text(
        "[System]\nScanner=Skyscan1172\nCamera Pixel Size (um)=20\nCameraXYRatio=.998\n"
        "[Acquisition]\nFilename Prefix=grain\nNumber of Files=3\nNumber of Rows=6\n"
        "Number of Columns=8\nCamera binning=2x2\nImage Pixel Size (um)=20\nImage Rotation=-.11\n"
        "Object to Source (mm)=20\nCamera to Source (mm)=40\nOptical Axis (line)=1.5\n"
        "Rotation Step (deg)=.2\nRotation Direction=CC\nDepth (bits)=16\n"
        "Geometrical Correction=ON\nFlat Field Correction=ON\nRandom Movement=ON (10)\n"
    )
    with pytest.raises(ValueError, match="detector="):
        tj.load_frames(path, flats=65536, darks=0)
    detector = tj.Detector(8, 6, 0.04, 0.04, (0, -0.06))
    frames = tj.load_frames(path, detector=detector, flats=65536, darks=0)
    assert frames.source is not None and frames.source.source_format == "skyscan"
    metadata = frames.source.geometry_metadata["skyscan"]
    assert metadata["System"]["cameraxyratio"] == ".998"
    assert metadata["Acquisition"]["image rotation"] == "-.11"
    assert metadata["Acquisition"]["random movement"] == "ON (10)"
    assert metadata["Acquisition"]["geometrical correction"] == "ON"
    assert isinstance(frames.geometry, ConeGeometry)
    assert frames.white_level == 65536
    assert frames.geometry.detector == tj.Detector(8, 6, 0.04, 0.04, center=(0, -0.06))
    assert frames.geometry.grid == tj.Grid(8, 8, 6, 0.02, 0.02, 0.02)
    assert frames.geometry.beam.source_to_axis == 20
    assert frames.geometry.beam.source_to_detector == 40
    np.testing.assert_allclose(frames.geometry.angles, [0, -0.2])
    # A content-recognized drift-reference log must not hide the sample stack.
    (tmp_path / "reference.metadata").write_text(path.read_text().replace("Files=3", "Files=4"))
    np.testing.assert_allclose(
        tj.load_frames(tmp_path, detector=detector, flats=65536, darks=0).geometry.angles,
        [0, -0.2],
    )
    cropped = frames.cropped(slice(1, 5), slice(2, 7)).selected([1])
    np.testing.assert_array_equal(cropped.counts[:], counts[1:2, ::-1, :][:, 1:5, 2:7])
    scan = frames.corrected()
    np.testing.assert_allclose(scan.projections, -np.log(counts[:, ::-1] / 65536), rtol=2e-6)

    assert frames.source.geometry_metadata["acquisition"]["resize"] is None
    path.write_text(path.read_text().replace("Rotation Direction=CC\n", ""))
    with pytest.raises(ValueError, match="angles="):
        tj.load_frames(path, detector=detector, flats=65536, darks=0)
    explicit = tj.load_frames(path, detector=detector, angles=[0, 0.2], flats=65536, darks=0)
    np.testing.assert_allclose(explicit.corrected().projections, scan.projections)


def test_the_logged_voxel_is_kept_and_the_grid_follows_binning(tmp_path: Path) -> None:
    # The log states the voxel (Image Pixel Size); the grid's shape comes from the detector.
    # Binning keeps the stated voxel's meaning: a quarter the samples, twice the voxel at 2.
    counts = (20000 + np.arange(2 * 6 * 8).reshape(2, 6, 8)).astype(np.uint16)
    for view, frame in enumerate(counts):
        tifffile.imwrite(tmp_path / f"grain{view:04d}.tif", frame)
    path = tmp_path / "scan.log"
    path.write_text(
        "[System]\nScanner=Skyscan1172\nCamera Pixel Size (um)=20\n"
        "[Acquisition]\nFilename Prefix=grain\nNumber of Files=2\nNumber of Rows=6\n"
        "Number of Columns=8\nCamera binning=2x2\nImage Pixel Size (um)=20\n"
        "Object to Source (mm)=20\nCamera to Source (mm)=40\nOptical Axis (line)=1.5\n"
        "Rotation Step (deg)=.2\nRotation Direction=CC\nDepth (bits)=16\n"
        "Geometrical Correction=ON\nFlat Field Correction=ON\n"
    )
    detector = tj.Detector(8, 6, 0.04, 0.04, (0, -0.06))
    frames = tj.load_frames(path, detector=detector, flats=65536, darks=0)
    assert frames.geometry.grid == tj.Grid(8, 8, 6, 0.02, 0.02, 0.02)  # as before
    meta = frames.source.geometry_metadata
    assert meta["grid_status"] == "derived" and meta["voxel_size"] == pytest.approx([0.02] * 3)
    binned = frames.corrected(binning=2).grid
    assert (binned.nx, binned.ny, binned.nz) == (4, 4, 3)
    assert (binned.vx, binned.vy, binned.vz) == pytest.approx((0.04, 0.04, 0.04))


@pytest.mark.parametrize(("direction", "row_after_turn"), [("CC", 2), ("CW", 5)])
def test_skyscan_quarter_turn_ray_passes_through_marker(tmp_path, direction, row_after_turn):
    # At zero degrees the marker (2, 0, 2) projects 4 px right and 4 px up.
    # A CC quarter turn brings it 2 mm nearer the source: magnification 12/4.
    # CW takes it 2 mm farther away: magnification 12/8. TIFF rows run down.
    for i, pixel in enumerate([(4, 12), (row_after_turn, 8)]):
        counts = np.full((17, 17), 60000, np.uint16)
        counts[pixel] = 30000
        tifffile.imwrite(tmp_path / f"marker{i:04d}.tif", counts)
    path = tmp_path / "marker.log"
    path.write_text(
        "[System]\nScanner=Skyscan1172\n[Acquisition]\nFilename Prefix=marker\n"
        "Number of Files=2\nNumber of Rows=17\nNumber of Columns=17\nCamera binning=1x1\n"
        "Image Pixel Size (um)=500\nObject to Source (mm)=6\nCamera to Source (mm)=12\n"
        f"Rotation Step (deg)=90\nRotation Direction={direction}\n"
        "Geometrical Correction=ON\nFlat Field Correction=ON\n"
    )
    frames = tj.load_frames(path, detector=tj.Detector(17, 17, 1, 1), flats=60000, darks=0)
    geometry = frames.geometry
    assert isinstance(geometry, ConeGeometry)
    for i, frame in enumerate(np.asarray(frames.corrected().projections)):
        v, u = np.unravel_index(np.argmax(frame), frame.shape)
        origin, ray = geometry.rays_for_view(i)
        marker = np.asarray(geometry.pose_matrix(i)) @ [2, 0, 2, 1]
        displacement = marker[:3] - origin(int(u), int(v))
        np.testing.assert_allclose(
            displacement / np.linalg.norm(displacement), ray(int(u), int(v)), atol=1e-12
        )
