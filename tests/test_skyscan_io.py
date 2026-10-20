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
    np.testing.assert_allclose(frames.geometry.angles, [0, 0.2])
    # A content-recognized drift-reference log must not hide the sample stack.
    (tmp_path / "reference.metadata").write_text(path.read_text().replace("Files=3", "Files=4"))
    np.testing.assert_allclose(
        tj.load_frames(tmp_path, detector=detector, flats=65536, darks=0).geometry.angles,
        [0, 0.2],
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
