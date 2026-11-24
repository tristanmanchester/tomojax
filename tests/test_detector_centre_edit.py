"""``--detector-center`` moves every view's detector by one amount and keeps everything else."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import tomojax as tj
from tomojax.cli.main import main
from tomojax.geometry import ConeSegments

from .test_zeiss import write_ole

pytestmark = pytest.mark.surface

_GRID = tj.Grid(8, 8, 6, 0.5, 0.5, 0.5)
_ANGLES = np.linspace(0.0, 180.0, 8, endpoint=False)


def _centred(tmp_path, source: tj.Scan, *options: str) -> tj.Scan:
    """Save ``source``, run ``correct --detector-center`` on it, and reload the output plainly."""
    path, out = tmp_path / "scan.nxs", tmp_path / "centred.nxs"
    tj.save(path, source)
    assert main(["correct", str(path), "-o", str(out), "--progress", "none", *options]) == 0
    return tj.load(out)


def test_the_centre_lands_where_asked_on_unequal_pitches(tmp_path):
    detector = tj.Detector(10, 6, 0.3, 0.2, (0.1, 0.5))
    source = tj.Scan(np.ones((8, 6, 10), np.float32), tj.ParallelGeometry(_GRID, detector, _ANGLES))
    centred = _centred(tmp_path, source, "--detector-center", "0.75", "0.5")
    assert centred.detector.center == pytest.approx((0.75, 0.5))
    assert (centred.detector.du, centred.detector.dv) == (0.3, 0.2)


def _segments(beams=None) -> ConeSegments:
    detector = tj.Detector(10, 6, 1.0, 1.0)
    beams = beams or (tj.ConeBeam(40.0, 80.0), tj.ConeBeam(40.0, 80.0))
    return ConeSegments(
        tuple(
            tj.ConeGeometry(_GRID, replace(detector, center=centre), list(angles), beam)
            for centre, angles, beam in zip(
                ((1.0, 2.0), (2.0, 3.0)), (_ANGLES[:4], _ANGLES[4:]), beams, strict=True
            )
        ),
        trajectory_lengths=(4, 4),
    )


def _centres(scan: tj.Scan) -> list[tuple[float, float]]:
    return [part.detector.center for part in scan.geometry.segments]


def test_segments_move_together_and_keep_their_arrangement(tmp_path):
    beams = (tj.ConeBeam(40.0, 80.0), tj.ConeBeam(40.0, 80.0, source_height=5.0))
    geometry = _segments(beams)
    source = tj.Scan(np.ones((8, 6, 10), np.float32), geometry)
    centred = _centred(tmp_path, source, "--detector-center", "1.75", "2.0")
    assert _centres(centred) == pytest.approx([(1.75, 2.0), (2.75, 3.0)])
    assert centred.geometry.trajectory_lengths == (4, 4)
    heights = [part.beam.source_height for part in centred.geometry.segments]
    assert heights == [0.0, 5.0]
    assert (
        tj.reconstruct(centred, "auto", iterations=None).method
        == tj.reconstruct(tj.load(tmp_path / "scan.nxs"), "auto").method
    )


def test_a_view_selection_keeps_its_acquisition_positions(tmp_path):
    detector = tj.Detector(10, 6, 1.0, 1.0)
    angles = np.linspace(0.0, 180.0, 12, endpoint=False)
    whole = tj.Scan(np.ones((12, 6, 10), np.float32), tj.ParallelGeometry(_GRID, detector, angles))
    selected = whole.selected([2, 7, 11])
    centred = _centred(tmp_path, selected, "--detector-center", "0.5", "0")
    np.testing.assert_array_equal(centred.view_positions, [2.5, 7.5, 11.5])
    assert centred.detector.center == (0.5, 0.0)


def test_a_raw_zeiss_scan_keeps_its_per_view_shifts(tmp_path):
    counts = (500 + np.arange(4 * 8 * 12)).astype("<u2").reshape(4, 8, 12)
    fields = {
        "ImageInfo/NoOfImages": ("<u4", [4]),
        "ImageInfo/ImageWidth": ("<u4", [12]),
        "ImageInfo/ImageHeight": ("<u4", [8]),
        "ImageInfo/DataType": ("<u4", [5]),
        "ImageInfo/AcquisitionMode": ("<u4", [0]),
        "ImageInfo/XrayGeometry": ("<u4", [1]),
        "ImageInfo/pixelsize": ("<f4", [10]),
        "ImageInfo/Angles": ("<f4", [0, 45, 90, 135]),
        "ImageInfo/StoRADistance": ("<f4", [-40] * 4),
        "ImageInfo/DtoRADistance": ("<f4", [20] * 4),
        "alignment/x-shifts": ("<f4", [0.25, 0.5, -0.75, 1.5]),
        "alignment/y-shifts": ("<f4", [-0.5, 0.25, 0.0, 1.0]),
        "AutoRecon/CenterShift": ("<f4", [1.25]),
        "ReferenceData/DataType": ("<u4", [10]),
    }
    streams = {key: np.asarray(values, dtype).tobytes() for key, (dtype, values) in fields.items()}
    streams.update({f"ImageData1/Image{i + 1}": value.tobytes() for i, value in enumerate(counts)})
    streams["ReferenceData/Image"] = np.full((8, 12), 2500, "<f4").tobytes()
    raw, plain_out, out = tmp_path / "raw.txrm", tmp_path / "plain.nxs", tmp_path / "centred.nxs"
    write_ole(raw, streams)
    assert main(["correct", str(raw), "-o", str(plain_out), "--darks", "0"]) == 0
    plain = _centres(tj.load(plain_out))
    argv = ["correct", str(raw), "-o", str(out), "--darks", "0", "--detector-center", "0.1", "-0.2"]
    assert main(argv) == 0
    moved = _centres(tj.load(out))
    assert moved[0] == pytest.approx((0.1, -0.2))
    delta = np.subtract(moved[0], plain[0])
    np.testing.assert_allclose(
        np.subtract(moved, plain), np.broadcast_to(delta, (4, 2)), atol=1e-12
    )


def test_the_centre_applies_after_a_new_pixel_size(tmp_path):
    detector = tj.Detector(10, 6, 1.0, 1.0, (0.5, 0.0))
    source = tj.Scan(np.ones((8, 6, 10), np.float32), tj.ParallelGeometry(_GRID, detector, _ANGLES))
    centred = _centred(tmp_path, source, "--pixel-size", "0.25", "--detector-center", "0.3", "0.1")
    assert (centred.detector.du, centred.detector.center) == (0.25, pytest.approx((0.3, 0.1)))


def test_a_set_centre_is_recorded_as_provided(tmp_path, capsys):
    from tomojax.io import ProjectionDataset

    detector = tj.Detector(10, 6, 1.0, 1.0)
    geometry = tj.ParallelGeometry(_GRID, detector, _ANGLES)
    meta = {"detector_center_status": "not_recorded"}
    record = ProjectionDataset(np.ones((8, 6, 10), np.float32), _ANGLES, geometry_metadata=meta)
    source = tj.Scan(np.ones((8, 6, 10), np.float32), geometry, source=record)
    centred = _centred(tmp_path, source, "--detector-center", "0.5", "0")
    assert centred.source.geometry_metadata["detector_center_status"] == "provided"
    argv = ["reconstruct", str(tmp_path / "centred.nxs"), "-o", str(tmp_path / "r.nxs")]
    assert main([*argv, "--progress", "none"]) == 0
    assert "records no rotation centre" not in capsys.readouterr().err


def test_a_calibration_patch_moves_every_segment_detector():
    from tomojax.geometry import Detector
    from tomojax.geometry.api import (
        CalibrationState,
        CalibrationVariable,
        build_calibrated_geometry_metadata_patch,
    )

    segments = [
        {"views": 2, "detector": Detector(10, 6, 0.5, 0.25, (1.0, 2.0)).to_dict()},
        {"views": 2, "detector": Detector(10, 6, 0.5, 0.25, (2.0, 3.0)).to_dict()},
    ]
    variables = tuple(
        CalibrationVariable(name=name, value=value, unit="px", status="estimated", frame="detector")
        for name, value in (("det_u_px", 2.0), ("det_v_px", -4.0))
    )
    patch = build_calibrated_geometry_metadata_patch(
        calibration_state=CalibrationState(detector=variables),
        detector=Detector(10, 6, 0.5, 0.25, (1.0, 2.0)).to_dict(),
        geometry_meta={"cone_segments": segments},
    )
    moved = [entry["detector"]["det_center"] for entry in patch["geometry_meta"]["cone_segments"]]
    assert moved == [[2.0, 1.0], [3.0, 2.0]]
    assert patch["detector"]["det_center"] == [2.0, 1.0]
