"""``correct --preview``: a row band corrected exactly as the full run corrects it."""

from __future__ import annotations

import json

import numpy as np
import pytest

import tomojax as tj
from tomojax.cli.main import main
from tomojax.corrections import Paganin, RejectViews, Rings, Stripes, Zingers
from tomojax.interactive import correction_preview

pytestmark = pytest.mark.surface

_VIEWS = 24


def _frames():
    grid = tj.Grid(12, 12, 10, 1, 1, 1)
    detector = tj.Detector(12, 10, 1, 1)
    geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, _VIEWS, endpoint=False))
    rng = np.random.default_rng(7)
    counts = rng.uniform(400, 900, (_VIEWS, 10, 12)).astype(np.float32)
    counts[:, :, 5] *= 0.7  # a dim column: a ring
    counts[9] *= 0.2  # a dark view to reject
    counts[3, 4, 6] = 5000  # a zinger
    flats = np.full((1, 10, 12), 1000, np.float32)
    return tj.Frames(counts, geometry, flats=flats, darks=np.zeros_like(flats))


_RAW_STEPS = [
    (),
    (Zingers(threshold=0.1),),
    (Paganin(100.0, 0.1, 20.0, 1e-6),),
    (Stripes(width=5),),
    (Rings(),),
    (RejectViews(z=3.0),),
    (Zingers(threshold=0.1), RejectViews(z=3.0), Stripes(width=5)),
]


@pytest.mark.parametrize("steps", _RAW_STEPS, ids=lambda s: "+".join(type(x).__name__ for x in s))
@pytest.mark.parametrize("binning", [1, 2])
@pytest.mark.parametrize("trim", [False, True], ids=["whole", "selected+cropped"])
def test_a_raw_band_equals_the_same_rows_of_the_full_correction(steps, binning, trim):
    frames = _frames()
    if trim:
        frames = frames.selected([0, 2, 3, 5, 8, 9, 13, 17, 21, 23]).cropped(
            slice(1, 9), slice(0, 12)
        )
    rows = (1, 3)
    band = correction_preview(frames, rows, *steps, binning=binning)
    full = frames.corrected(*steps, binning=binning)
    np.testing.assert_allclose(
        band.after, full.projections[:, rows[0] : rows[1]], rtol=1e-6, atol=1e-6
    )
    assert band.domain == "counts"


_SCAN_STEPS = [
    (Stripes(width=5),),
    (Rings(),),
    (RejectViews(z=3.0),),
    (RejectViews(z=3.0), Stripes(width=5)),
]


@pytest.mark.parametrize("steps", _SCAN_STEPS, ids=lambda s: "+".join(type(x).__name__ for x in s))
@pytest.mark.parametrize("binning", [1, 2])
def test_a_scan_band_equals_the_same_rows_corrected_then_binned(steps, binning):
    scan = _frames().corrected()
    rows = (1, 3)
    band = correction_preview(scan, rows, *steps, binning=binning)
    full = scan.corrected(*steps).binned(binning)
    np.testing.assert_allclose(
        band.after, full.projections[:, rows[0] : rows[1]], rtol=1e-6, atol=1e-6
    )
    np.testing.assert_array_equal(
        band.before, np.asarray(scan.projections)[:, rows[0] * binning : rows[1] * binning]
    )
    assert band.domain == "line_integrals"


def test_a_view_dropped_inside_a_selection_is_named_by_its_source_number():
    scan = _frames().corrected()
    selection = np.array([v for v in range(_VIEWS) if v % 4 != 3])  # view 9 is the dark one
    band = correction_preview(scan.selected(selection), (1, 3), RejectViews(z=3.0))
    dropped = sorted(set(range(len(selection))) - set(band.views.tolist()))
    assert selection[dropped].tolist() == [9]


@pytest.fixture
def scan_path(tmp_path):
    path = tmp_path / "scan.nxs"
    tj.save(path, _frames().corrected())
    return path


def test_a_dry_run_writes_no_scan_and_bounds_what_rejection_may_drop(tmp_path, scan_path, capsys):
    out = tmp_path / "out.nxs"
    argv = ["correct", str(scan_path), "-o", str(out), "--reject-views", "--dry-run"]
    assert main(argv) == 0
    text = capsys.readouterr().out
    assert not out.exists()
    assert "at most 24 views of 10 x 12 pixels" in text and "read every detector row" in text
    assert main(["correct", str(scan_path), "--stripes", "5", "--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "24 views" in text and "at most" not in text and "read only its band" in text


@pytest.mark.parametrize("preview", [False, True])
def test_a_json_dry_run_ends_with_its_plan(tmp_path, scan_path, capsys, preview):
    png = tmp_path / "band.png"
    argv = ["correct", str(scan_path), "--stripes", "5", "--dry-run", "--progress", "json"]
    assert main([*argv, *(["--preview", str(png)] if preview else [])]) == 0
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    summary = result["info"]["summary"]
    assert summary["shape"] == [24, 10, 12] and summary["bytes"] == 4 * 24 * 10 * 12
    assert summary["steps"] == ["stripes(width=5, varying=False)"]
    assert summary["upper_bound"] is False and summary["full_rows"] is False
    assert result["outputs"] == ([str(png)] if preview else [])
    assert png.exists() is preview


def test_a_full_run_writes_the_panel_too(tmp_path, scan_path):
    png, out = tmp_path / "band.png", tmp_path / "out.nxs"
    argv = ["correct", str(scan_path), "-o", str(out), "--stripes", "5", "--preview", str(png)]
    assert main([*argv, "--views", "0:24:2", "--preview-rows", "2:4", "--progress", "none"]) == 0
    assert out.exists() and png.exists()


def test_the_default_band_fits_a_one_row_output(tmp_path, capsys):
    path = tmp_path / "thin.nxs"
    tj.save(path, _frames().corrected().cropped(slice(4, 5), slice(None)))
    assert main(["correct", str(path), "--dry-run", "--preview", str(tmp_path / "b.png")]) == 0
    assert "output rows 0:1" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("rows", "message"),
    [("3:2", "B > A >= 0"), ("a:b", "B > A >= 0"), ("0:99", "past the 10 output rows")],
)
def test_bad_preview_rows_are_usage_errors(tmp_path, scan_path, capsys, rows, message):
    with pytest.raises(SystemExit) as error:
        main(
            [
                "correct",
                str(scan_path),
                "--dry-run",
                "--preview",
                str(tmp_path / "b.png"),
                "--preview-rows",
                rows,
            ]
        )
    assert error.value.code == 2 and message in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["correct", str(scan_path), "--dry-run", "--preview-rows", "0:2"])  # no --preview
