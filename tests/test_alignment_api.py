"""The public fit selection and calibrated scan survive the workflow and files."""

import inspect
import json

import jax
import numpy as np
import pytest

import tomojax as tj

# check-public-imports: allow-private
from tomojax._workflow import smooth_alignment as _smooth_alignment
from tomojax.cli.main import main
from tomojax.geometry import stack_view_poses

pytestmark = pytest.mark.surface


def _scan(cone=False):
    grid, detector = tj.Grid(24, 24, 24, 2, 2, 2), tj.Detector(48, 48, 1, 1, (0.2, -0.3))
    angles = np.linspace(0, 360, 60, endpoint=False)
    geometry = (
        tj.ConeGeometry(grid, detector, angles, tj.ConeBeam(192, 288))
        if cone
        else tj.ParallelGeometry(grid, detector, angles)
    )
    return tj.Scan(np.zeros((60, 48, 48), np.float32), geometry)


def _fitter(monkeypatch, *, changes=None, poses=None):
    calls = []

    def fit(projections, nominal, grid, detector, **options):
        calls.append((projections, np.asarray(nominal), grid, detector, options))
        names = options["fit"]
        names = (names,) if isinstance(names, str) else names
        local = {"translations", "rotations"} & set(names)
        table = poses if poses is not None else np.zeros((len(nominal), 6)) if local else None
        record = {
            "fit": list(names),
            "calibration": changes or {},
            "report": {"isolated_views": [], "unusual_views": [], "advisories": []},
            "seconds": 0.0,
        }
        return np.zeros((grid.nz, grid.ny, grid.nx), np.float32), grid, table, record

    monkeypatch.setattr(_smooth_alignment, "fit_poses", fit)
    return calls


def test_alignment_has_one_fit_argument_and_an_optional_grid():
    parameters = inspect.signature(tj.align).parameters
    assert tuple(parameters) == (
        "scan",
        "fit",
        "grid",
        "region",
        "method",
        "motion",
        *("iterations", "image_iterations", "tolerance", "interpolation", "resolution"),
        "config",
        "checkpoint",
    )
    assert parameters["fit"].default is None  # the method's own fit
    assert parameters["fit"].kind == inspect.Parameter.KEYWORD_ONLY
    assert parameters["grid"].default is None


@pytest.mark.parametrize("fit", [(), [], "unknown", ("roll", "roll"), ("center", 1)])
def test_invalid_fit_fails_before_a_solve(fit):
    with pytest.raises(ValueError, match="fit"):
        tj.align(_scan(), fit=fit)


@pytest.mark.parametrize("cone", [False, True])
@pytest.mark.parametrize("name", ["center", "tilt", "roll"])
def test_global_fits_publish_the_named_calibration_and_hold_local_poses(monkeypatch, cone, name):
    scan = _scan(cone)
    correction = {"center": (1.25, -0.75) if cone else 1.25, "tilt": 0.5, "roll": 0.75}[name]
    calls = _fitter(monkeypatch, changes={name: correction})
    result = tj.align(scan, fit=name)
    assert calls[0][4]["fit"] == (name,)
    assert result.scan.projections is scan.projections
    assert set(result.info["calibration"]) == {name}
    assert result.scan.poses is None
    np.testing.assert_array_equal(result.poses, 0)
    expected = {"center": (1.45, -1.05) if cone else 1.45, "tilt": 0.5, "roll": 0.75}[name]
    np.testing.assert_allclose(result.info["calibration"][name]["value"], expected, atol=2e-6)
    assert result.info["calibration"][name]["change"] == correction
    if name != "center":
        assert result.scan.detector.center == scan.detector.center


def test_a_center_and_roll_fit_keeps_fourier_as_the_automatic_method(monkeypatch):
    scan = _scan()
    _fitter(monkeypatch, changes={"center": 1.25, "roll": 0.75})
    result = tj.align(scan, fit=("center", "roll"))
    automatic = tj.reconstruct(result.scan).info["automatic"]
    assert automatic["method"] == "fourier" and automatic["status"] == "ok"


def test_default_fit_preserves_existing_rotations_and_uses_the_requested_grid(monkeypatch):
    scan = _scan()
    previous = np.zeros((60, 6))
    previous[:, :3] = [0.01, -0.02, 0.03]
    previous[:, 3:5] = [0.2, -0.1]
    scan = scan.with_poses(previous)
    correction = np.zeros_like(previous)
    correction[:, 3:5] = [0.07, 0.04]
    calls = _fitter(monkeypatch, poses=correction)
    grid = tj.Grid(20, 18, 16, 2, 2, 2)
    result = tj.align(scan, grid=grid)
    np.testing.assert_allclose(calls[0][1], stack_view_poses(scan.geometry, 60), atol=1e-7)
    assert calls[0][4]["fit"] == ("translations",) and calls[0][4]["keep_grid"]
    assert result.scan.grid == grid and result.image.shape == (16, 18, 20)
    np.testing.assert_array_equal(result.poses[:, :3], scan.poses[:, :3])
    np.testing.assert_allclose(result.poses[:, 3:], previous[:, 3:] + correction[:, 3:], atol=1e-7)
    assert result.info["calibration"] == {}


@pytest.mark.parametrize("cone", [False, True])
@pytest.mark.parametrize("suffix", [".npz", ".nxs"])
def test_fitted_global_geometry_round_trips_with_held_local_poses(
    monkeypatch, tmp_path, cone, suffix
):
    changes = {"center": (1.25, -0.75) if cone else 1.25, "tilt": 0.5, "roll": 0.75}
    _fitter(monkeypatch, changes=changes)
    result = tj.align(_scan(cone), fit=("center", "tilt", "roll"))
    target = tmp_path / f"aligned{suffix}"
    tj.save(target, result)
    loaded = tj.load(target)
    assert loaded.detector.center == result.scan.detector.center
    np.testing.assert_array_equal(loaded.projections, result.scan.projections)
    assert loaded.poses is None
    np.testing.assert_allclose(
        stack_view_poses(loaded.geometry, 60), stack_view_poses(result.scan.geometry, 60), atol=2e-6
    )
    from bench.fit_alignment import scan_rays

    rays, expected = scan_rays(loaded), scan_rays(result.scan)
    points = np.array([[0, 0, 0], [-9, 6, 15], [7, -11, -13]])
    np.testing.assert_allclose(rays.pixels(points), expected.pixels(points), atol=2e-5)


def test_cli_fit_forwards_one_selection_and_saves_its_calibration(monkeypatch, tmp_path):
    scan, output, manifest = tmp_path / "scan.nxs", tmp_path / "aligned.nxs", tmp_path / "run.json"
    tj.save(scan, _scan())
    calls = _fitter(monkeypatch, changes={"tilt": 0.5, "roll": 0.75})
    assert (
        main(
            [
                "align",
                str(scan),
                "-o",
                str(output),
                "--fit",
                "translations",
                "tilt",
                "roll",
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )
    assert calls[0][4]["fit"] == ("tilt", "roll", "translations")  # stable order
    saved = json.loads(manifest.read_text())["resolved_config"]
    assert saved["fit"] == ["translations", "tilt", "roll"]
    values = {name: entry["value"] for name, entry in saved["alignment"]["calibration"].items()}
    assert values == pytest.approx({"tilt": 0.5, "roll": 0.75})
    np.testing.assert_array_equal(tj.load(output).poses, 0)


def test_cli_dry_run_reports_the_selection_without_fitting(monkeypatch, tmp_path, capsys):
    scan, output = tmp_path / "scan.nxs", tmp_path / "aligned.nxs"
    tj.save(scan, _scan())
    calls = _fitter(monkeypatch)
    argv = ["align", str(scan), "-o", str(output), "--fit", "center", "--dry-run"]
    assert main([*argv, "--progress", "json"]) == 0
    record = json.loads(capsys.readouterr().out.splitlines()[-1])["info"]["summary"]
    assert record["fit"] == ["center"] and record["bounded_working_grid"]
    assert not calls and not output.exists()


@pytest.mark.parametrize("fit", [["unknown"], ["roll", "roll"]])
def test_cli_rejects_unknown_or_repeated_fit_names(fit, capsys):
    with pytest.raises(SystemExit) as error:
        main(["align", "scan.nxs", "-o", "aligned.nxs", "--fit", *fit])
    assert error.value.code == 2
    assert "--fit" in capsys.readouterr().err


def test_running_out_of_memory_on_an_explicit_grid_says_how_to_bound_it(monkeypatch):
    def exhausted(*args, **kwargs):
        raise jax.errors.JaxRuntimeError(
            "RESOURCE_EXHAUSTED: Out of memory while trying to allocate 789.17MiB"
        )

    monkeypatch.setattr(_smooth_alignment, "fit_poses", exhausted)
    scan = _scan()
    with pytest.raises(MemoryError, match="used as given.*Omit grid= to let the fit bound its own"):
        tj.align(scan, grid=scan.grid)
    # Without an explicit grid the fit already chose its scale: the device error is unchanged.
    with pytest.raises(jax.errors.JaxRuntimeError, match="RESOURCE_EXHAUSTED"):
        tj.align(scan)
