"""tools/check_numerical_equivalence.py: ULP distances, the pose gauge, and the pass/fail verdicts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
import zipfile

import numpy as np
import pytest

import tomojax as tj

ROOT = Path(__file__).resolve().parents[1]


def _tool() -> ModuleType:
    name = "check_numerical_equivalence"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses resolve their annotations through it
        spec.loader.exec_module(module)
    return sys.modules[name]


tool = _tool()


def _steps(value: float, k: int) -> np.float32:
    """``value`` moved ``k`` representable float32 values towards +inf, one at a time."""
    x = np.float32(value)
    for _ in range(k):
        x = np.nextafter(x, np.float32(np.inf), dtype=np.float32)
    return x


@pytest.mark.surface
def test_ulp_distance_counts_representable_floats_between_values():
    a = np.array([1.0, -2.5, 0.0, 1e-3, 3.0], np.float32)
    b = np.array([_steps(1.0, 1), _steps(-2.5, 7), -0.0, _steps(1e-3, 300), 3.0], np.float32)
    np.testing.assert_array_equal(tool.ulp_distance(a, b), [1, 7, 0, 300, 0])
    tiny = np.float32(1e-45)  # the smallest subnormal: zero lies one step either side of it
    assert tool.ulp_distance(np.array([tiny]), np.array([-tiny]))[0] == 2
    # 1.0 is 0x3F800000 steps above zero, and -1.0 as many below it.
    assert tool.ulp_distance(np.array([1.0]), np.array([-1.0]))[0] == 2 * 0x3F800000


def _case(method="cgls"):
    return tool.Case("parallel", method, 8, 6, None if method in ("fbp", "align") else 5)


def _measured(case, error, outputs):
    metrics = {case.error_metric: list(error), "residual": [0.1] * len(error)}
    return tool.Measured(case, metrics, [np.asarray(o, np.float32) for o in outputs])


def _baseline(case, error, reference, spread=()):
    metrics = {case.error_metric: list(error)}
    return tool.BaselineCase(case, metrics, list(spread), np.asarray(reference, np.float32))


def _verdicts(current, baseline, tolerance=0.01):
    return {c.name: c.passed for c in tool.compare(current, baseline, tolerance=tolerance)}


@pytest.mark.surface
@pytest.mark.parametrize("method", ["cgls", "align"])
def test_an_error_beyond_the_tolerance_fails_and_one_within_passes(method):
    case, out = _case(method), np.ones(4)
    baseline = _baseline(case, [0.100, 0.098], out)
    assert _verdicts(_measured(case, [0.1009], [out]), baseline) == {"error": True, "ulp": True}
    assert _verdicts(_measured(case, [0.1011], [out]), baseline)["error"] is False
    assert _verdicts(_measured(case, [float("nan")], [out]), baseline)["error"] is False
    # The median repeat is judged: one unlucky repeat out of three does not fail, and one
    # lucky repeat does not pass.
    assert _verdicts(_measured(case, [0.09, 0.2, 0.095], [out] * 3), baseline)["error"] is True
    assert _verdicts(_measured(case, [0.09, 0.2, 0.15], [out] * 3), baseline)["error"] is False


@pytest.mark.surface
def test_without_measured_spread_a_single_ulp_fails():
    case, out = _case(), np.linspace(-1, 1, 10, dtype=np.float32)
    baseline = _baseline(case, [0.1], out)
    assert _verdicts(_measured(case, [0.1], [out]), baseline)["ulp"] is True
    nudged = out.copy()
    nudged[3] = _steps(nudged[3], 1)
    assert _verdicts(_measured(case, [0.1], [nudged]), baseline) == {"error": True, "ulp": False}


@pytest.mark.surface
def test_a_difference_within_the_baseline_spread_passes_and_beyond_it_fails():
    case = _case()
    reference = np.full(100, 0.5, np.float32)

    def moved(n_voxels, k):
        out = reference.copy()
        out[:n_voxels] = _steps(0.5, k)
        return out

    # The baseline's repeats sat up to 3 ULP off in a few voxels; the median voxel never moved.
    spread = [tool.UlpStats(0.0, 2), tool.UlpStats(0.0, 3)]
    baseline = _baseline(case, [0.1], reference, spread)
    assert _verdicts(_measured(case, [0.1], [moved(5, 3)]), baseline)["ulp"] is True
    assert _verdicts(_measured(case, [0.1], [moved(5, 4)]), baseline)["ulp"] is False
    # Every voxel moving by 1 ULP is a change, even though no single voxel left the spread.
    assert _verdicts(_measured(case, [0.1], [moved(100, 1)]), baseline)["ulp"] is False
    # One repeat inside the spread is enough: the others may be noise drawn further out.
    repeats = [moved(5, 9), moved(5, 2), moved(5, 9)]
    assert _verdicts(_measured(case, [0.1] * 3, repeats), baseline)["ulp"] is True


@pytest.mark.surface
def test_a_changed_output_shape_fails_the_ulp_check():
    case = _case()
    baseline = _baseline(case, [0.1], np.zeros((2, 3)))
    assert _verdicts(_measured(case, [0.1], [np.zeros((3, 2))]), baseline)["ulp"] is False


@pytest.mark.surface
def test_pose_error_removes_exactly_an_object_shift_in_parallel_beam():
    # Parallel beam: an object shift (x, y, z) moves view theta by a cos + b sin on u and by a
    # constant on v. Those three patterns are the gauge; what is left is the error.
    angles = np.linspace(0, 180, 23, endpoint=False) + 7.0
    geometry = tj.ParallelGeometry(tj.Grid(8, 8, 8, 1, 1, 1), tj.Detector(8, 8, 1, 1), angles)
    t = np.deg2rad(angles)
    rng = np.random.default_rng(5)
    true = np.zeros((len(t), 6))
    true[:, 3:5] = rng.normal(size=(len(t), 2))
    gauge_u = 0.7 * np.cos(t) - 0.4 * np.sin(t)
    estimated = true.copy()
    estimated[:, 3] += gauge_u
    estimated[:, 4] += 0.3
    assert tool.pose_error(estimated, true, geometry) < 1e-9
    noise = rng.normal(scale=0.1, size=(len(t), 2))
    estimated[:, 3:5] += noise
    u_basis = np.stack([np.cos(t), np.sin(t)], 1)
    u = noise[:, 0] - u_basis @ np.linalg.lstsq(u_basis, noise[:, 0], rcond=None)[0]
    v = noise[:, 1] - noise[:, 1].mean()
    expected = np.sqrt(np.mean(u**2 + v**2))
    np.testing.assert_allclose(tool.pose_error(estimated, true, geometry), expected, rtol=1e-9)


@pytest.mark.surface
@pytest.mark.parametrize("kind", ["cone", "lamino"])  # parallel: the test above
def test_a_shift_along_a_tilted_rotation_axis_is_gauge_but_one_moved_view_is_not(kind):
    grid, detector = tj.Grid(8, 8, 8, 1, 1, 1), tj.Detector(8, 8, 1, 1)
    angles = np.linspace(0, 360, 12, endpoint=False) + 7.0
    geometry = (
        tj.ConeGeometry(grid, detector, angles, tj.ConeBeam(40, 60), tilt=10.0)
        if kind == "cone"
        else tj.LaminographyGeometry(grid, detector, angles, tilt=30.0)
    )
    axis = np.asarray(geometry.rotation_axis())  # world (x, y, z): every view leaves it in place
    true = np.zeros((12, 6))
    estimated = true.copy()
    estimated[:, 3:5] = 0.8 * axis[[0, 2]]
    assert tool.pose_error(estimated, true, geometry) < 1e-6
    estimated[4, 3] += 1.0
    assert tool.pose_error(estimated, true, geometry) > 0.2


@pytest.mark.surface
def test_a_saved_baseline_reads_back_unchanged(tmp_path):
    case = _case()
    first = np.linspace(-1, 1, 6, dtype=np.float32).reshape(2, 3)
    second = first.copy()
    second[1, 2] = _steps(second[1, 2], 2)
    outputs = [first, second]
    measured = tool.Measured(case, {"nrmse": [0.25, 0.5], "seconds": [1.0, 2.0]}, outputs)
    path = tmp_path / "baseline.npz"
    tool.save_baseline(path, "gpu", [measured])
    loaded = tool.load_baseline(path)
    assert loaded.device == "gpu"
    restored = loaded.cases[case.name]
    assert restored.case == case and restored.metrics == measured.metrics
    assert restored.spread == [tool.UlpStats(0.0, 2)]
    np.testing.assert_array_equal(restored.reference, outputs[0])
    with pytest.raises(FileExistsError):
        tool.save_baseline(path, "gpu", [measured])


@pytest.mark.surface
@pytest.mark.parametrize(
    "arguments",
    [
        ["--quick", "--compare-baseline", "{baseline}"],
        ["--json-out", "{baseline}"],
        ["--compare-baseline", "{junk}"],
        ["--repeats", "0"],
    ],
)
def test_usage_errors_exit_2_before_any_work(tmp_path, arguments):
    baseline = tmp_path / "baseline.npz"
    zipfile.ZipFile(baseline, "w").close()
    junk = tmp_path / "junk.npz"
    junk.write_text("not a zip")
    argv = [a.format(baseline=baseline, junk=junk) for a in arguments]
    with pytest.raises(SystemExit) as stop:
        tool.main(argv)
    assert stop.value.code == 2


def _rewrite(source, target, *, meta=None, array=None):
    """Copy a baseline, passing meta.json through ``meta`` and output_0.npy through ``array``."""
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(target, "x") as new:
        for name in old.namelist():
            if name == "meta.json" and meta is not None:
                new.writestr(name, json.dumps(meta(json.loads(old.read(name)))))
            elif name == "output_0.npy" and array is not None:
                with old.open(name) as stream:
                    out = array(np.lib.format.read_array(stream))
                with new.open(name, "w") as stream:
                    np.lib.format.write_array(stream, out)
            else:
                new.writestr(name, old.read(name))


@pytest.mark.numerical
def test_a_cpu_baseline_compares_equal_and_injected_regressions_exit_1(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")  # main() sets it for --device; restore it after
    baseline = tmp_path / "baseline.npz"
    common = ["--device", "cpu", "--geometries", "parallel", "--methods", "fbp", "cgls"]
    assert tool.main([*common, "--quick", "--save-baseline", str(baseline)]) == 0
    record = tmp_path / "record.json"
    assert tool.main([*common, "--compare-baseline", str(baseline), "--json-out", str(record)]) == 0
    saved = json.loads(record.read_text())
    assert saved["passed"] is True and [c["case"]["method"] for c in saved["cases"]] == [
        "fbp",
        "cgls",
    ]
    assert all(check["passed"] for c in saved["cases"] for check in c["checks"])

    def one_ulp_off(volume):
        volume = volume.copy()
        volume.flat[volume.size // 2] = _steps(volume.flat[volume.size // 2], 1)
        return volume

    nudged = tmp_path / "nudged.npz"
    _rewrite(baseline, nudged, array=one_ulp_off)
    _ = capsys.readouterr()
    assert tool.main([*common, "--compare-baseline", str(nudged)]) == 1
    assert "parallel/fbp ulp:" in capsys.readouterr().out

    def more_accurate(meta):
        meta["cases"][1]["metrics"]["nrmse"] = [
            0.95 * e for e in meta["cases"][1]["metrics"]["nrmse"]
        ]
        return meta

    stricter = tmp_path / "stricter.npz"
    _rewrite(baseline, stricter, meta=more_accurate)
    assert tool.main([*common, "--compare-baseline", str(stricter)]) == 1
    out = capsys.readouterr().out
    assert "parallel/cgls error:" in out and "ulp:" not in out
