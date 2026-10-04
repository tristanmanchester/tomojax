"""Benchmark failures and independent quality gates must not become speed wins."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def benchmark():
    path = Path(__file__).resolve().parents[1] / "bench" / "compare_reconstructions.py"
    spec = importlib.util.spec_from_file_location("reconstruction_benchmark_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_provenance_tracks_executed_snapshot_not_caller(benchmark, tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    caller = tmp_path / "caller"
    (snapshot / "src").mkdir(parents=True)
    (snapshot / "bench").mkdir()
    caller.mkdir()
    (snapshot / "src" / "solver.py").write_text("version = 1\n")
    script = snapshot / "bench" / "compare_reconstructions.py"
    script.write_text("# frozen benchmark\n")
    monkeypatch.setattr(benchmark, "__file__", str(script))
    monkeypatch.setattr(benchmark.subprocess, "check_output", lambda *a, **kw: "diagnostic\n")
    monkeypatch.chdir(caller)
    initial = benchmark.environment()
    (caller / "unrelated.py").write_text("version = 2\n")
    assert benchmark.environment()["source_tree_sha256"] == initial["source_tree_sha256"]
    (snapshot / "src" / "solver.py").write_text("version = 3\n")
    assert benchmark.environment()["source_tree_sha256"] != initial["source_tree_sha256"]
    assert initial["source_root"] == str(snapshot)


def test_quality_preserves_physical_scale_and_rejects_nonfinite(benchmark):
    truth = np.linspace(0.1, 1.0, 12).reshape(3, 2, 2)
    assert benchmark.quality(truth, truth, 0.03)["accepted"]
    assert not benchmark.quality(2 * truth, truth, 0.03)["accepted"]
    for value in (np.inf, np.nan):
        result = benchmark.quality(np.full(truth.shape, value), truth, 0.03)
        assert not result["accepted"]
        assert result["volume_relative_l2"] is None
        json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("layout", ["contiguous", "reversed", "mapped", "large_range"])
def test_quality_matches_full_fp64_metric_across_buffers(benchmark, tmp_path, layout):
    rng = np.random.default_rng(2531)
    truth = rng.uniform(-1, 1, (65, 66, 67)).astype(np.float32)
    volume = truth + rng.normal(0, 0.01, truth.shape).astype(np.float32)
    if layout == "reversed":
        volume, truth = volume[::-1].transpose(2, 0, 1), truth[::-1].transpose(2, 0, 1)
    elif layout == "mapped":
        mapped = np.memmap(tmp_path / "volume.bin", mode="w+", dtype=np.float32, shape=volume.shape)
        mapped[:] = volume
        mapped.flush()
        volume = np.memmap(tmp_path / "volume.bin", mode="r", dtype=np.float32, shape=volume.shape)
    elif layout == "large_range":
        volume, truth = volume.astype(np.float64) * 1e80, truth.astype(np.float64) * 1e80
    difference = volume.astype(np.float64) - truth
    expected = np.sqrt(
        np.sum(difference**2, dtype=np.float64)
        / np.sum(np.square(truth, dtype=np.float64), dtype=np.float64)
    )
    result = benchmark.quality(volume, truth, 0.03)
    assert result["finite"] and result["accepted"]
    np.testing.assert_allclose(result["volume_relative_l2"], expected, rtol=1e-14)


def test_quality_rejects_late_nonfinite_and_broadcast_shapes(benchmark):
    volume = np.ones((65, 66, 67), np.float32)
    truth = volume.copy()
    volume[-1, -1, -1] = np.nan
    assert benchmark.quality(volume, truth, 0.03) == {
        "finite": False,
        "volume_relative_l2": None,
        "accepted": False,
    }
    with pytest.raises(ValueError, match="matching.*shapes"):
        benchmark.quality(truth[:1], truth, 0.03)


def test_worker_counts_unsuccessful_budgets_and_repeats_from_zero(benchmark, monkeypatch, tmp_path):
    case = SimpleNamespace(name="parallel-64-180", volume=np.ones((3, 2, 2)))
    monkeypatch.setattr(benchmark, "load_fixture", lambda _: case)
    budgets = []

    def solve(case, budget, batch, method):
        budgets.append(budget)
        return case.volume * (0.99 if budget >= 4 else 0.5), {}

    monkeypatch.setattr(benchmark, "solve_tomojax", solve)
    args = argparse.Namespace(
        fixture=tmp_path / "unused",
        kind="parallel",
        method="tomojax_fista",
        started=time.perf_counter(),
        batch=3,
        max_iters=16,
        repeats=2,
        output=tmp_path / "result.json",
    )
    record = benchmark.run_worker(args)
    assert budgets == [1, 2, 4, 4, 4]
    assert record["status"] == "accepted"
    assert record["cold_search_verified_ms"] >= sum(r["verified_ms"] for r in record["search"])


def test_numerical_breakdown_does_not_count_as_accepted(benchmark, monkeypatch, tmp_path):
    case = SimpleNamespace(name="parallel-64-180", volume=np.ones((3, 2, 2)))
    monkeypatch.setattr(benchmark, "load_fixture", lambda _: case)
    monkeypatch.setattr(
        benchmark,
        "solve_tomojax",
        lambda *args: (case.volume, {"termination": "numerical_breakdown"}),
    )
    args = argparse.Namespace(
        fixture=tmp_path / "unused",
        kind="parallel",
        method="tomojax_cgls_jax",
        started=time.perf_counter(),
        batch=3,
        max_iters=4,
        repeats=2,
        output=tmp_path / "result.json",
    )
    record = benchmark.run_worker(args)
    assert record["status"] == "target_not_reached"
    assert not record["repeats"]
    assert len(record["failed_budget_repeats"]) == 2
    assert not any(row["accepted"] for row in record["failed_budget_repeats"])


def test_process_failure_overrides_partial_success(benchmark, tmp_path):
    output = tmp_path / "result.json"
    output.write_text(json.dumps({"status": "accepted"}))
    record = benchmark.isolated_run([sys.executable, "-c", "raise SystemExit(7)"], output, 10)
    assert record["status"] == "execution_failed"
    assert record["exit_code"] == 7


def test_process_memory_filters_other_gpu_processes(benchmark, tmp_path):
    path = tmp_path / "memory.csv"
    path.write_text("71, 128\n999, 8192\n71, 196\n71, [N/A]\n")
    record = benchmark.read_peak_memory(path, 71)
    assert record["sampled_process_peak_mib"] == 196
    assert record["memory_samples"] == 2
    assert benchmark.read_peak_memory(path, 123)["sampled_process_peak_mib"] is None


def test_external_fixture_and_geometry_never_import_jax_or_pallas(tmp_path):
    fixture = tmp_path / "physical.npz"
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    poses[1, :2, :2] = [[0, -1], [1, 0]]
    poses[:, :3, 3] = [[0.2, -0.4, 0.1], [-0.3, 0.5, -0.2]]
    np.savez(
        fixture,
        name="anisotropic-independent",
        grid=json.dumps(
            dict(nx=5, ny=4, nz=3, vx=0.8, vy=1.2, vz=1.4, vol_origin=[-1.3, -2.1, -0.8])
        ),
        detector=json.dumps(dict(nu=7, nv=5, du=0.7, dv=1.1, det_center=[0.27, -0.31])),
        poses=poses,
        angles=np.array([0, 90], np.float32),
        truth=np.zeros((5, 4, 3), np.float32),
        data=np.ones((2, 5, 7), np.float32),
    )
    # A fresh process is essential: the pytest process already imports JAX.
    code = """
import importlib.abc
import sys
import types
class RejectSolverImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax', 'jaxlib'} or fullname.startswith('tomojax.core.pallas'):
            raise AssertionError('unrelated solver imported: '+fullname)
sys.meta_path.insert(0, RejectSolverImport())
sys.path.insert(0, sys.argv[1])
from compare_reconstructions import load_fixture
from direct_reconstruction import astra_geometries, extend_filter_support
import numpy as np
case = load_fixture(sys.argv[2])
stub = types.ModuleType('astra')
stub.create_vol_geom = lambda *args: args
stub.create_proj_geom = lambda *args: args
sys.modules['astra'] = stub
volume, projections = astra_geometries(case)
np.testing.assert_allclose(volume, [4,5,3,-1.7,2.3,-2.7,2.1,-1.5,2.7])
vectors = projections[-1]
for i, pose in enumerate(case.poses):
    world_center = np.array([.27, 0, -.31])
    np.testing.assert_allclose(vectors[i,:3], pose[:3,:3].T @ [0,1,0], atol=1e-7)
    np.testing.assert_allclose(vectors[i,3:6], pose[:3,:3].T @ (world_center-pose[:3,3]))
    np.testing.assert_allclose(vectors[i,6:9], pose[:3,:3].T @ [.7,0,0], atol=1e-7)
    np.testing.assert_allclose(vectors[i,9:12], pose[:3,:3].T @ [0,0,1.1], atol=1e-7)
padded, width = extend_filter_support(case)
assert padded.analytic.shape[-1] == case.detector.nu + 2*width
assert 'compare_projectors' not in sys.modules
assert 'jax' not in sys.modules
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(Path(__file__).resolve().parents[1] / "bench"),
            str(fixture),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_resume_rejects_partial_failed_budget_repeats(benchmark, tmp_path):
    expected = resume_record()
    previous = json.loads(json.dumps(expected))
    previous["records"][0] = {
        "case": "parallel-64-180",
        "method": "tomojax_fbp_pallas",
        "status": "target_not_reached",
        "failed_budget_repeats": [{"accepted": False}],
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(previous))
    with pytest.raises(ValueError, match="failed-budget warm attempts are incomplete"):
        benchmark.resume_payload(path, expected)


def test_multires_policy_preserves_total_work_and_fine_refinement(benchmark):
    for size in (16, 32, 64, 127, 256):
        for budget in benchmark.BUDGETS:
            factors, budgets = benchmark.multires_schedule((size, size - 3, size // 2), budget)
            assert sum(budgets) == budget
            assert factors[-1] == 1
            assert min(budgets) >= 1
            if budget > 1:
                assert np.ceil(size / factors[0]) <= 32


def test_analytic_chords_match_sphere_geometry_and_direction_scaling(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "bench"))
    module = importlib.import_module("compare_projectors")
    base = np.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.0, 0.0, 0.0], [2.1, 0.0, 0.0]])
    for speed in (0.5, 1.0, 2.0):
        lengths = module.ellipsoid_chord_lengths(
            base, np.asarray([0.0, speed, 0.0]), np.zeros(3), np.eye(3) / 4
        )
        np.testing.assert_allclose(lengths, [4.0, 3.2, 0.0, 0.0], atol=1e-12)


def test_structured_noise_fixture_is_reproducible_and_retains_truth(
    benchmark, monkeypatch, tmp_path
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "bench"))
    clean_path, noisy_path = tmp_path / "clean.npz", tmp_path / "noisy.npz"
    benchmark.generate_fixture(clean_path, 16, 9, "lamino", "structured-v1")
    benchmark.generate_fixture(noisy_path, 16, 9, "lamino", "structured-noisy-v1")
    clean = benchmark.load_fixture(clean_path)
    noisy = benchmark.load_fixture(noisy_path)
    np.testing.assert_array_equal(clean.volume, noisy.volume)
    scale = np.sqrt(np.mean(np.square(clean.analytic, dtype=np.float64)))
    assert np.std(noisy.analytic - clean.analytic) / scale == pytest.approx(0.01, rel=0.08)
    benchmark.generate_fixture(noisy_path, 16, 9, "lamino", "structured-noisy-v1")
    np.testing.assert_array_equal(noisy.analytic, benchmark.load_fixture(noisy_path).analytic)


def test_failed_direct_reconstruction_keeps_search_and_warm_attempts_separate(
    benchmark, monkeypatch, tmp_path
):
    case = SimpleNamespace(name="parallel-64-180", volume=np.ones((3, 2, 2)))
    monkeypatch.setattr(benchmark, "load_fixture", lambda _: case)
    calls = []

    def solve(case, budget, batch, method):
        calls.append(budget)
        return case.volume * 0.5, {}

    monkeypatch.setattr(benchmark, "solve_tomojax", solve)
    args = argparse.Namespace(
        fixture=tmp_path / "unused",
        kind="parallel",
        method="tomojax_fbp_pallas",
        started=time.perf_counter(),
        batch=3,
        max_iters=256,
        repeats=7,
        output=tmp_path / "result.json",
    )
    result = benchmark.run_worker(args)
    assert calls == [1] * 8
    assert result["budget_kind"] == "single_pass"
    assert result["status"] == "target_not_reached"
    assert result["repeats"] == []
    assert len(result["search"]) == 1
    assert len(result["failed_budget_repeats"]) == 7
    assert result["failed_budget_iterations"] == 1
    assert result["failed_budget_warm_verified_median_ms"] > 0
    assert "cold_search_verified_ms" not in result
    assert "selected_budget_warm_verified_median_ms" not in result


def resume_record():
    return {
        "suite": "gaussian-v1",
        "standard_suite": False,
        "environment": {"source_tree_sha256": "frozen", "gpu": "test GPU"},
        "arguments": {
            "sizes": [64, 128],
            "geometries": ["parallel"],
            "methods": ["tomojax_fbp_pallas"],
            "views": 180,
            "batch": 16,
            "fourier_slices": 16,
            "repeats": 2,
            "max_iters": 256,
            "timeout": 30,
        },
        "records": [
            {
                "case": "parallel-64-180",
                "method": "tomojax_fbp_pallas",
                "status": "accepted",
                "repeats": [{"accepted": True}] * 2,
            }
        ],
    }


@pytest.mark.parametrize(
    "change", ["source", "gpu", "budget", "slabs", "duplicate", "case", "partial", "failed"]
)
def test_resume_rejects_incompatible_or_incomplete_evidence(benchmark, tmp_path, change):
    expected = resume_record()
    previous = json.loads(json.dumps(expected))
    if change == "source":
        previous["environment"]["source_tree_sha256"] = "changed"
    elif change == "gpu":
        previous["environment"]["gpu"] = None
    elif change == "budget":
        previous["arguments"]["max_iters"] = 128
    elif change == "slabs":
        previous["arguments"]["fourier_slices"] = 8
    elif change == "duplicate":
        previous["records"] *= 2
    elif change == "case":
        previous["records"][0]["case"] = "parallel-512-180"
    elif change == "partial":
        previous["records"][0]["repeats"].pop()
    else:
        previous["records"][0]["repeats"][0]["accepted"] = False
    path = tmp_path / "result.json"
    path.write_text(json.dumps(previous))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Cannot resume"):
        benchmark.resume_payload(path, expected)
    assert path.read_bytes() == before


def test_atomic_write_preserves_previous_record_on_replace_failure(
    benchmark, monkeypatch, tmp_path
):
    path = tmp_path / "result.json"
    path.write_text('{"retained": true}\n')

    def fail(*args):
        raise OSError("injected replacement failure")

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError, match="injected"):
        benchmark.write_result(path, {"new": True})
    assert json.loads(path.read_text()) == {"retained": True}
    assert list(tmp_path.iterdir()) == [path]


def test_resume_skips_completed_cases_and_preserves_failures(benchmark, monkeypatch, tmp_path):
    path = tmp_path / "result.json"
    argv = [
        "benchmark",
        "--sizes",
        "64",
        "128",
        "--geometries",
        "parallel",
        "--methods",
        "tomojax_fbp_pallas",
        "--repeats",
        "2",
        "--timeout",
        "30",
        "--output",
        str(path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(benchmark, "environment", lambda: resume_record()["environment"])
    generated = []
    monkeypatch.setattr(benchmark.subprocess, "run", lambda cmd, **kw: generated.append(cmd))
    calls = []

    def first_run(command, output, timeout):
        calls.append(command)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return {
            "status": "target_not_reached",
            "search": [{"accepted": False}],
            "failed_budget_repeats": [{"accepted": False}] * 2,
        }

    monkeypatch.setattr(benchmark, "isolated_run", first_run)
    with pytest.raises(KeyboardInterrupt):
        benchmark.main()
    retained = json.loads(path.read_text())["records"][0]
    assert retained["status"] == "target_not_reached"
    generated.clear()
    monkeypatch.setattr(sys, "argv", [*argv, "--resume"])
    monkeypatch.setattr(
        benchmark,
        "isolated_run",
        lambda *args: {
            "status": "accepted",
            "repeats": [{"accepted": True}] * 2,
        },
    )
    assert benchmark.main() == 1
    result = json.loads(path.read_text())
    assert len(generated) == 1
    assert generated[0][generated[0].index("--sizes") + 1] == "128"
    assert result["records"][0] == retained
    assert len(result["records"]) == 2
    assert not result["passed"]
    generated.clear()
    assert benchmark.main() == 1
    assert generated == []
    monkeypatch.setattr(sys, "argv", argv)
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        benchmark.main()
    assert path.read_bytes() == before


def test_fourier_slab_option_reaches_isolated_worker_and_result(benchmark, monkeypatch, tmp_path):
    from tomojax.geometry import Detector, Grid
    import tomojax.recon

    case = SimpleNamespace(
        name="parallel-8-2",
        grid=Grid(8, 8, 8, 1.0, 1.0, 1.0),
        detector=Detector(8, 8, 1.0, 1.0),
        angles_deg=np.array([0.0, 90.0]),
        analytic=np.zeros((2, 8, 8), np.float32),
        volume=np.ones((8, 8, 8), np.float32),
    )
    slabs = []

    def reconstruct(*args, config):
        slabs.append(config.slices_per_batch)
        return case.volume

    monkeypatch.setattr(tomojax.recon, "fourier_reconstruct", reconstruct)
    monkeypatch.setattr(benchmark, "load_fixture", lambda _: case)
    monkeypatch.setattr(benchmark, "environment", dict)
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs: None)

    def worker(command, output, timeout):
        with monkeypatch.context() as context:
            context.setattr(sys, "argv", command[1:])
            return benchmark.run_worker(benchmark.parse_arguments())

    monkeypatch.setattr(benchmark, "isolated_run", worker)
    output = tmp_path / "slabs.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--sizes",
            "8",
            "--views",
            "2",
            "--geometries",
            "parallel",
            "--methods",
            "tomojax_fourier_cupy",
            "--batch",
            "7",
            "--fourier-slices",
            "3",
            "--repeats",
            "2",
            "--output",
            str(output),
        ],
    )
    assert benchmark.main() == 0
    result = json.loads(output.read_text())
    assert result["arguments"]["fourier_slices"] == 3
    assert slabs == [3, 3, 3]
    rows = [*result["records"][0]["search"], *result["records"][0]["repeats"]]
    assert all(row["info"]["slices_per_batch"] == 3 for row in rows)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_fourier_slab_cli_rejects_nonpositive_values(benchmark, monkeypatch, value):
    monkeypatch.setattr(sys, "argv", ["benchmark", "--fourier-slices", value])
    with pytest.raises(SystemExit) as error:
        benchmark.parse_arguments()
    assert error.value.code == 2
