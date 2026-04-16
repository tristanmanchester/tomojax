from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = spec_from_file_location(name, module_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_exp_spdhg_report_main_writes_markdown_summary(monkeypatch, tmp_path, capsys):
    report_mod = _load_module("exp_spdhg_report_test", "scripts/exp_spdhg_report.py")
    metrics = {
        "dataset": {"nx": 8, "ny": 9, "nz": 10, "n_views": 11, "phantom": "cube"},
        "fbp": {"psnr": 1.0, "ssim_center": 0.2, "mse": 3.0, "tv": 4.0},
        "fista": {"psnr": 5.0, "ssim_center": 0.6, "mse": 7.0, "tv": 8.0},
        "spdhg": {"psnr": 9.0, "ssim_center": 1.0, "mse": 11.0, "tv": 12.0},
        "timing_sec": {"fbp": 0.1, "fista": 0.2, "spdhg": 0.3},
    }
    indir = tmp_path / "exp_spdhg_256"
    indir.mkdir()
    (indir / "metrics.json").write_text(json.dumps(metrics))

    monkeypatch.setattr(sys, "argv", ["exp_spdhg_report", "--indir", str(indir)])

    report_mod.main()

    out_path = indir / "REPORT.md"
    text = out_path.read_text()
    captured = capsys.readouterr()
    assert "# CT Benchmark Report" in text
    assert "Dataset: 8x9x10, views=11, phantom=cube" in text
    assert "- FISTA: PSNR=5.0, SSIM_center=0.6, MSE=7.0, TV=8.0" in text
    assert "- SPDHG: 0.3" in text
    assert f"Wrote {out_path}" in captured.out


def test_perf_harness_run_sets_progress_env_and_truncates_stdout(monkeypatch):
    perf_mod = _load_module("perf_harness_test", "scripts/perf_harness.py")
    calls: list[tuple[list[str], dict[str, str]]] = []
    command_stdout = "x" * 2500

    def fake_subprocess_run(cmd, *, env, stdout, stderr, text):
        calls.append((cmd, env))
        return SimpleNamespace(returncode=7, stdout=command_stdout)

    class FakeProcess:
        def memory_info(self):
            return SimpleNamespace(rss=10)

    perf_mod.psutil = SimpleNamespace(Process=lambda: FakeProcess())
    monkeypatch.setattr(perf_mod.time, "perf_counter", iter([1.0, 2.25]).__next__)
    monkeypatch.setattr(perf_mod.subprocess, "run", fake_subprocess_run)

    result = perf_mod.run(["python", "-m", "tomojax.cli.recon"])

    assert calls[0][1]["TOMOJAX_PROGRESS"] == "0"
    assert result["cmd"] == ["python", "-m", "tomojax.cli.recon"]
    assert result["rc"] == 7
    assert result["secs"] == 1.25
    assert result["rss_delta"] == 0
    assert result["stdout"] == command_stdout[-2000:]


def test_perf_harness_main_writes_json_and_prints_summary(monkeypatch, tmp_path, capsys):
    perf_mod = _load_module("perf_harness_main_test", "scripts/perf_harness.py")
    run_calls: list[list[str]] = []

    def fake_run(cmd: list[str]) -> dict:
        run_calls.append(cmd)
        return {"cmd": cmd, "rc": 0, "secs": 0.5, "rss_delta": 0, "stdout": "ok"}

    monkeypatch.setattr(perf_mod, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "perf_harness",
            "--data",
            "dataset.nxs",
            "--outdir",
            str(tmp_path),
            "--modes",
            "fbp",
        ],
    )

    perf_mod.main()

    saved = json.loads((tmp_path / "perf_results.json").read_text())
    captured = capsys.readouterr()
    assert len(run_calls) == 4
    assert len(saved) == 4
    assert all(row["algo"] == "fbp" for row in saved)
    assert "fbp   dt=fp32" in captured.out


def test_visualize_helpers_cover_empty_and_filtered_cases():
    viz_mod = _load_module("bench_visualize_test", "bench/visualize.py")
    volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    slices = viz_mod._central_slices(volume)
    assert slices["xy"].shape == (2, 3)
    assert slices["xz"].shape == (2, 4)
    assert slices["yz"].shape == (3, 4)

    lo, hi = viz_mod._display_limits(np.array([[np.nan, np.inf], [-np.inf, np.nan]]))
    assert (lo, hi) == (0.0, 1.0)

    trace = [
        {"outer_idx": 1, "metric": "bad"},
        {"outer_idx": 2, "metric": 0.5, "level_factor": 4},
        {"outer_idx": 3, "metric": 0.25},
    ]
    xs, ys, levels = viz_mod._trace_points(trace, value_key="metric")
    assert xs.tolist() == [2, 3]
    assert np.allclose(ys, np.array([0.5, 0.25], dtype=np.float32))
    assert levels == [4, None]
    assert viz_mod._error_limit(np.array([[np.nan, 5.0], [1.0, 3.0]], dtype=np.float32)) > 0.0


def test_bench_memory_monitor_parses_nvidia_smi_and_tracks_pids(monkeypatch):
    mem_mod = _load_module("bench_memory_test", "bench/memory.py")

    class FakeChild:
        def __init__(self, pid: int, running: bool) -> None:
            self.pid = pid
            self._running = running

        def is_running(self) -> bool:
            return self._running

    class FakeRoot:
        def children(self, recursive: bool = True):
            return [FakeChild(2, True), FakeChild(3, False)]

    def fake_subprocess_run(*args, **kwargs):
        return SimpleNamespace(stdout="2, 10\n4, 20\n")

    monitor = mem_mod.GpuMemoryMonitor(
        enabled=True,
        interval_seconds=0.5,
        root_pid=1,
        process_factory=lambda pid: FakeRoot(),
    )
    monkeypatch.setattr(mem_mod.subprocess, "run", fake_subprocess_run)

    assert monitor._resolve_pids() == {1, 2}
    assert monitor._sample_process_memory_via_nvidia_smi({2}) == int(10 * mem_mod.MB)
    assert monitor._process_supported is True
    assert monitor._process_source == "nvidia-smi-compute-apps"


def test_bench_memory_snapshot_prefers_process_scope():
    mem_mod = _load_module("bench_memory_snapshot_test", "bench/memory.py")
    monitor = mem_mod.GpuMemoryMonitor(enabled=False, interval_seconds=0.5)
    monitor._peak_process_mb = 12.0
    monitor._peak_device_mb = 24.0
    monitor._sample_count = 3
    monitor._observed_gpu_count = 1

    snapshot = monitor.snapshot()

    assert snapshot.scope == "process"
    assert snapshot.backend == "nvml"
    assert snapshot.process_peak_mb == 12.0
    assert snapshot.device_peak_mb == 24.0
