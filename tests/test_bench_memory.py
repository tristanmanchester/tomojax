from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

fitness = importlib.import_module("fitness")
memory = importlib.import_module("memory")


class _FakeProc:
    def __init__(self, pid: int, *, children: list["_FakeProc"] | None = None) -> None:
        self.pid = pid
        self._children = list(children or [])

    def children(self, recursive: bool = True) -> list["_FakeProc"]:
        return list(self._children)

    def is_running(self) -> bool:
        return True


class _FakeMemoryInfo:
    def __init__(self, used: int) -> None:
        self.used = used


class _FakeProcessRow:
    def __init__(self, pid: int, used_gpu_memory: int) -> None:
        self.pid = pid
        self.usedGpuMemory = used_gpu_memory


class _FakeNVML:
    def __init__(self, *, device_used: list[int], process_rows: dict[int, list[_FakeProcessRow]]) -> None:
        self._device_used = list(device_used)
        self._process_rows = dict(process_rows)

    def nvmlInit(self) -> None:
        return None

    def nvmlShutdown(self) -> None:
        return None

    def nvmlDeviceGetCount(self) -> int:
        return len(self._device_used)

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        return index

    def nvmlDeviceGetMemoryInfo(self, handle: int) -> _FakeMemoryInfo:
        return _FakeMemoryInfo(self._device_used[handle])

    def nvmlDeviceGetComputeRunningProcesses_v3(self, handle: int) -> list[_FakeProcessRow]:
        return list(self._process_rows.get(handle, []))


def test_gpu_memory_monitor_prefers_process_scope_and_sums_child_processes() -> None:
    child = _FakeProc(4321)
    fake_nvml = _FakeNVML(
        device_used=[512 * 1024 * 1024],
        process_rows={
            0: [
                _FakeProcessRow(1234, 128 * 1024 * 1024),
                _FakeProcessRow(4321, 64 * 1024 * 1024),
                _FakeProcessRow(9999, 256 * 1024 * 1024),
            ]
        },
    )
    monitor = memory.GpuMemoryMonitor(
        enabled=True,
        interval_seconds=0.01,
        root_pid=1234,
        nvml_module=fake_nvml,
        process_factory=lambda pid: _FakeProc(pid, children=[child]),
    )

    monitor.sample_once()
    snapshot = monitor.snapshot()

    assert snapshot.backend == "nvml"
    assert snapshot.scope == "process"
    assert snapshot.process_peak_mb == 192.0
    assert snapshot.device_peak_mb == 512.0
    assert snapshot.process_source == "nvml-process-query"
    assert snapshot.process_supported is True
    assert snapshot.sample_count == 1
    assert snapshot.observed_gpu_count == 1
    assert snapshot.sampler_error is None


class _DeviceOnlyNVML(_FakeNVML):
    def nvmlDeviceGetComputeRunningProcesses_v3(self, handle: int) -> list[_FakeProcessRow]:
        raise RuntimeError("process queries unavailable")


def test_gpu_memory_monitor_falls_back_to_device_scope() -> None:
    monitor = memory.GpuMemoryMonitor(
        enabled=True,
        interval_seconds=0.01,
        root_pid=1234,
        nvml_module=_DeviceOnlyNVML(device_used=[768 * 1024 * 1024], process_rows={}),
        process_factory=lambda pid: _FakeProc(pid),
    )

    monitor.sample_once()
    snapshot = monitor.snapshot()

    assert snapshot.backend == "nvml"
    assert snapshot.scope == "device_fallback"
    assert snapshot.process_peak_mb is None
    assert snapshot.device_peak_mb == 768.0
    assert snapshot.process_source is None
    assert snapshot.process_supported is False


class _UnavailableNVML:
    def nvmlInit(self) -> None:
        raise RuntimeError("driver missing")


def test_gpu_memory_monitor_reports_unavailable_nvml() -> None:
    monitor = memory.GpuMemoryMonitor(
        enabled=True,
        interval_seconds=0.01,
        root_pid=1234,
        nvml_module=_UnavailableNVML(),
        process_factory=lambda pid: _FakeProc(pid),
    )

    monitor.sample_once()
    snapshot = monitor.snapshot()

    assert snapshot.backend == "none"
    assert snapshot.scope == "unavailable"
    assert snapshot.process_peak_mb is None
    assert snapshot.device_peak_mb is None
    assert snapshot.process_source is None
    assert snapshot.process_supported is False
    assert "NVML unavailable" in str(snapshot.sampler_error)


def test_timed_call_uses_process_peak_before_device_peak() -> None:
    class _Mods:
        class jax:
            @staticmethod
            def block_until_ready(value: object) -> object:
                return value

    class _StubMonitor:
        def __init__(self, **_: object) -> None:
            self.peak_host_rss_mb = 321.0

        def start(self) -> None:
            return None

        def stop(self) -> memory.GpuMemorySnapshot:
            return memory.GpuMemorySnapshot(
                backend="nvml",
                scope="process",
                process_peak_mb=111.0,
                device_peak_mb=222.0,
                process_source="nvml-process-query",
                process_supported=True,
                sample_interval_seconds=0.01,
                sample_count=7,
                observed_gpu_count=1,
                sampler_error=None,
            )

    original = fitness.PeakMemoryMonitor
    fitness.PeakMemoryMonitor = _StubMonitor
    try:
        result = fitness._timed_call(lambda: {"ok": True}, _Mods(), {})
    finally:
        fitness.PeakMemoryMonitor = original

    assert result.peak_gpu_memory_mb == 111.0
    assert result.peak_gpu_memory_process_mb == 111.0
    assert result.peak_gpu_memory_device_mb == 222.0
    assert result.gpu_memory_backend == "nvml"
    assert result.gpu_memory_scope == "process"
    assert result.gpu_memory_process_source == "nvml-process-query"
    assert result.gpu_memory_process_supported is True
    assert result.gpu_memory_sample_count == 7
