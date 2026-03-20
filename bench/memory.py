from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any

import psutil

MB = 1024.0 * 1024.0


def _load_pynvml() -> Any:
    import pynvml  # type: ignore[import-not-found]

    return pynvml


@dataclass
class GpuMemorySnapshot:
    backend: str
    scope: str
    process_peak_mb: float | None
    device_peak_mb: float | None
    sample_interval_seconds: float
    sample_count: int
    observed_gpu_count: int
    sampler_error: str | None


class GpuMemoryMonitor:
    def __init__(
        self,
        *,
        enabled: bool,
        interval_seconds: float,
        root_pid: int | None = None,
        nvml_module: Any | None = None,
        process_factory: Any = psutil.Process,
    ) -> None:
        self.enabled = bool(enabled)
        self.interval_seconds = max(float(interval_seconds), 0.01)
        self.root_pid = int(root_pid or os.getpid())
        self._nvml_module = nvml_module
        self._process_factory = process_factory
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml: Any | None = None
        self._nvml_initialized = False
        self._device_handles: list[Any] = []
        self._peak_process_mb: float | None = None
        self._peak_device_mb: float | None = None
        self._sample_count = 0
        self._observed_gpu_count = 0
        self._sampler_error: str | None = None

    def _set_error(self, message: str) -> None:
        if self._sampler_error is None:
            self._sampler_error = message

    def _resolve_pids(self) -> set[int]:
        try:
            root = self._process_factory(self.root_pid)
        except Exception:
            return {self.root_pid}
        pids = {self.root_pid}
        try:
            for child in root.children(recursive=True):
                try:
                    if child.is_running():
                        pids.add(int(child.pid))
                except Exception:
                    continue
        except Exception:
            pass
        return pids

    def _ensure_nvml(self) -> bool:
        if not self.enabled:
            return False
        if self._nvml_initialized:
            return True
        try:
            self._nvml = self._nvml_module or _load_pynvml()
            self._nvml.nvmlInit()
            count = int(self._nvml.nvmlDeviceGetCount())
            self._device_handles = [
                self._nvml.nvmlDeviceGetHandleByIndex(index) for index in range(count)
            ]
            self._nvml_initialized = True
            return True
        except Exception as exc:
            self._set_error(f"NVML unavailable: {exc}")
            self.enabled = False
            return False

    def _query_process_rows(self, handle: Any) -> list[Any] | None:
        if self._nvml is None:
            return None
        query_names = (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetGraphicsRunningProcesses_v3",
            "nvmlDeviceGetGraphicsRunningProcesses_v2",
            "nvmlDeviceGetGraphicsRunningProcesses",
        )
        for name in query_names:
            fn = getattr(self._nvml, name, None)
            if fn is None:
                continue
            try:
                return list(fn(handle) or [])
            except Exception:
                continue
        return None

    def _extract_process_bytes(self, row: Any) -> int | None:
        value = getattr(row, "usedGpuMemory", None)
        if value is None:
            return None
        try:
            used = int(value)
        except (TypeError, ValueError):
            return None
        if used < 0:
            return None
        return used

    def sample_once(self) -> None:
        if not self._ensure_nvml():
            return
        assert self._nvml is not None

        target_pids = self._resolve_pids()
        device_peaks: list[float] = []
        process_total_bytes = 0
        saw_process_data = False

        for handle in self._device_handles:
            try:
                info = self._nvml.nvmlDeviceGetMemoryInfo(handle)
                device_peaks.append(float(info.used) / MB)
            except Exception as exc:
                self._set_error(f"NVML device memory query failed: {exc}")
                continue

            rows = self._query_process_rows(handle)
            if rows is None:
                continue
            for row in rows:
                if getattr(row, "pid", None) not in target_pids:
                    continue
                used = self._extract_process_bytes(row)
                if used is None:
                    continue
                saw_process_data = True
                process_total_bytes += used

        if device_peaks:
            self._peak_device_mb = max(self._peak_device_mb or 0.0, max(device_peaks))
        if saw_process_data:
            self._peak_process_mb = max(self._peak_process_mb or 0.0, process_total_bytes / MB)
        self._sample_count += 1
        self._observed_gpu_count = max(self._observed_gpu_count, len(self._device_handles))

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval_seconds)
        self.sample_once()

    def start(self) -> None:
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._run, name="gpu-memory-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> GpuMemorySnapshot:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._nvml_initialized and self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False
        return self.snapshot()

    def snapshot(self) -> GpuMemorySnapshot:
        if self._peak_process_mb is not None:
            scope = "process"
        elif self._peak_device_mb is not None:
            scope = "device_fallback"
        else:
            scope = "unavailable"
        backend = "nvml" if (self._sample_count > 0 or self._observed_gpu_count > 0) else "none"
        return GpuMemorySnapshot(
            backend=backend,
            scope=scope,
            process_peak_mb=self._peak_process_mb,
            device_peak_mb=self._peak_device_mb,
            sample_interval_seconds=self.interval_seconds,
            sample_count=self._sample_count,
            observed_gpu_count=self._observed_gpu_count,
            sampler_error=self._sampler_error,
        )
