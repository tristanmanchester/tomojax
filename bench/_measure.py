"""What a benchmark ran on, and what each GPU did while it ran.

:func:`environment` records the machine, the command and every installed
package's version. :class:`GpuSampler` polls ``nvidia-smi`` in the background;
:meth:`GpuSampler.window` summarises each GPU's samples between two times. The
memory figures are sampled process footprints (allocator pools included, short
peaks between samples missed), not exact operation peaks, and NVIDIA averages
utilisation over its own period of roughly 1/6 to 1 s, whatever the polling rate.
"""

from __future__ import annotations

import hashlib
from importlib import metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from typing import Any

_QUERY = "index,memory.used,utilization.gpu"


def environment() -> dict[str, Any]:
    """The GPUs and their topology, driver, CPU, memory, command and package versions."""
    record: dict[str, Any] = {
        "command": sys.argv,
        "python": platform.python_version(),
        "host": platform.node(),
        "cpu": _cpu(),
        "cpu_count": os.cpu_count(),
        "tomojax_commit": os.environ.get("TOMOJAX_COMMIT") or _git_commit(),
        "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()},
    }
    record["gpus"] = _nvidia_smi(
        "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"
    ).splitlines()
    record["gpu_topology"] = _nvidia_smi("topo", "-m")
    pages = os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else 0
    record["host_memory_gib"] = round(pages * os.sysconf("SC_PAGE_SIZE") / 2**30, 1)
    return record


def run_bounded(command: list[str], *, log: Path, timeout: float | None, **options: Any) -> str:
    """Run ``command`` with its output in ``log``; ``"exit N"``, or ``"stopped after T s"``.

    The command runs in a process group of its own, which a timeout ends
    whole, children included.
    """
    import signal

    with log.open("w") as stream:
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, **options
        )
        try:
            return f"exit {process.wait(timeout=timeout)}"
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return f"stopped after {timeout:g} s"


def file_hashes(paths: list[Path]) -> dict[str, str]:
    """The SHA-256 of each file, by name."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def _nvidia_smi(*args: str) -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _cpu() -> str:
    try:
        with Path("/proc/cpuinfo").open(encoding="utf-8") as info:
            for line in info:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class GpuSampler:
    """``nvidia-smi`` samples of every GPU's memory and utilisation, every ``period`` s.

    The baseline is each GPU's memory in use when sampling began: create the
    sampler before the library under test touches a GPU.
    """

    def __init__(self, period: float = 0.05, *, settle: float = 3.0) -> None:
        self.samples: list[tuple[float, int, float, float]] = []  # time, GPU, MiB, %
        self.baseline: dict[int, float] = {}
        self._stop = threading.Event()
        self._process: subprocess.Popen[str] | None = None
        try:
            self._process = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits",
                 f"-lms={max(1, int(period * 1000))}"],
                stdout=subprocess.PIPE, text=True,
            )  # fmt: skip
        except OSError:
            return
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        gpus = len(_nvidia_smi("-L").splitlines())
        deadline = time.perf_counter() + settle
        # Each GPU reported twice, so every one has a baseline.
        while time.perf_counter() < deadline:
            counts: dict[int, int] = {}
            for _, gpu, _, _ in self.samples:
                counts[gpu] = counts.get(gpu, 0) + 1
            if len(counts) >= gpus and min(counts.values(), default=0) >= 2:
                break
            time.sleep(period)
        self.baseline = {gpu: memory for _, gpu, memory, _ in self.samples}

    def _read(self) -> None:
        assert self._process is not None
        assert self._process.stdout is not None
        for line in self._process.stdout:
            if self._stop.is_set():
                break
            try:
                gpu, memory, utilisation = (float(x) for x in line.split(","))
            except ValueError:
                continue
            self.samples.append((time.perf_counter(), int(gpu), memory, utilisation))

    def window(self, start: float, stop: float) -> dict[str, dict[str, float | None]]:
        """Each GPU between ``start`` and ``stop``: samples, memory (MiB) and utilisation (%).

        ``peak_memory_mib`` is the largest sampled memory in use, and
        ``peak_above_baseline_mib`` that less what the GPU held when sampling
        began (None if it had not reported by then).
        """
        per: dict[int, list[tuple[float, float]]] = {}
        for t, gpu, memory, utilisation in self.samples:
            if start <= t <= stop:
                per.setdefault(gpu, []).append((memory, utilisation))
        summary: dict[str, dict[str, float | None]] = {}
        for gpu, values in sorted(per.items()):
            peak = max(m for m, _ in values)
            base = self.baseline.get(gpu)
            summary[str(gpu)] = {
                "samples": len(values),
                "peak_memory_mib": peak,
                "peak_above_baseline_mib": None if base is None else peak - base,
                "mean_utilisation": sum(u for _, u in values) / len(values),
            }
        return summary

    def close(self) -> None:
        self._stop.set()
        if self._process is not None:
            self._process.terminate()
