"""Runtime helpers shared by real-laminography developer probes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Callable, Iterable, Mapping

import jax
import numpy as np

from tomojax.io import normalize_json


def real_lamino_json_safe(value: Any) -> Any:
    return normalize_json(value, sort_mapping_keys=True, catch_to_dict_errors=True)


def write_real_lamino_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(real_lamino_json_safe(payload), indent=2, sort_keys=True) + "\n")


def append_real_lamino_csv(
    path: Path,
    row: Mapping[str, Any],
    fieldnames: Iterable[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = list(fieldnames)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({name: real_lamino_json_safe(row.get(name)) for name in fields})


def update_real_lamino_status(path: Path, **updates: Any) -> None:
    current: dict[str, Any] = {}
    if path.exists():
        try:
            current = json.loads(path.read_text())
        except Exception:
            current = {}
    if updates.get("state") == "completed" and "error" not in updates:
        current.pop("error", None)
    if "stage" in updates and "message" not in updates:
        current.pop("message", None)
    current.update(updates)
    current["updated_at"] = time.time()
    write_real_lamino_json(path, current)


class RealLaminoGpuMonitor:
    """Small nvidia-smi CSV sampler used by real-laminography probes."""

    def __init__(self, path: Path, interval: float = 2.0) -> None:
        self.path = path
        self.interval = float(interval)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("timestamp,used_mib,total_mib,util_pct,temp_c\n")
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                if out:
                    with self.path.open("a", encoding="utf-8") as handle:
                        handle.write(out.replace(", ", ",") + "\n")
            except Exception:
                pass
            self.stop.wait(self.interval)


def real_lamino_commit_info(worktree: Path) -> dict[str, Any]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(
                args,
                cwd=worktree,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return ""

    return {
        "worktree": str(worktree),
        "commit": run(["git", "rev-parse", "--short", "HEAD"]),
        "dirty_status": run(["git", "status", "--short"]).splitlines(),
    }


def select_real_lamino_views(
    projections: np.ndarray,
    thetas: np.ndarray,
    *,
    max_views: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_views = int(projections.shape[0])
    if int(max_views) <= 0 or int(max_views) >= n_views:
        idx = np.arange(n_views, dtype=np.int32)
    else:
        idx = np.unique(np.rint(np.linspace(0, n_views - 1, int(max_views))).astype(np.int32))
    return projections[idx], thetas[idx], idx


def relative_l2(a: Any, b: Any) -> float:
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(arr_a - arr_b) / max(np.linalg.norm(arr_b), 1e-8))


def timed_repeats(
    *,
    name: str,
    fn: Callable[[], Any],
    repeats: int,
    warmups: int,
) -> tuple[Any, dict[str, Any]]:
    cold_start = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    cold_seconds = time.perf_counter() - cold_start
    for _ in range(max(0, int(warmups))):
        out = fn()
        jax.block_until_ready(out)
    times: list[float] = []
    for _ in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times, dtype=np.float64)
    return out, {
        "name": name,
        "cold_seconds": float(cold_seconds),
        "warmup_repeats": int(warmups),
        "measured_repeats": int(repeats),
        "median_seconds": float(np.median(arr)),
        "mean_seconds": float(np.mean(arr)),
        "min_seconds": float(np.min(arr)),
        "max_seconds": float(np.max(arr)),
        "times_seconds": [float(v) for v in times],
    }


__all__ = [
    "RealLaminoGpuMonitor",
    "append_real_lamino_csv",
    "real_lamino_commit_info",
    "real_lamino_json_safe",
    "relative_l2",
    "select_real_lamino_views",
    "timed_repeats",
    "update_real_lamino_status",
    "write_real_lamino_json",
]
