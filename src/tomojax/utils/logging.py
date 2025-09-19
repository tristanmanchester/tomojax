from __future__ import annotations

import logging
import math
import os
from typing import Iterable, Iterator, Optional


def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)s | %(message)s")


def log_jax_env() -> None:
    try:
        import jax
        logging.info("JAX backend: %s", jax.default_backend())
        logging.info("Devices: %s", jax.devices())
    except Exception:  # pragma: no cover
        logging.info("JAX not available for logging")


def _progress_enabled() -> bool:
    v = os.environ.get("TOMOJAX_PROGRESS", "0").lower()
    return v in ("1", "true", "yes", "on")


def progress_iter(iterable: Iterable, *, total: Optional[int] = None, desc: str = "") -> Iterator:
    """Yield elements from iterable, showing a progress bar if enabled.

    Enable by setting environment variable `TOMOJAX_PROGRESS=1` or calling CLIs with `--progress`.
    Uses tqdm if available; otherwise prints occasional step counters.
    """
    if not _progress_enabled():
        for x in iterable:
            yield x
        return
    try:
        from tqdm import tqdm  # type: ignore

        leave = os.environ.get("TOMOJAX_PROGRESS_LEAVE", "0").lower() in ("1","true","yes","on")
        for x in tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=leave):
            yield x
    except Exception:
        # Lightweight fallback: print at ~10% increments
        if total is None:
            for i, x in enumerate(iterable, 1):
                if i == 1 or i % 10 == 0:
                    print(f"{desc} step {i}", flush=True)
                yield x
        else:
            step = max(1, total // 10)
            for i, x in enumerate(iterable, 1):
                if i == 1 or i == total or i % step == 0:
                    print(f"{desc} {i}/{total}", flush=True)
                yield x


def format_duration(seconds: float | None) -> str:
    """Render a wall-clock duration as a compact human-readable string."""
    if seconds is None:
        return "-"
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(value):
        return "-"
    value = max(value, 0.0)
    if value < 1e-3:
        return f"{value * 1e6:.0f}µs"
    if value < 1.0:
        return f"{value * 1e3:.0f}ms" if value < 0.1 else f"{value:.2f}s"

    minutes, seconds_rem = divmod(value, 60.0)
    hours, minutes = divmod(minutes, 60.0)
    if hours >= 1.0:
        return f"{int(hours)}h{int(minutes):02d}m{seconds_rem:04.1f}s"
    if minutes >= 1.0:
        return f"{int(minutes)}m{seconds_rem:04.1f}s"
    return f"{seconds_rem:.1f}s"
